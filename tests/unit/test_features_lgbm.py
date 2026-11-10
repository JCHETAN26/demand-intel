from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from demand_intel.forecasting.backtesting.harness import FoldContext, truncate
from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, HORIZON
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.models.lgbm.model import LightGBMForecaster, QuantileMap
from demand_intel.pipelines.features.build import BUCKETS, build_features
from tests.conftest import synthetic_m5


@pytest.mark.parametrize("hb", [b[1] for b in BUCKETS])
def test_sales_features_do_not_depend_on_data_after_the_anchor(hb: int) -> None:
    m5 = synthetic_m5()
    rows = np.arange(len(m5.series))
    days = np.array([900, 950, 1000])
    full = build_features(m5, rows, days, hb, ("sales",)).X
    # Corrupt everything after the earliest anchor + the remaining anchors: values must not move
    anchor_max = days.max() - hb
    cut = replace(m5, sales=m5.sales[:, :anchor_max])
    pd.testing.assert_frame_equal(full, build_features(cut, rows, days, hb, ("sales",)).X)


def test_anchor_beyond_observed_data_raises() -> None:
    m5 = truncate(synthetic_m5(), 1000)
    with pytest.raises(ValueError, match="leak"):
        build_features(m5, np.arange(4), np.array([1010]), 7, ("sales",))


def test_feature_families_and_eligibility() -> None:
    m5 = synthetic_m5()
    rows = np.arange(len(m5.series))
    ff = build_features(
        m5,
        rows,
        np.arange(500, 510),
        7,
        ("sales", "calendar", "snap", "price", "hierarchy"),
        with_target=True,
    )
    assert len(ff.X) == len(rows) * 10
    assert {"lag_7", "mean_28", "wday", "snap", "price", "item_id", "event_name"} <= set(
        ff.X.columns
    )
    assert ff.X["item_id"].dtype == "category"
    assert ff.y is not None and len(ff.y) == len(ff.X)
    # rows with no price are not eligible
    assert not ff.eligible[np.isnan(ff.X["price"].to_numpy())].any()


def test_quantile_map_is_calibrated_in_sample_and_ordered() -> None:
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.1, 10, 50_000)
    y = rng.poisson(mu).astype(float)
    qmap = QuantileMap.fit(mu, y)
    q = qmap.apply(mu)
    for tau in (0.1, 0.5, 0.9):
        # discrete calibration: P(y < q) <= tau <= P(y <= q), up to sampling noise
        assert (y < q[tau]).mean() <= tau + 0.02
        assert (y <= q[tau]).mean() >= tau - 0.02
    stacked = np.stack([q[t] for t in QUANTILES])
    assert (np.diff(stacked, axis=0) >= 0).all()


def test_lightgbm_forecaster_end_to_end() -> None:
    m5 = synthetic_m5()
    fold = BACKTEST_FOLDS[-1]
    params = {
        "objective": "tweedie",
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "num_threads": 2,
    }
    model = LightGBMForecaster("lgbm_test", train_days=120, params=params, max_rounds=60)
    ctx = FoldContext(fold=fold, data=truncate(m5, fold.origin), rows=np.arange(len(m5.series)))
    fc = model.forecast(ctx)
    assert fc.point.shape == (len(m5.series), HORIZON)
    assert (fc.point >= 0).all()
    assert fc.quantiles is not None
    truth = m5.sales[:, fold.origin : fold.origin + HORIZON]
    # beats a flat zero forecast comfortably on Poisson data with a real signal
    assert np.abs(fc.point - truth).mean() < 0.8 * np.abs(truth).mean()
    assert {r["bucket"] for r in model.log} == {7, 14, 28}


def test_bias_correction_moves_mean_toward_calibration_window() -> None:
    m5 = synthetic_m5()
    fold = BACKTEST_FOLDS[-1]
    params = {
        "objective": "tweedie",
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "num_threads": 2,
    }
    ctx = FoldContext(fold=fold, data=truncate(m5, fold.origin), rows=np.arange(len(m5.series)))
    plain = LightGBMForecaster("p", train_days=120, params=params, max_rounds=40).forecast(ctx)
    model = LightGBMForecaster(
        "bc", train_days=120, params=params, max_rounds=40, bias_correct=True
    )
    corrected = model.forecast(ctx)
    factors = [f for r in model.log for f in r["bias_factors"].values()]
    assert factors and all(0.9 <= f <= 1.1 for f in factors)
    assert corrected.quantiles is not None and plain.quantiles is not None
    np.testing.assert_allclose(
        corrected.quantiles[0.5], plain.quantiles[0.5]
    )  # quantiles untouched


def test_quantile_map_scales_with_volume_beyond_the_top_bin() -> None:
    rng = np.random.default_rng(1)
    mu = np.concatenate([rng.uniform(0.1, 3, 45_000), rng.uniform(5, 100, 5_000)])
    y = rng.poisson(mu).astype(float)
    qmap = QuantileMap.fit(mu, y)
    q = qmap.apply(np.array([60.0, 90.0]))
    # bands bracket the forecast for a high-volume item (the bug: P90 far below the mean)
    assert q[0.1][0] < 60 < q[0.9][0] and q[0.9][1] > q[0.9][0]
    assert 0.8 < q[0.5][0] / 60 < 1.2
