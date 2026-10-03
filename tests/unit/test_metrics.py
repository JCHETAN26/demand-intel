import numpy as np
import pandas as pd
import pytest

from demand_intel.data import M5Data
from demand_intel.forecasting.backtesting.harness import (
    FoldContext,
    Forecast,
    TruthCache,
    folds_for,
    run_backtest,
    stockout_mask,
)
from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, HORIZON
from demand_intel.forecasting.evaluation.hierarchy import LEVELS, build_hierarchy
from demand_intel.forecasting.evaluation.metrics import (
    FoldTruth,
    history_scales,
    per_series_metrics,
    pinball,
    summarize,
    wrmsse,
)
from demand_intel.models.statistical.baselines import MovingAverage, Naive, SeasonalNaive

SERIES = pd.DataFrame(
    {
        "id": ["A_1_001_CA_1", "A_1_001_TX_1", "B_1_001_CA_1", "B_1_001_TX_1"],
        "item_id": ["A_1_001", "A_1_001", "B_1_001", "B_1_001"],
        "dept_id": ["A_1", "A_1", "B_1", "B_1"],
        "cat_id": ["A", "A", "B", "B"],
        "store_id": ["CA_1", "TX_1", "CA_1", "TX_1"],
        "state_id": ["CA", "TX", "CA", "TX"],
    }
)


def test_hierarchy_levels_and_sums() -> None:
    hier = build_hierarchy(SERIES)
    counts = np.bincount(hier.level)
    # total, 2 states, 2 stores, 2 cats, 2 depts, 4, 4, 4, 4, 2 items, 4, 4
    assert counts.tolist() == [1, 2, 2, 2, 2, 4, 4, 4, 4, 2, 4, 4]
    assert len(LEVELS) == 12
    x = np.array([1.0, 2.0, 3.0, 4.0])
    agg = hier.S @ x
    assert agg[0] == 10
    assert sorted(agg[hier.level == 1].tolist()) == [4.0, 6.0]  # CA = 1+3, TX = 2+4


def test_history_scales_start_at_first_sale() -> None:
    hist = np.array([[0, 0, 0, 2, 4, 2], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    sq, ab = history_scales(hist)
    # diffs from first sale: (4-2), (2-4) -> sq 4, abs 2
    assert sq[0] == pytest.approx(4.0) and ab[0] == pytest.approx(2.0)
    assert np.isnan(sq[1])  # constant history: no usable scale
    assert np.isnan(sq[2])


def test_pinball_asymmetry() -> None:
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    assert pinball(np.array([10.0]), np.array([12.0]), 0.9)[0] == pytest.approx(0.2)


def _truth(y: np.ndarray, hist: np.ndarray) -> FoldTruth:
    prices = np.ones((len(y), 28), dtype=np.float32)
    return FoldTruth.build(y=y, history=hist, prices_28=prices, hierarchy=build_hierarchy(SERIES))


def test_perfect_forecast_scores_zero_and_bias_sign() -> None:
    rng = np.random.default_rng(0)
    hist = rng.poisson(3, size=(4, 100)).astype(float)
    y = rng.poisson(3, size=(4, HORIZON)).astype(float)
    truth = _truth(y, hist)
    score, levels = wrmsse(truth, y, 28)
    assert score == pytest.approx(0.0) and levels.shape == (12,)
    over = summarize(per_series_metrics(truth, y + 1, None, 28))
    assert over["bias"] > 0 and over["mae"] == pytest.approx(1.0)
    assert over["wape"] == pytest.approx(4 * 28 / y.sum())


def test_wrmsse_bottom_level_matches_direct_rmsse() -> None:
    rng = np.random.default_rng(1)
    hist = rng.poisson(4, size=(4, 120)).astype(float)
    y = rng.poisson(4, size=(4, HORIZON)).astype(float)
    point = np.full_like(y, 4.0)
    truth = _truth(y, hist)
    _, levels = wrmsse(truth, point, 28)
    direct = np.sqrt(((y - point) ** 2).mean(axis=1) / truth.scale_sq)
    w = truth.dollar_28 / truth.dollar_28.sum()
    assert levels[-1] == pytest.approx((direct * w).sum())
    ps = per_series_metrics(truth, point, None, 28)
    assert summarize(ps)["wrmsse_l12"] == pytest.approx((direct * w).sum())


def test_quantile_coverage_counts() -> None:
    hist = np.tile(np.arange(1, 11, dtype=float), (4, 10))
    y = np.full((4, HORIZON), 5.0)
    truth = _truth(y, hist)
    q = {0.1: np.full_like(y, 4.0), 0.5: np.full_like(y, 5.0), 0.9: np.full_like(y, 6.0)}
    s = summarize(per_series_metrics(truth, q[0.5], q, 28))
    assert s["cov_q10"] == 0.0 and s["cov_q50"] == 1.0 and s["cov_q90"] == 1.0
    assert s["cov_lt_q50"] == 0.0 and s["cov_lt_q90"] == 1.0
    assert s["cov_80"] == 1.0 and s["width_80"] == pytest.approx(2.0)


def _synthetic_m5(n_days: int = 1941) -> M5Data:
    t = np.arange(n_days)
    weekly = np.array([5, 6, 2, 1, 1, 2, 3])
    sales = np.stack([weekly[t % 7] * (i + 1) for i in range(4)]).astype(np.int32)
    cal = pd.DataFrame({"d": np.arange(1, 1970)})
    return M5Data(series=SERIES, sales=sales, calendar=cal, prices=np.ones((4, 1969), np.float32))


class _Spy:
    name = "spy"
    widths: list[int]

    def __init__(self) -> None:
        self.widths = []

    def forecast(self, ctx: FoldContext) -> Forecast:
        self.widths.append(ctx.data.sales.shape[1])
        return SeasonalNaive().forecast(ctx)


def test_backtest_hides_future_and_seasonal_naive_is_exact_on_weekly_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m5 = _synthetic_m5()
    attrs = pd.DataFrame(
        {
            "cat_id": SERIES["cat_id"],
            "store_id": SERIES["store_id"],
            "demand_class": "smooth",
            "volume_tercile": "mid",
        }
    )
    spy = _Spy()
    rows = np.arange(4)
    res = run_backtest(spy, m5, rows, "full", folds_for("all"), attrs, tmp_path, tmp_path / "fc")
    assert spy.widths == [f.origin for f in folds_for("all")]  # never sees past the origin
    overall = res[(res["slice"] == "all")]
    assert np.allclose(overall["wrmsse"], 0.0)
    assert (tmp_path / "spy__full.csv").exists()
    assert (tmp_path / "fc" / "spy" / "full" / "test.npz").exists()

    cache = TruthCache(m5)
    for model in (Naive(), MovingAverage()):
        r = run_backtest(
            model,
            m5,
            rows,
            "full",
            BACKTEST_FOLDS[:1],
            attrs,
            tmp_path,
            tmp_path,
            truth_cache=cache,
        )
        assert (r.loc[r["slice"] == "all", "wrmsse"] > 0).all()


def test_stockout_mask_clips_runs_to_window() -> None:
    fold = BACKTEST_FOLDS[0]
    runs = pd.DataFrame(
        {
            "series_idx": [0, 1, 1],
            "start_day": [fold.first_target - 5, fold.first_target + 3, 10],
            "end_day": [fold.first_target + 1, fold.first_target + 4, 30],
            "kind": ["stockout", "stockout", "stockout"],
        }
    )
    keep = stockout_mask(runs, np.array([0, 1]), fold)
    assert (~keep[0]).sum() == 2 and not keep[0, 0] and not keep[0, 1]
    assert (~keep[1]).sum() == 2 and not keep[1, 3] and not keep[1, 4]


def test_statistical_models_produce_valid_forecasts() -> None:
    from demand_intel.forecasting.backtesting.harness import truncate
    from demand_intel.models.statistical.classical import statistical_models

    m5 = _synthetic_m5()
    fold = BACKTEST_FOLDS[0]
    attrs = pd.DataFrame({"demand_class": ["smooth", "lumpy", "intermittent", "erratic"]})
    ctx = FoldContext(fold=fold, data=truncate(m5, fold.origin), rows=np.arange(4), attrs=attrs)
    models = statistical_models()
    for name in ("ses_seasonal", "tsb", "stat_class_selected", "ets"):
        fc = models[name]().forecast(ctx)
        assert fc.point.shape == (4, HORIZON)
        assert np.isfinite(fc.point).all() and (fc.point >= 0).all()
    ets = models["ets"]().forecast(ctx)
    assert ets.quantiles is not None
    qs = sorted(ets.quantiles)
    assert qs == [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    stacked = np.stack([ets.quantiles[q] for q in qs])
    assert (np.diff(stacked, axis=0) >= -1e-4).all()  # quantiles are ordered
