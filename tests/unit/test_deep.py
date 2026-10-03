import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demand_intel.forecasting.backtesting.harness import FoldContext, truncate  # noqa: E402
from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, HORIZON  # noqa: E402
from demand_intel.forecasting.evaluation.metrics import QUANTILES  # noqa: E402
from demand_intel.models.deep_learning.data import ENC, SequenceArrays, window  # noqa: E402
from demand_intel.models.deep_learning.lstm import Seq2SeqLSTM  # noqa: E402
from demand_intel.models.deep_learning.train import (  # noqa: E402
    DeepForecaster,
    TrainConfig,
    pinball_loss,
)
from tests.conftest import synthetic_m5  # noqa: E402


def test_window_shapes_and_no_future_sales() -> None:
    m5 = truncate(synthetic_m5(), 1000)
    arr = SequenceArrays.build(m5, np.arange(4))
    s = window(arr, 0, 1000, with_target=False)
    assert s["past"].shape == (ENC, 2)
    assert s["known"].shape == (ENC + HORIZON, arr.known.shape[2])
    assert "y" not in s
    with pytest.raises(ValueError, match="past the observed"):
        window(arr, 0, 1000, with_target=True)  # its targets would be future sales


def test_quantile_head_is_monotone_and_pinball_is_zero_at_truth() -> None:
    m5 = truncate(synthetic_m5(), 1000)
    arr = SequenceArrays.build(m5, np.arange(4))
    model = Seq2SeqLSTM(arr, hidden=8)
    batch = {
        k: torch.stack([window(arr, i, 900, True)[k] for i in range(4)])
        for k in window(arr, 0, 900, True)
    }
    mean, q = model(batch)
    assert mean.shape == (4, HORIZON) and q.shape == (4, HORIZON, len(QUANTILES))
    assert (q.diff(dim=-1) >= 0).all() and (mean >= 0).all()
    y = torch.rand(2, HORIZON)
    assert float(pinball_loss(y[..., None].expand(-1, -1, len(QUANTILES)), y)) == pytest.approx(0.0)


def test_lstm_forecaster_runs_end_to_end() -> None:
    m5 = synthetic_m5(n_items=3)
    fold = BACKTEST_FOLDS[-1]
    cfg = TrainConfig(train_days=200, samples_per_epoch=600, batch_size=64, epochs=2, threads=2)
    model = DeepForecaster("lstm_test", lambda a: Seq2SeqLSTM(a, hidden=8), cfg)
    ctx = FoldContext(fold=fold, data=truncate(m5, fold.origin), rows=np.arange(len(m5.series)))
    fc = model.forecast(ctx)
    assert fc.point.shape == (len(m5.series), HORIZON)
    assert np.isfinite(fc.point).all() and (fc.point >= 0).all()
    assert fc.quantiles is not None and len(fc.quantiles) == len(QUANTILES)
    assert {r["epoch"] for r in model.log} <= {0, 1}


def test_tft_forward_shapes_weights_and_causal_attention() -> None:
    from demand_intel.models.deep_learning.tft import TemporalFusionTransformer

    m5 = truncate(synthetic_m5(), 1000)
    arr = SequenceArrays.build(m5, np.arange(4))
    model = TemporalFusionTransformer(arr, d=16, heads=2)
    batch = {
        k: torch.stack([window(arr, i, 900, True)[k] for i in range(4)])
        for k in window(arr, 0, 900, True)
    }
    mean, q = model(batch)
    assert mean.shape == (4, HORIZON) and q.shape == (4, HORIZON, len(QUANTILES))
    w = model.last_weights
    assert torch.allclose(w["encoder"].sum(-1), torch.ones(4), atol=1e-5)
    assert w["static"].shape == (4, 5)
    # decoder step k attends only to positions <= ENC + k
    att = w["attention"]
    assert att.shape == (4, ENC + HORIZON)
    # changing a future decoder input must not change earlier forecasts
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["known"][:, -1, :] += 5.0
    model.eval()
    with torch.no_grad():
        m1, _ = model(batch)
        m2, _ = model(b2)
    assert torch.allclose(m1[:, :-1], m2[:, :-1], atol=1e-5)
