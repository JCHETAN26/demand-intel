"""Statistical forecasters F1/F2 (PROJECT_SPEC.md §5.3) via statsforecast.

All fit on the last `HISTORY_DAYS` of each series (two years), which keeps local models
affordable over 30k series while covering every weekly/annual pattern they can use.

Scope (spec §5.3: "not every model needs to run over all 30K series"):
- FULL: seasonal exponential smoothing, Croston (optimized), TSB, and the class-selected
  combination (smoothing for smooth/erratic series, TSB for intermittent/lumpy ones).
- PANEL: AutoETS with prediction intervals, and a restricted AutoARIMA.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from demand_intel.forecasting.backtesting.harness import FoldContext, Forecast
from demand_intel.forecasting.backtesting.splits import HORIZON

HISTORY_DAYS = 728
N_JOBS = 4
# statsforecast interval levels → our quantiles (central intervals)
LEVELS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}


def _long(history: np.ndarray) -> pd.DataFrame:
    n, t = history.shape
    return pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(n), t),
            "ds": np.tile(np.arange(t), n),
            "y": history.astype(np.float64).ravel(),
        }
    )


def _wide(out: pd.DataFrame, column: str, n: int) -> np.ndarray:
    out = out.sort_values(["unique_id", "ds"])
    arr: np.ndarray = out[column].to_numpy(dtype=np.float32).reshape(n, HORIZON)
    clipped: np.ndarray = np.clip(arr, 0, None)
    return clipped


def _run(models: list[Any], history: np.ndarray, intervals: bool) -> pd.DataFrame:
    from statsforecast import StatsForecast

    sf = StatsForecast(models=models, freq=1, n_jobs=N_JOBS)
    kwargs: dict[str, Any] = {"level": list(LEVELS)} if intervals else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out: pd.DataFrame = sf.forecast(df=_long(history[:, -HISTORY_DAYS:]), h=HORIZON, **kwargs)
    return out.reset_index() if "unique_id" not in out.columns else out


def _quantiles(out: pd.DataFrame, alias: str, n: int, point: np.ndarray) -> dict[float, np.ndarray]:
    q: dict[float, np.ndarray] = {0.5: point}
    for level, (lo, hi) in LEVELS.items():
        q[lo] = _wide(out, f"{alias}-lo-{level}", n)
        q[hi] = _wide(out, f"{alias}-hi-{level}", n)
    return dict(sorted(q.items()))


@dataclass
class StatsModel:
    """One statsforecast model as a `Forecaster`."""

    name: str
    factory: Any  # () -> statsforecast model
    alias: str
    intervals: bool = False
    history_days: int = HISTORY_DAYS

    def forecast(self, ctx: FoldContext) -> Forecast:
        n = len(ctx.rows)
        out = _run([self.factory()], ctx.history[:, -self.history_days :], self.intervals)
        point = _wide(out, self.alias, n)
        return Forecast(
            point=point,
            quantiles=_quantiles(out, self.alias, n, point) if self.intervals else None,
        )


@dataclass
class ClassSelected:
    """F1 combination: per-series model chosen by demand class (fixed rule, spec §5.3)."""

    name: str = "stat_class_selected"
    smooth_classes: tuple[str, ...] = ("smooth", "erratic")
    extras: dict[str, Any] = field(default_factory=dict)

    def forecast(self, ctx: FoldContext) -> Forecast:
        from statsforecast.models import TSB, SeasonalExponentialSmoothingOptimized

        if ctx.attrs is None:
            raise ValueError("stat_class_selected needs series attributes (demand_class)")
        n = len(ctx.rows)
        out = _run(
            [SeasonalExponentialSmoothingOptimized(season_length=7), TSB(alpha_d=0.2, alpha_p=0.2)],
            ctx.history,
            intervals=False,
        )
        ses, tsb = _wide(out, "SeasESOpt", n), _wide(out, "TSB", n)
        smooth = ctx.attrs["demand_class"].isin(self.smooth_classes).to_numpy()
        return Forecast(point=np.where(smooth[:, None], ses, tsb))


def statistical_models() -> dict[str, Any]:
    from statsforecast.models import (
        TSB,
        AutoARIMA,
        AutoETS,
        CrostonOptimized,
        SeasonalExponentialSmoothingOptimized,
    )

    return {
        "ses_seasonal": lambda: StatsModel(
            "ses_seasonal",
            lambda: SeasonalExponentialSmoothingOptimized(season_length=7),
            "SeasESOpt",
        ),
        "croston": lambda: StatsModel("croston", CrostonOptimized, "CrostonOptimized"),
        "tsb": lambda: StatsModel("tsb", lambda: TSB(alpha_d=0.2, alpha_p=0.2), "TSB"),
        "stat_class_selected": ClassSelected,
        "ets": lambda: StatsModel(
            "ets", lambda: AutoETS(season_length=7), "AutoETS", intervals=True
        ),
        "arima": lambda: StatsModel(
            "arima",
            lambda: AutoARIMA(
                season_length=7,
                max_p=2,
                max_q=2,
                max_P=1,
                max_Q=1,
                max_order=4,
                stepwise=True,
                approximation=True,
            ),
            "AutoARIMA",
            intervals=True,
            history_days=364,  # one year: ARIMA search cost grows with length
        ),
    }
