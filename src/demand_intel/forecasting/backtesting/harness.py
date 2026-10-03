"""Rolling-origin backtest harness (PROJECT_SPEC.md §5.1).

Every model implements `Forecaster.forecast(ctx)`. The context exposes sales only up to the
fold origin, so a model cannot see the evaluation window. Calendar and prices cover the
full calendar because they are known-future inputs (spec §2.2).

Outputs per (model, population):
- `evaluation/backtest/<model>__<population>.csv`: metrics per fold × window × slice
- `data/forecasts/<model>/<population>/<fold>.npz`: the forecasts and per-series errors,
  kept for paired comparisons (§9.2) and the inventory simulation
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from demand_intel.data import M5Data
from demand_intel.forecasting.backtesting.splits import (
    BACKTEST_FOLDS,
    HORIZON,
    TEST_FOLD,
    Fold,
)
from demand_intel.forecasting.evaluation.hierarchy import build_hierarchy
from demand_intel.forecasting.evaluation.metrics import (
    WINDOWS,
    FoldTruth,
    per_series_metrics,
    summarize,
    wrmsse,
)

Population = Literal["full", "panel"]
SLICES = ("cat_id", "store_id", "demand_class", "volume_tercile")


@dataclass
class FoldContext:
    fold: Fold
    data: M5Data  # sales truncated at the origin; calendar/prices complete
    rows: np.ndarray  # series (row indices into data) to forecast
    weather: pd.DataFrame | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def origin(self) -> int:
        return self.fold.origin

    @property
    def history(self) -> np.ndarray:
        """(len(rows), origin) sales of the series to forecast."""
        hist: np.ndarray = self.data.sales[self.rows]
        return hist


@dataclass
class Forecast:
    point: np.ndarray  # (len(rows), HORIZON)
    quantiles: dict[float, np.ndarray] | None = None


class Forecaster(Protocol):
    name: str

    def forecast(self, ctx: FoldContext) -> Forecast: ...


def truncate(m5: M5Data, origin: int) -> M5Data:
    """A view of the data with sales observed only up to `origin` (inclusive)."""
    return replace(m5, sales=m5.sales[:, :origin])


def folds_for(which: Literal["backtest", "test", "all"]) -> tuple[Fold, ...]:
    return {"backtest": BACKTEST_FOLDS, "test": (TEST_FOLD,), "all": (*BACKTEST_FOLDS, TEST_FOLD)}[
        which
    ]


class TruthCache:
    """Model-independent evaluation inputs, built once per (population, fold)."""

    def __init__(self, m5: M5Data) -> None:
        self.m5 = m5
        self._cache: dict[tuple[int, int], FoldTruth] = {}

    def get(self, rows: np.ndarray, fold: Fold) -> FoldTruth:
        key = (hash(rows.tobytes()), fold.origin)
        if key not in self._cache:
            m5, o = self.m5, fold.origin
            assert m5.prices is not None
            self._cache[key] = FoldTruth.build(
                y=m5.sales[rows, o : o + HORIZON],
                history=m5.sales[rows, :o],
                prices_28=m5.prices[rows, o - 28 : o],
                hierarchy=build_hierarchy(m5.series.iloc[rows]),
            )
        return self._cache[key]


def stockout_mask(runs: pd.DataFrame, rows: np.ndarray, fold: Fold) -> np.ndarray:
    """(len(rows), HORIZON) True where a day is NOT inside a suspected stockout run."""
    keep = np.ones((len(rows), HORIZON), dtype=bool)
    pos = pd.Series(np.arange(len(rows)), index=rows)
    sel = runs[(runs["kind"] == "stockout") & runs["series_idx"].isin(rows)]
    for idx, start, end in zip(
        sel["series_idx"].to_numpy(),
        sel["start_day"].to_numpy(),
        sel["end_day"].to_numpy(),
        strict=True,
    ):
        lo = max(int(start), fold.first_target) - fold.first_target
        hi = min(int(end), fold.last_target) - fold.first_target
        if lo <= hi:
            keep[pos[idx], lo : hi + 1] = False
    return keep


def evaluate_forecast(
    model: str,
    population: str,
    fold: Fold,
    fc: Forecast,
    truth: FoldTruth,
    attrs: pd.DataFrame,
    mask: np.ndarray | None,
    seconds: float,
) -> list[dict[str, Any]]:
    base = {"model": model, "population": population, "fold": fold.name, "seconds": seconds}
    rows: list[dict[str, Any]] = []
    for h in WINDOWS:
        ps = per_series_metrics(truth, fc.point, fc.quantiles, h)
        overall, per_level = wrmsse(truth, fc.point, h)
        rows.append(
            {
                **base,
                "window": h,
                "slice": "all",
                "value": "all",
                "wrmsse": overall,
                **{f"wrmsse_level{i + 1}": v for i, v in enumerate(per_level)},
                **summarize(ps),
            }
        )
        joined = pd.concat([ps.reset_index(drop=True), attrs.reset_index(drop=True)], axis=1)
        for col in SLICES:
            for val, grp in joined.groupby(col, observed=True):
                rows.append({**base, "window": h, "slice": col, "value": val, **summarize(grp)})
        if mask is not None:
            masked = per_series_metrics(truth, fc.point, fc.quantiles, h, mask=mask)
            rows.append(
                {
                    **base,
                    "window": h,
                    "slice": "stockouts_masked",
                    "value": "all",
                    **summarize(masked),
                }
            )
    # Per horizon day (window of one day at offset k)
    for k in range(HORIZON):
        y = truth.y[:, k]
        e = fc.point[:, k] - y
        rows.append(
            {
                **base,
                "window": 0,
                "slice": "horizon_day",
                "value": k + 1,
                "wape": float(np.abs(e).sum() / y.sum()) if y.sum() > 0 else np.nan,
                "bias": float(e.sum() / y.sum()) if y.sum() > 0 else np.nan,
            }
        )
    return rows


def run_backtest(
    forecaster: Forecaster,
    m5: M5Data,
    rows: np.ndarray,
    population: Population,
    folds: Iterable[Fold],
    attrs: pd.DataFrame,
    eval_dir: Path,
    forecast_dir: Path,
    stockout_runs: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    truth_cache: TruthCache | None = None,
) -> pd.DataFrame:
    """Run `forecaster` over `folds` for the series `rows` and write results."""
    cache = truth_cache or TruthCache(m5)
    out_rows: list[dict[str, Any]] = []
    fdir = forecast_dir / forecaster.name / population
    fdir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        ctx = FoldContext(fold=fold, data=truncate(m5, fold.origin), rows=rows, weather=weather)
        t0 = time.perf_counter()
        fc = forecaster.forecast(ctx)
        seconds = time.perf_counter() - t0
        if fc.point.shape != (len(rows), HORIZON):
            raise ValueError(f"{forecaster.name}: forecast shape {fc.point.shape}")
        fc.point = np.clip(fc.point, 0, None)
        truth = cache.get(rows, fold)
        mask = stockout_mask(stockout_runs, rows, fold) if stockout_runs is not None else None
        out_rows += evaluate_forecast(
            forecaster.name, population, fold, fc, truth, attrs, mask, seconds
        )
        ps28 = per_series_metrics(truth, fc.point, fc.quantiles, HORIZON)
        arrays: dict[str, Any] = {
            "rows": rows,
            "point": fc.point.astype(np.float32),
            "rmsse": ps28["rmsse"].to_numpy(),
            "abs_err": ps28["abs_err"].to_numpy(),
            "weight": ps28["weight"].to_numpy(),
        }
        for t, q in (fc.quantiles or {}).items():
            arrays[f"q{round(t * 100):02d}"] = q.astype(np.float32)
        np.savez_compressed(fdir / f"{fold.name}.npz", **arrays)
    result = pd.DataFrame(out_rows)
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"{forecaster.name}__{population}.csv"
    if path.exists():  # keep results for folds not re-run in this invocation
        old = pd.read_csv(path)
        old = old[~old["fold"].isin(result["fold"].unique())]
        result = pd.concat([old, result], ignore_index=True)
    result.to_csv(path, index=False)
    return result
