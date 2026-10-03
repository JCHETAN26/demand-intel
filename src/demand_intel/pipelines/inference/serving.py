"""Batch inference and the serving bundle (plan Phases 30-34).

The champion is retrained on all observed history (origin d_1941, the last labelled day)
and forecasts d_1942..d_1969, the 28 days that follow. The bundle written to
`data/serving/` is everything the API and web app read:

- series.parquet    series attributes (hierarchy, demand class)
- forecast.parquet  per series × day: mean and the 7 quantiles
- history.parquet   per series × day for the last 182 observed days: sales and price
- drivers.parquet   per series × day: SHAP contributions grouped into families (units)
- risk.parquet      per series: SCENARIO stock (7 days of recent mean sales), P90 of 8-day
                    demand and P95 of 28-day demand from copula sample paths, risk flags
- summary.json      origin, model, headline evaluation metrics, inventory calibration

Forecasts here have no actuals yet: they are the "production" forecasts a planner would
act on, and the monitoring job scores them once actuals arrive.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_intel.data import M5Data
from demand_intel.forecasting.backtesting.harness import FoldContext
from demand_intel.forecasting.backtesting.splits import HORIZON, Fold, day_to_date
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.inventory.policies.policies import lead_time_quantile
from demand_intel.models.lgbm.explain import contributions
from demand_intel.models.lgbm.model import LightGBMForecaster
from demand_intel.pipelines.features.build import BUCKETS, build_features

HISTORY_DAYS = 182
SCENARIO_COVER_DAYS = 7
RISK_LEAD_TIME = 7
DEFAULT_RHO = 0.3
QCOLS = [f"q{round(q * 100):02d}" for q in QUANTILES]


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    origin: int
    model: str
    series: int


def production_fold(m5: M5Data) -> Fold:
    return Fold("production", m5.n_days)


def _drivers(model_dir: Path, m5: M5Data, origin: int, families: list[str]) -> pd.DataFrame:
    frames = []
    rows = np.arange(len(m5.series))
    for lo, hi in BUCKETS:
        booster = lgb.Booster(model_file=str(model_dir / f"bucket_{hi}.txt"))
        days = np.arange(origin + lo, origin + hi + 1)
        ff = build_features(m5, rows, days, hi, families)
        fam = contributions(booster, ff.X).by_family(units=True).astype(np.float32)
        fam.insert(0, "d", ff.day.astype(np.int16))
        fam.insert(0, "series", ff.series_idx.astype(np.int32))
        frames.append(fam)
    return pd.concat(frames, ignore_index=True)


def build_bundle(
    m5: M5Data,
    out_dir: Path,
    model_root: Path,
    evaluation_dir: Path,
    profile: pd.DataFrame,
    forecaster: LightGBMForecaster | None = None,
) -> BundleInfo:
    """Train the champion on all history, forecast the next HORIZON days, write the bundle."""
    model = forecaster or LightGBMForecaster("lgbm", model_dir=model_root)
    fold = production_fold(m5)
    ctx = FoldContext(fold=fold, data=m5, rows=np.arange(len(m5.series)), population="full")
    fc = model.forecast(ctx)
    assert fc.quantiles is not None
    origin = fold.origin
    days = np.arange(origin + 1, origin + HORIZON + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    series = m5.series.copy()
    series["demand_class"] = profile["demand_class"].to_numpy()
    series["volume_tercile"] = profile["volume_tercile"].to_numpy()
    series.to_parquet(out_dir / "series.parquet", index=False)

    n = len(series)
    forecast = pd.DataFrame(
        {
            "series": np.repeat(np.arange(n, dtype=np.int32), HORIZON),
            "d": np.tile(days.astype(np.int16), n),
            "mean": fc.point.ravel(),
            **{
                c: fc.quantiles[q].ravel().astype(np.float32)
                for c, q in zip(QCOLS, QUANTILES, strict=True)
            },
        }
    )
    forecast.to_parquet(out_dir / "forecast.parquet", index=False)

    assert m5.prices is not None
    hist_days = np.arange(origin - HISTORY_DAYS + 1, origin + 1)
    history = pd.DataFrame(
        {
            "series": np.repeat(np.arange(n, dtype=np.int32), len(hist_days)),
            "d": np.tile(hist_days.astype(np.int16), n),
            "sales": m5.sales[:, hist_days - 1].ravel().astype(np.int16),
            "price": m5.prices[:, hist_days - 1].ravel().astype(np.float32),
        }
    )
    history.to_parquet(out_dir / "history.parquet", index=False)
    future_price = pd.DataFrame(
        {
            "series": np.repeat(np.arange(n, dtype=np.int32), HORIZON),
            "d": np.tile(days.astype(np.int16), n),
            "price": m5.prices[:, days - 1].ravel().astype(np.float32),
        }
    )
    future_price.to_parquet(out_dir / "future_price.parquet", index=False)

    rho = DEFAULT_RHO
    calib_path = evaluation_dir / "inventory_calibration.json"
    if calib_path.exists():
        entry = json.loads(calib_path.read_text()).get("rho", {}).get("L=7,SL=0.95")
        if isinstance(entry, dict):
            rho = float(entry["rho"])
    qarr = np.stack([fc.quantiles[q] for q in QUANTILES], axis=-1)
    stock = m5.sales[:, origin - 28 : origin].mean(axis=1) * SCENARIO_COVER_DAYS
    p90_lt = lead_time_quantile(qarr[:, : RISK_LEAD_TIME + 1], rho, 0.90, n_paths=300)
    p95_28 = lead_time_quantile(qarr, rho, 0.95, n_paths=300)
    pd.DataFrame(
        {
            "series": np.arange(n, dtype=np.int32),
            "scenario_stock": stock.astype(np.float32),
            "lead_time_p90": p90_lt.astype(np.float32),
            "demand28_p95": p95_28.astype(np.float32),
            "stockout_risk": p90_lt > stock,
            "excess_risk": stock > p95_28,
        }
    ).to_parquet(out_dir / "risk.parquet", index=False)

    model_dir = model_root / model.name / "full" / fold.name
    _drivers(model_dir, m5, origin, list(model.families)).to_parquet(
        out_dir / "drivers.parquet", index=False
    )

    summary: dict[str, Any] = {
        "origin_day": origin,
        "origin_date": day_to_date(origin).isoformat(),
        "forecast_dates": [
            day_to_date(int(days[0])).isoformat(),
            day_to_date(int(days[-1])).isoformat(),
        ],
        "model": model.name,
        "model_dir": str(model_dir),
        "families": list(model.families),
        "training_log": model.log,
        "generated_at": datetime.now(UTC).isoformat(),
        "series": n,
        "risk": {
            "definition": (
                f"SCENARIO stock = {SCENARIO_COVER_DAYS} days of the last 28 days' mean sales. "
                f"Stockout risk: P90 of demand over {RISK_LEAD_TIME + 1} days "
                "(lead time + review) exceeds stock. Excess risk: stock exceeds P95 of "
                f"28-day demand. Quantiles of sums come from copula sample paths, rho={rho}."
            ),
            "rho": rho,
        },
        "calendar": {
            int(d): day_to_date(int(d)).isoformat()
            for d in np.arange(origin - HISTORY_DAYS + 1, origin + HORIZON + 1)
        },
    }
    comparison = evaluation_dir / "model_comparison.csv"
    if comparison.exists():
        mc = pd.read_csv(comparison)
        row = mc[(mc["model"] == model.name) & (mc["population"] == "full")]
        if len(row):
            summary["evaluation"] = {
                k: float(row[k].iloc[0])
                for k in (
                    "test_wrmsse",
                    "test_wape",
                    "test_bias",
                    "test_wspl",
                    "backtest_wrmsse",
                    "backtest_wape",
                    "backtest_bias",
                )
                if k in row and pd.notna(row[k].iloc[0])
            }
    calib = evaluation_dir / "inventory_calibration.json"
    if calib.exists():
        summary["inventory_calibration"] = json.loads(calib.read_text())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return BundleInfo(path=out_dir, origin=origin, model=model.name, series=n)
