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
- summary.json      origin, served model vs registry champion (and any fallback reason),
                    headline evaluation metrics, inventory calibration

Forecasts here have no actuals yet: they are the "production" forecasts a planner would
act on, and the monitoring job scores them once actuals arrive.
"""

import json
from collections.abc import Callable
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
from demand_intel.forecasting.evaluation.metrics import QCOLS, QUANTILES
from demand_intel.inventory.policies.policies import lead_time_quantile
from demand_intel.models.lgbm.explain import contributions
from demand_intel.models.lgbm.model import LightGBMForecaster
from demand_intel.pipelines.features.build import BUCKETS, build_features

HISTORY_DAYS = 182
RISK_LEAD_TIME = 7
RISK_SERVICE = 0.95
STOCKOUT_LEVEL = 0.90  # flag when P(stockout) > 10%, twice the 5% target
EXCESS_RATIO = 1.5
DEFAULT_RHO = 0.3
RISK_DEFINITION = (
    "SCENARIO stock = what a simple replenishment rule would hold: the last 28 days' mean "
    "sales over {k} days (lead time + review) plus a 95% normal safety stock. Stockout risk: "
    "the forecast gives more than a 10% chance that {k}-day demand exceeds that stock (twice "
    "the 5% target). Excess risk: the stock is more than {r}x the forecast's P95 need. "
    "Quantiles of sums come from copula sample paths "
    "(rho={rho})."
)


def calibrated_rho(evaluation_dir: Path) -> float:
    path = evaluation_dir / "inventory_calibration.json"
    if path.exists():
        entry = json.loads(path.read_text()).get("rho", {}).get("L=7,SL=0.95")
        if isinstance(entry, dict):
            return float(entry["rho"])
    return DEFAULT_RHO


def compute_risk(qarr: np.ndarray, recent_sales: np.ndarray, rho: float) -> pd.DataFrame:
    """Per-series risk against the simple-rule SCENARIO stock (spec §8: scenario, not data)."""
    k = RISK_LEAD_TIME + 1
    mean, sd = recent_sales.mean(axis=1), recent_sales.std(axis=1)
    stock = mean * k + 1.645 * sd * np.sqrt(k)
    need = lead_time_quantile(qarr[:, :k], rho, RISK_SERVICE, n_paths=400)
    p90 = lead_time_quantile(qarr[:, :k], rho, STOCKOUT_LEVEL, n_paths=400, seed=1)
    return pd.DataFrame(
        {
            "series": np.arange(len(stock), dtype=np.int32),
            "scenario_stock": stock.astype(np.float32),
            "lead_time_p90": p90.astype(np.float32),
            "lead_time_p95": need.astype(np.float32),
            "stockout_risk": p90 > stock + 1e-9,
            "excess_risk": stock > EXCESS_RATIO * np.maximum(need, 1.0),
        }
    )


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    origin: int
    model: str
    series: int


FALLBACK_MODEL = "lgbm"


class ServingSelectionError(ValueError):
    """The requested or registered model cannot be published and fallback is not allowed."""


@dataclass(frozen=True)
class ServingChoice:
    """Which forecaster the bundle publishes, and why.

    The bundle needs quantile maps and TreeSHAP drivers, so only LightGBM models can be
    published. When the registry champion is something else, the choice records the
    fallback openly instead of hiding it: `served` differs from `champion`, `fallback` is
    true and `reason` says why. All of it is written to summary.json and exposed by the API.
    """

    served: str
    champion: str
    fallback: bool
    reason: str | None

    def provenance(self) -> dict[str, Any]:
        return {
            "registry_champion": self.champion,
            "fallback": self.fallback,
            "fallback_reason": self.reason,
        }


def choose_serving_model(
    champion: str,
    servable: Callable[[str], bool],
    requested: str = "",
    allow_fallback: bool = True,
) -> ServingChoice:
    """Pick the model to publish: an explicit request, else the champion, else the fallback.

    Anything other than the registry champion is a fallback and carries a reason. With
    `allow_fallback=False` a non-servable champion raises instead.
    """
    if requested:
        if not servable(requested):
            raise ServingSelectionError(f"{requested} is not a LightGBM model and cannot be served")
        if requested == champion:
            return ServingChoice(requested, champion, False, None)
        return ServingChoice(
            requested,
            champion,
            True,
            f"explicit override (--model {requested}); registry champion is {champion}",
        )
    if servable(champion):
        return ServingChoice(champion, champion, False, None)
    reason = (
        f"registry champion {champion} is not a LightGBM model (the bundle needs quantile maps "
        f"and TreeSHAP drivers); serving unpromoted {FALLBACK_MODEL} instead"
    )
    if not allow_fallback:
        raise ServingSelectionError(reason)
    return ServingChoice(FALLBACK_MODEL, champion, True, reason)


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
    choice: ServingChoice | None = None,
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

    rho = calibrated_rho(evaluation_dir)
    qarr = np.stack([fc.quantiles[q] for q in QUANTILES], axis=-1)
    compute_risk(qarr, m5.sales[:, origin - 28 : origin], rho).to_parquet(
        out_dir / "risk.parquet", index=False
    )

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
        # Without a choice the bundle was not resolved against the registry: say so (None).
        **(
            choice.provenance()
            if choice
            else {"registry_champion": None, "fallback": None, "fallback_reason": None}
        ),
        "model_dir": str(model_dir),
        "families": list(model.families),
        "training_log": model.log,
        "generated_at": datetime.now(UTC).isoformat(),
        "series": n,
        "risk": {
            "definition": RISK_DEFINITION.format(k=RISK_LEAD_TIME + 1, r=EXCESS_RATIO, rho=rho),
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


LAB_FILES = {
    "model_comparison": "model_comparison.csv",
    "forecast_ablation": "forecast_ablation.csv",
    "paired_tests": "paired_tests.csv",
    "probabilistic_calibration": "probabilistic_calibration.csv",
    "nlp_results": "nlp_results.csv",
    "inventory_simulation": "inventory_simulation.csv",
    "inventory_policy_differences": "inventory_policy_differences.csv",
    "multimodal_ablation": "multimodal_ablation.csv",
}


def write_extras(
    out_dir: Path,
    evaluation_dir: Path,
    recalls: pd.DataFrame | None,
    event_predictions: pd.DataFrame | None,
    origin_date: str,
) -> None:
    """Model Lab payload (lab.json) and recent recall events (events.parquet) for the UI.

    Only evaluation artifacts are copied; nothing is recomputed or typed in here.
    """
    lab: dict[str, Any] = {}
    for key, name in LAB_FILES.items():
        path = evaluation_dir / name
        if path.exists():
            lab[key] = json.loads(pd.read_csv(path).to_json(orient="records"))
    for key, name in (
        ("registry", "registry.json"),
        ("monitoring", "monitoring_report.json"),
        ("nlp_dataset", "nlp_dataset.json"),
    ):
        path = evaluation_dir / name
        if path.exists():
            lab[key] = json.loads(path.read_text())
    (out_dir / "lab.json").write_text(json.dumps(lab, default=str))

    if recalls is None:
        return
    end = pd.Timestamp(origin_date)
    recent = recalls[
        (recalls["report_date"] > end - pd.Timedelta(days=365)) & (recalls["report_date"] <= end)
    ]
    from demand_intel.models.nlp.labeling import weak_label

    events = pd.DataFrame(
        {
            "doc_id": recent["doc_id"].to_numpy(),
            "report_date": recent["report_date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "reason": recent["reason_for_recall"].to_numpy(),
            "product": recent["product_description"].str.slice(0, 300).to_numpy(),
            "distribution": recent["distribution_pattern"].fillna("").to_numpy(),
            "states": recent["states"].map(list).to_numpy(),
            "nationwide": recent["nationwide"].to_numpy(),
            "fda_severity": recent["severity"].to_numpy(),
        }
    )
    if event_predictions is not None and len(event_predictions):
        preds = event_predictions.set_index("doc_id")
        events["event_type"] = events["doc_id"].map(preds["event_pred"])
        events["confidence"] = events["doc_id"].map(preds["event_conf"])
        events["model"] = str(event_predictions.attrs.get("model", "nlp-champion"))
    missing = events.get("event_type", pd.Series(index=events.index, dtype=object)).isna()
    if missing.any():
        rules = events.loc[missing, "reason"].map(weak_label)
        events.loc[missing, "event_type"] = rules.map(lambda w: w.label)
        events.loc[missing, "confidence"] = rules.map(
            lambda w: 0.3 if not w.fired else (0.6 if w.conflict else 0.9)
        )
        events.loc[missing, "model"] = "rules-v1"
    events.sort_values("report_date", ascending=False).to_parquet(
        out_dir / "events.parquet", index=False
    )
