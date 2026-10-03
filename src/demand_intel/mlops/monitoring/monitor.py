"""Forecast, data and NLP monitoring (PROJECT_SPEC.md §11.4, plan Phase 33).

- Forecast: once actuals arrive for served forecasts, WAPE, bias and P90 coverage per
  category and store are compared with the same model's backtest values.
- Data: population stability index (PSI) of each model feature, reference = the training
  window, current = the window being forecast; plus missing-data rates.
- NLP: Jensen-Shannon divergence between the predicted event-type mix of a batch and the
  training label mix.

Each check returns a status (ok / warn / alert) using the thresholds in the spec.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Status = Literal["ok", "warn", "alert"]

PSI_WARN, PSI_ALERT = 0.10, 0.25
WAPE_WARN, WAPE_ALERT = 0.10, 0.25  # relative increase vs backtest
BIAS_WARN, BIAS_ALERT = 0.05, 0.10
COVERAGE_WARN = (0.85, 0.95)
COVERAGE_ALERT = (0.80, 0.97)
JS_WARN, JS_ALERT = 0.10, 0.20
MISSING_WARN, MISSING_ALERT = 0.005, 0.02
# Deterministic known-future inputs (spec §2.2) shift by construction between a training
# year and the next week; drift on them is not a data problem, so they are not scored.
KNOWN_FUTURE_CALENDAR = frozenset(
    {"wday", "mday", "month", "week", "year", "is_weekend", "event_2", "snap"}
)


@dataclass
class Check:
    name: str
    value: float
    status: Status
    detail: str = ""


@dataclass
class MonitorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> Status:
        levels = {c.status for c in self.checks}
        return "alert" if "alert" in levels else "warn" if "warn" in levels else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": [c.__dict__ for c in self.checks]}


def _level(value: float, warn: float, alert: float) -> Status:
    return "alert" if value > alert else "warn" if value > warn else "ok"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference; numeric features only."""
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, sample: int = 200_000, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num = [
        c
        for c in reference.columns
        if pd.api.types.is_numeric_dtype(reference[c]) and c not in KNOWN_FUTURE_CALENDAR
    ]
    ref = reference.iloc[rng.choice(len(reference), min(sample, len(reference)), replace=False)]
    cur = current.iloc[rng.choice(len(current), min(sample, len(current)), replace=False)]
    rows = []
    for col in num:
        value = psi(ref[col].to_numpy(dtype=float), cur[col].to_numpy(dtype=float))
        rows.append(
            {
                "feature": col,
                "psi": value,
                "status": _level(value, PSI_WARN, PSI_ALERT),
                "missing_ref": float(ref[col].isna().mean()),
                "missing_cur": float(cur[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def forecast_checks(
    y: np.ndarray,
    point: np.ndarray,
    q90: np.ndarray | None,
    groups: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[MonitorReport, pd.DataFrame]:
    """y, point, q90: (n, H). `groups` has cat_id/store_id per row; `baseline` holds the
    backtest WAPE per slice value (columns: slice, value, wape)."""
    rep = MonitorReport()
    rows = []
    for col in ("cat_id", "store_id"):
        for val, idx in groups.groupby(col, observed=True).indices.items():
            ys, ps = y[idx], point[idx]
            denom = ys.sum()
            wape = float(np.abs(ps - ys).sum() / denom) if denom else np.nan
            bias = float((ps - ys).sum() / denom) if denom else np.nan
            cov90 = float((ys <= q90[idx]).mean()) if q90 is not None else np.nan
            base = baseline[(baseline["slice"] == col) & (baseline["value"] == val)]["wape"]
            base_wape = float(base.iloc[0]) if len(base) else np.nan
            rel = wape / base_wape - 1 if base_wape == base_wape and base_wape else np.nan
            rows.append(
                {
                    "slice": col,
                    "value": val,
                    "wape": wape,
                    "backtest_wape": base_wape,
                    "wape_rel_change": rel,
                    "bias": bias,
                    "cov_q90": cov90,
                }
            )
            if rel == rel:
                rep.checks.append(
                    Check(
                        f"wape:{col}={val}",
                        rel,
                        _level(rel, WAPE_WARN, WAPE_ALERT),
                        f"WAPE {wape:.3f} vs backtest {base_wape:.3f}",
                    )
                )
            if bias == bias:
                rep.checks.append(
                    Check(f"bias:{col}={val}", bias, _level(abs(bias), BIAS_WARN, BIAS_ALERT))
                )
            if cov90 == cov90:
                lo_w, hi_w = COVERAGE_WARN
                lo_a, hi_a = COVERAGE_ALERT
                status: Status = (
                    "alert"
                    if not lo_a <= cov90 <= hi_a
                    else "warn"
                    if not lo_w <= cov90 <= hi_w
                    else "ok"
                )
                rep.checks.append(
                    Check(
                        f"coverage_p90:{col}={val}",
                        cov90,
                        status,
                        "share of actuals at or below P90",
                    )
                )
    return rep, pd.DataFrame(rows)


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def nlp_checks(
    train_mix: dict[str, float], batch_labels: list[str], schema_failures: int
) -> MonitorReport:
    rep = MonitorReport()
    counts = pd.Series(batch_labels).value_counts(normalize=True).to_dict()
    js = js_divergence(train_mix, {str(k): float(v) for k, v in counts.items()})
    rep.checks.append(
        Check(
            "nlp:event_mix_js",
            js,
            _level(js, JS_WARN, JS_ALERT),
            "Jensen-Shannon divergence (bits) vs training label mix",
        )
    )
    rate = schema_failures / max(len(batch_labels), 1)
    rep.checks.append(Check("nlp:schema_failures", rate, _level(rate, MISSING_WARN, MISSING_ALERT)))
    return rep
