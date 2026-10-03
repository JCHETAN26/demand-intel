"""Forecast metrics (PROJECT_SPEC.md §5.4, §5.5).

Conventions:
- Arrays are (n_series, horizon). A window `h` uses the first `h` horizon days.
- RMSSE scale (per series) = mean squared one-step difference of the in-sample history
  from the series' first nonzero sale up to the origin (M5 definition). The pinball scale
  uses mean absolute differences. Series without a usable scale (no sale before the origin
  or a constant history) get NaN and are left out of scaled metrics.
- WRMSSE weights = dollar sales over the 28 days before the origin, normalized within each
  of the 12 levels; the score is the mean over levels of the weighted RMSSE.
- Quantile metrics are computed at the bottom level (item × store), where the models
  produce quantiles; see spec §5.5 (v0.3).
- Demand is discrete, so a calibrated τ-quantile satisfies P(y < q) <= τ <= P(y <= q)
  rather than P(y <= q) = τ. Both shares are reported (`cov_lt_qXX`, `cov_qXX`).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_intel.data import first_sale_index
from demand_intel.forecasting.evaluation.hierarchy import Hierarchy

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = [f"q{round(q * 100):02d}" for q in QUANTILES]  # column names in files
WINDOWS = (7, 14, 28)


def history_scales(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(squared, absolute) one-step-difference scales of each row of `history`."""
    first = first_sale_index(history)
    n_t = history.shape[1]
    sq = np.zeros(len(history))
    ab = np.zeros(len(history))
    cnt = np.zeros(len(history))
    # Chunk over rows: aggregate histories can be large.
    for lo in range(0, len(history), 4096):
        h = history[lo : lo + 4096].astype(np.float64)
        diff = np.diff(h, axis=1)
        valid = np.arange(n_t - 1)[None, :] >= first[lo : lo + 4096, None]
        sq[lo : lo + 4096] = np.where(valid, diff**2, 0.0).sum(axis=1)
        ab[lo : lo + 4096] = np.where(valid, np.abs(diff), 0.0).sum(axis=1)
        cnt[lo : lo + 4096] = valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_sq = np.where(cnt > 0, sq / cnt, np.nan)
        scale_abs = np.where(cnt > 0, ab / cnt, np.nan)
    scale_sq[scale_sq <= 0] = np.nan
    scale_abs[scale_abs <= 0] = np.nan
    return scale_sq, scale_abs


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    loss: np.ndarray = np.maximum(tau * diff, (tau - 1) * diff)
    return loss


@dataclass
class FoldTruth:
    """Everything about a fold's evaluation set that does not depend on the model."""

    y: np.ndarray  # (n, H) actuals
    history: np.ndarray  # (n, origin) in-sample sales
    dollar_28: np.ndarray  # (n,) dollar sales over the 28 days before the origin
    scale_sq: np.ndarray
    scale_abs: np.ndarray
    hierarchy: Hierarchy
    agg_scale_sq: np.ndarray  # scales of every aggregate row
    agg_weight: np.ndarray  # level-normalized weights of every aggregate row

    @classmethod
    def build(
        cls,
        y: np.ndarray,
        history: np.ndarray,
        prices_28: np.ndarray,
        hierarchy: Hierarchy,
    ) -> "FoldTruth":
        scale_sq, scale_abs = history_scales(history)
        dollar = np.nansum(history[:, -28:] * prices_28, axis=1)
        agg_hist = hierarchy.S @ history.astype(np.float32)
        agg_scale_sq, _ = history_scales(np.asarray(agg_hist))
        agg_dollar = hierarchy.S @ dollar
        level_tot = np.bincount(hierarchy.level, weights=agg_dollar)
        agg_weight = agg_dollar / level_tot[hierarchy.level]
        return cls(
            y=y,
            history=history,
            dollar_28=dollar,
            scale_sq=scale_sq,
            scale_abs=scale_abs,
            hierarchy=hierarchy,
            agg_scale_sq=agg_scale_sq,
            agg_weight=agg_weight,
        )

    @property
    def bottom_weight(self) -> np.ndarray:
        """Level-12 (item × store) weights in bottom-series order."""
        total = self.dollar_28.sum()
        return self.dollar_28 / total if total > 0 else np.zeros_like(self.dollar_28)


def wrmsse(truth: FoldTruth, point: np.ndarray, h: int) -> tuple[float, np.ndarray]:
    """Overall WRMSSE for window h and the per-level weighted RMSSE (12 values)."""
    err = np.asarray(truth.hierarchy.S @ (truth.y[:, :h] - point[:, :h]).astype(np.float64))
    mse = (err**2).mean(axis=1)
    rmsse = np.sqrt(mse / truth.agg_scale_sq)
    contrib = np.where(np.isnan(rmsse), 0.0, rmsse * truth.agg_weight)
    per_level = np.bincount(truth.hierarchy.level, weights=contrib)
    return float(per_level.mean()), per_level


def per_series_metrics(
    truth: FoldTruth,
    point: np.ndarray,
    quantiles: dict[float, np.ndarray] | None,
    h: int,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-series sums and scaled errors for window h; `mask` (n, H) excludes days."""
    y = truth.y[:, :h].astype(np.float64)
    p = point[:, :h].astype(np.float64)
    m = np.ones_like(y, dtype=bool) if mask is None else mask[:, :h]
    days = m.sum(axis=1)
    err = np.where(m, p - y, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = (err**2).sum(axis=1) / days
        out = {
            "days": days,
            "y_sum": np.where(m, y, 0.0).sum(axis=1),
            "abs_err": np.abs(err).sum(axis=1),
            "sq_err": (err**2).sum(axis=1),
            "err": err.sum(axis=1),
            "rmsse": np.sqrt(mse / truth.scale_sq),
            "weight": truth.bottom_weight,
        }
        if quantiles:
            spl = np.zeros(len(y))
            for tau, q in quantiles.items():
                qq = q[:, :h].astype(np.float64)
                loss = np.where(m, pinball(y, qq, tau), 0.0).sum(axis=1) / days
                spl += loss / truth.scale_abs
                out[f"below_q{round(tau * 100):02d}"] = np.where(m, y <= qq, False).sum(axis=1)
                out[f"under_q{round(tau * 100):02d}"] = np.where(m, y < qq, False).sum(axis=1)
            out["spl"] = spl / len(quantiles)
            if 0.1 in quantiles and 0.9 in quantiles:
                lo, hi = quantiles[0.1][:, :h], quantiles[0.9][:, :h]
                inside = (y >= lo) & (y <= hi)
                out["inside_80"] = np.where(m, inside, False).sum(axis=1)
                out["width_80"] = np.where(m, hi - lo, 0.0).sum(axis=1)
    return pd.DataFrame(out)


def summarize(ps: pd.DataFrame) -> dict[str, float]:
    """Aggregate per-series rows (any slice) into the reported metrics."""
    days = ps["days"].sum()
    y_sum = ps["y_sum"].sum()
    scaled = ps["rmsse"].notna()
    w = ps.loc[scaled, "weight"]
    res = {
        "n_series": float(len(ps)),
        "wrmsse_l12": float((ps.loc[scaled, "rmsse"] * w).sum() / w.sum())
        if w.sum() > 0
        else np.nan,
        "rmsse": float(ps.loc[scaled, "rmsse"].mean()),
        "wape": float(ps["abs_err"].sum() / y_sum) if y_sum > 0 else np.nan,
        "mae": float(ps["abs_err"].sum() / days),
        "rmse": float(np.sqrt(ps["sq_err"].sum() / days)),
        "bias": float(ps["err"].sum() / y_sum) if y_sum > 0 else np.nan,
    }
    if "spl" in ps:
        ok = ps["spl"].notna()
        wq = ps.loc[ok, "weight"]
        res["wspl"] = float((ps.loc[ok, "spl"] * wq).sum() / wq.sum()) if wq.sum() > 0 else np.nan
        for col in [c for c in ps.columns if c.startswith("below_q")]:
            res[col.replace("below_", "cov_")] = float(ps[col].sum() / days)
        for col in [c for c in ps.columns if c.startswith("under_q")]:
            res[col.replace("under_", "cov_lt_")] = float(ps[col].sum() / days)
        if "inside_80" in ps:
            res["cov_80"] = float(ps["inside_80"].sum() / days)
            res["width_80"] = float(ps["width_80"].sum() / days)
    return res
