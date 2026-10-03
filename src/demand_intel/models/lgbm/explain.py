"""SHAP explanations for the LightGBM forecaster (plan Phase 13).

LightGBM computes exact TreeSHAP values natively (`pred_contrib=True`). With the Tweedie
objective they are additive in the *log* of the forecast: raw = base + Σ contrib_j and
forecast = exp(raw). For explanations in units we allocate the change from the baseline
forecast, exp(base) → exp(raw), to features in proportion to their log-space contribution.
This allocation is exact in total, approximate per feature, and labelled as such in the UI.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

FAMILY_OF_PREFIX = (
    ("lag_", "recent demand"),
    ("mean_", "recent demand"),
    ("std_", "recent demand"),
    ("max_", "recent demand"),
    ("min_", "recent demand"),
    ("nonzero_share", "recent demand"),
    ("momentum", "demand trend"),
    ("slope", "demand trend"),
    ("accel", "demand trend"),
    ("days_since", "recent demand"),
    ("price", "price"),
    ("discount", "promotion (price proxy)"),
    ("event", "events/holidays"),
    ("snap", "SNAP"),
    ("wx_", "weather"),
    ("wday", "calendar"),
    ("mday", "calendar"),
    ("month", "calendar"),
    ("week", "calendar"),
    ("year", "calendar"),
    ("is_weekend", "calendar"),
    ("item_id", "product identity"),
    ("dept_id", "product identity"),
    ("cat_id", "product identity"),
    ("store_id", "store"),
    ("state_id", "store"),
)


def family(feature: str) -> str:
    for prefix, fam in FAMILY_OF_PREFIX:
        if feature.startswith(prefix):
            return fam
    return "other"


@dataclass
class Contributions:
    log_space: pd.DataFrame  # (rows, features) SHAP values on the log scale
    base: np.ndarray  # (rows,) expected raw score
    forecast: np.ndarray  # (rows,) model forecast in units

    def units(self) -> pd.DataFrame:
        """Proportional allocation of forecast - exp(base) to features (sums exactly)."""
        total = self.log_space.sum(axis=1).to_numpy()
        delta = self.forecast - np.exp(self.base)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(np.abs(total) > 1e-12, delta / total, 0.0)
        return self.log_space.mul(share, axis=0)

    def by_family(self, units: bool = True) -> pd.DataFrame:
        frame = self.units() if units else self.log_space
        return frame.T.groupby(frame.columns.map(family)).sum().T


def contributions(booster: lgb.Booster, features: pd.DataFrame) -> Contributions:
    raw = np.asarray(booster.predict(features, pred_contrib=True))
    names = booster.feature_name()
    return Contributions(
        log_space=pd.DataFrame(raw[:, :-1], columns=names, index=features.index),
        base=raw[:, -1],
        forecast=np.asarray(booster.predict(features)),
    )


def global_importance(c: Contributions, groups: pd.Series | None = None) -> pd.DataFrame:
    """Mean |SHAP| (log scale) per feature, overall or per group (e.g. category)."""
    absval = c.log_space.abs()
    if groups is None:
        out = absval.mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
        out["family"] = out.index.map(family)
        return out.reset_index(names="feature")
    out = absval.groupby(groups.to_numpy()).mean().T
    out["family"] = out.index.map(family)
    return out.reset_index(names="feature")
