"""Global LightGBM forecaster F3 (PROJECT_SPEC.md §5.3, plan Phases 11-12).

One model per horizon bucket (1-7, 8-14, 15-28), each trained across all series in scope
with the hierarchy as features. Per fold and bucket:

1. Training targets are the `train_days` days ending 28 days before the origin; the last
   28 days before the origin form the *calibration* window (always out-of-sample).
2. The calibration window drives early stopping and the quantile map: predictions are
   binned (equal-count bins of the predicted mean) and each bin stores the empirical
   quantiles of the actual sales it contains. A forecast's quantiles are its bin's.
3. Forecasts use features anchored at or before the origin; days on which an item has no
   price are forecast as 0 (not on sale).
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from demand_intel import SEED
from demand_intel.forecasting.backtesting.harness import FoldContext, Forecast
from demand_intel.forecasting.backtesting.splits import HORIZON
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.pipelines.features.build import BUCKETS, DEFAULT_FAMILIES, build_features

CALIB_DAYS = 28
N_BINS = 50
RATIO_FROM = 3.0  # units/day: bins at or above this use ratio quantiles

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 4,
    "seed": SEED,
    "verbose": -1,
}


@dataclass
class QuantileMap:
    """Empirical conditional quantiles of demand given the predicted mean.

    Predictions are binned (equal-count bins). Low-volume bins (mean prediction below
    RATIO_FROM units) store the empirical quantiles of y directly, which keeps them exact
    for small counts. Higher bins store quantiles of the ratio y / prediction and are
    multiplied by the prediction, so quantiles scale with volume: an item forecast at 60
    units gets bands around 60, not the bin's typical item's.
    """

    edges: np.ndarray  # (n_bins - 1,) inner bin edges on the predicted mean
    table: np.ndarray  # (n_bins, Q): absolute quantiles, or ratio quantiles where ratio[b]
    ratio: np.ndarray  # (n_bins,) bool

    @classmethod
    def fit(cls, pred: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> "QuantileMap":
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)[1:-1]))
        b = np.searchsorted(edges, pred, side="right")
        k_bins = len(edges) + 1
        table = np.zeros((k_bins, len(QUANTILES)))
        ratio = np.zeros(k_bins, dtype=bool)
        for k in range(k_bins):
            sel = b == k
            if not sel.any():
                table[k], ratio[k] = table[max(k - 1, 0)], ratio[max(k - 1, 0)]
                continue
            if pred[sel].mean() >= RATIO_FROM:
                ratio[k] = True
                table[k] = np.quantile(y[sel] / np.maximum(pred[sel], 1e-9), QUANTILES)
            else:
                table[k] = np.quantile(y[sel], QUANTILES)
        table = np.maximum.accumulate(table, axis=1)  # enforce ordering across quantiles
        return cls(edges=edges, table=table, ratio=ratio)

    def apply(self, pred: np.ndarray) -> dict[float, np.ndarray]:
        b = np.searchsorted(self.edges, pred, side="right")
        scale = np.where(self.ratio[b], pred, 1.0)
        return {q: self.table[b, j] * scale for j, q in enumerate(QUANTILES)}


@dataclass
class LightGBMForecaster:
    name: str = "lgbm"
    families: Sequence[str] = DEFAULT_FAMILIES
    train_days: int = 364
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    max_rounds: int = 1500
    early_stopping: int = 50
    model_dir: Path | None = None  # if set, boosters + metadata are saved per fold
    # Mean-bias correction: per-category factor Σy / Σŷ on the out-of-sample calibration
    # window, clipped to 1 ± bias_clip, applied to the point forecast only (quantiles keep
    # their own empirical map).
    bias_correct: bool = False
    bias_clip: float = 0.10
    log: list[dict[str, Any]] = field(default_factory=list)

    def _bias_factors(
        self, ctx: FoldContext, series_idx: np.ndarray, y: np.ndarray, pred: np.ndarray
    ) -> dict[str, float]:
        cats = ctx.data.series["cat_id"].to_numpy()[ctx.rows][series_idx]
        out = {}
        for c in np.unique(cats):
            sel = cats == c
            ratio = float(y[sel].sum() / max(pred[sel].sum(), 1e-9))
            out[str(c)] = float(np.clip(ratio, 1 - self.bias_clip, 1 + self.bias_clip))
        return out

    def _fit_bucket(
        self, ctx: FoldContext, hb: int
    ) -> tuple[lgb.Booster, QuantileMap, dict[str, float]]:
        o = ctx.origin
        fit_days = np.arange(o - CALIB_DAYS - self.train_days + 1, o - CALIB_DAYS + 1)
        cal_days = np.arange(o - CALIB_DAYS + 1, o + 1)
        text = ctx.extras.get("text")
        tr = build_features(
            ctx.data, ctx.rows, fit_days, hb, self.families, ctx.weather, True, text
        )
        va = build_features(
            ctx.data, ctx.rows, cal_days, hb, self.families, ctx.weather, True, text
        )
        assert tr.y is not None and va.y is not None
        t0 = time.perf_counter()
        dtrain = lgb.Dataset(tr.X[tr.eligible], tr.y[tr.eligible], free_raw_data=True)
        dvalid = lgb.Dataset(va.X[va.eligible], va.y[va.eligible], reference=dtrain)
        booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )
        pred = booster.predict(va.X[va.eligible], num_iteration=booster.best_iteration)
        qmap = QuantileMap.fit(np.asarray(pred), va.y[va.eligible])
        factors = (
            self._bias_factors(ctx, va.series_idx[va.eligible], va.y[va.eligible], np.asarray(pred))
            if self.bias_correct
            else {}
        )
        self.log.append(
            {
                "fold": ctx.fold.name,
                "bucket": hb,
                "train_rows": int(tr.eligible.sum()),
                "best_iteration": int(booster.best_iteration),
                "fit_seconds": time.perf_counter() - t0,
                "bias_factors": factors,
            }
        )
        return booster, qmap, factors

    def forecast(self, ctx: FoldContext) -> Forecast:
        n = len(ctx.rows)
        point = np.zeros((n, HORIZON), dtype=np.float32)
        quant = {q: np.zeros((n, HORIZON), dtype=np.float32) for q in QUANTILES}
        for lo, hi in BUCKETS:
            booster, qmap, factors = self._fit_bucket(ctx, hi)
            days = np.arange(ctx.origin + lo, ctx.origin + hi + 1)
            ff = build_features(
                ctx.data,
                ctx.rows,
                days,
                hi,
                self.families,
                ctx.weather,
                text=ctx.extras.get("text"),
            )
            pred = np.asarray(booster.predict(ff.X, num_iteration=booster.best_iteration))
            on_sale = ~np.isnan(np.asarray(ctx.data.prices)[ctx.rows][:, days - 1]).ravel()
            pred = np.where(on_sale, np.clip(pred, 0, None), 0.0)
            cols = slice(lo - 1, hi)
            scale = np.ones(len(pred))
            if factors:
                cats = ctx.data.series["cat_id"].to_numpy()[ctx.rows][ff.series_idx]
                scale = np.array([factors.get(str(c), 1.0) for c in cats])
            point[:, cols] = (pred * scale).reshape(n, len(days))
            for q, v in qmap.apply(pred).items():
                quant[q][:, cols] = np.where(on_sale, v, 0.0).reshape(n, len(days))
            if self.model_dir is not None:
                d = self.model_dir / self.name / ctx.population / ctx.fold.name
                d.mkdir(parents=True, exist_ok=True)
                booster.save_model(
                    str(d / f"bucket_{hi}.txt"), num_iteration=booster.best_iteration
                )
                np.savez(
                    d / f"bucket_{hi}_qmap.npz",
                    edges=qmap.edges,
                    table=qmap.table,
                    ratio=qmap.ratio,
                )
                meta: dict[str, Any] = {"families": list(self.families), "params": self.params}
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                meta.setdefault("bias_factors", {})[str(hi)] = factors
                meta_path.write_text(json.dumps(meta, indent=2))
        return Forecast(point=point, quantiles=quant)
