"""Model comparison and paired significance tests (PROJECT_SPEC.md §5.6, §6, §9.2).

- `summary_table`: one row per (model, population) with fold-averaged backtest metrics,
  the test-window metrics and cost.
- `paired_test`: model A vs B on the same series. The statistic is the level-12 WRMSSE
  difference, Σ_i w_i (RMSSE_A,i − RMSSE_B,i), averaged over folds; its 95% CI comes from
  a bootstrap that resamples series (clusters) with all their folds. A Wilcoxon
  signed-rank test on per-series fold-averaged RMSSE is the cross-check. "Improvement"
  means the CI lies entirely below 0. `fold_wins` counts the folds where A beats B.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from demand_intel import SEED
from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, TEST_FOLD

BACKTEST = tuple(f.name for f in BACKTEST_FOLDS)
METRICS = (
    "wrmsse",
    "wrmsse_l12",
    "rmsse",
    "wape",
    "mae",
    "rmse",
    "bias",
    "wspl",
    "cov_lt_q10",
    "cov_q10",
    "cov_lt_q90",
    "cov_q90",
    "cov_80",
    "width_80",
)


def load_results(backtest_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(backtest_dir.glob("*__*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary_table(results: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    overall = results[(results["slice"] == "all") & (results["window"] == window)]
    rows = []
    for (model, pop), grp in overall.groupby(["model", "population"]):
        bt = grp[grp["fold"].isin(BACKTEST)]
        te = grp[grp["fold"] == TEST_FOLD.name]
        row: dict[str, object] = {"model": model, "population": pop, "folds": len(bt)}
        for m in METRICS:
            if m in grp:
                row[f"backtest_{m}"] = bt[m].mean() if len(bt) else np.nan
                row[f"test_{m}"] = te[m].iloc[0] if len(te) else np.nan
        row["seconds_per_fold"] = grp["seconds"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["population", "backtest_wrmsse"]).reset_index(drop=True)


@dataclass
class PairedResult:
    model_a: str
    model_b: str
    population: str
    folds: tuple[str, ...]
    diff: float  # mean over folds of level-12 WRMSSE(A) - WRMSSE(B); negative = A better
    ci_low: float
    ci_high: float
    fold_wins: int  # folds where A has the lower level-12 WRMSSE
    wilcoxon_p: float
    n_series: int

    @property
    def improvement(self) -> bool:
        return self.ci_high < 0

    def as_row(self) -> dict[str, object]:
        return {**self.__dict__, "folds": ",".join(self.folds), "improvement": self.improvement}


def _per_series(forecast_dir: Path, model: str, population: str, fold: str) -> pd.DataFrame:
    z = np.load(forecast_dir / model / population / f"{fold}.npz")
    return pd.DataFrame({"rmsse": z["rmsse"], "weight": z["weight"]}, index=z["rows"])


def paired_test(
    forecast_dir: Path,
    model_a: str,
    model_b: str,
    population: str,
    folds: Sequence[str] = BACKTEST,
    n_boot: int = 1000,
) -> PairedResult:
    a = [_per_series(forecast_dir, model_a, population, f) for f in folds]
    b = [_per_series(forecast_dir, model_b, population, f) for f in folds]
    # contributions w_i * RMSSE_i per fold, aligned on series present in both with a scale
    da, db, wins, keep = [], [], 0, None
    for fa, fb in zip(a, b, strict=True):
        common = fa.index.intersection(fb.index)
        ok = fa.loc[common, "rmsse"].notna() & fb.loc[common, "rmsse"].notna()
        idx = common[ok.to_numpy()]
        keep = idx if keep is None else keep.intersection(idx)
    assert keep is not None
    for fa, fb in zip(a, b, strict=True):
        ca = (fa.loc[keep, "rmsse"] * fa.loc[keep, "weight"]).to_numpy()
        cb = (fb.loc[keep, "rmsse"] * fb.loc[keep, "weight"]).to_numpy()
        da.append(ca)
        db.append(cb)
        wins += int(ca.sum() < cb.sum())
    diff_matrix = np.stack(da) - np.stack(db)  # folds × series
    per_series = diff_matrix.mean(axis=0)
    rng = np.random.default_rng(SEED)
    n = len(per_series)
    boots = per_series[rng.integers(0, n, size=(n_boot, n))].sum(axis=1)
    ra = np.mean([fa.loc[keep, "rmsse"].to_numpy() for fa in a], axis=0)
    rb = np.mean([fb.loc[keep, "rmsse"].to_numpy() for fb in b], axis=0)
    p = float(stats.wilcoxon(ra, rb).pvalue) if np.any(ra != rb) else 1.0
    return PairedResult(
        model_a=model_a,
        model_b=model_b,
        population=population,
        folds=tuple(folds),
        diff=float(per_series.sum()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        fold_wins=wins,
        wilcoxon_p=p,
        n_series=n,
    )


def available_models(
    forecast_dir: Path, population: str, folds: Sequence[str] = BACKTEST
) -> list[str]:
    return sorted(
        d.name
        for d in forecast_dir.iterdir()
        if d.is_dir() and all((d / population / f"{f}.npz").exists() for f in folds)
    )
