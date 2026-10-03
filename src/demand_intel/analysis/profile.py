"""Per-series demand profile, demand classes and suspicious zero runs (PROJECT_SPEC §2.1, §5.2).

Everything here is computed on a fixed *profiling window* that ends at the first backtest
origin (d_1773), so the PANEL sample and demand classes never use evaluation-period data.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from demand_intel.data import M5Data, first_sale_index
from demand_intel.forecasting.backtesting.splits import BACKTEST_ORIGINS

PROFILE_END_DAY = BACKTEST_ORIGINS[0]  # last day used for profiling (inclusive)

# Syntetos–Boylan cut-offs
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

# Suspicious zero runs: length >= MIN_RUN and P(a zero run this long | trailing history) < P_MAX.
# "empirical" (default) uses the trailing share of zero days, which respects overdispersion;
# "poisson" uses the trailing mean rate and over-flags lumpy series (spec v0.2 change log).
MIN_RUN = 14
# Runs this long or longer are dormant periods (delisting with a price kept, long-term
# out-of-stock or collapsed demand), not replenishment stockouts; they are reported apart.
DORMANT_RUN = 56
TRAILING_DAYS = 90
P_MAX = 1e-3


def demand_class(adi: np.ndarray, cv2: np.ndarray, nonzero: np.ndarray) -> np.ndarray:
    cls = np.where(
        adi < ADI_CUTOFF,
        np.where(cv2 < CV2_CUTOFF, "smooth", "erratic"),
        np.where(cv2 < CV2_CUTOFF, "intermittent", "lumpy"),
    ).astype(object)
    cls[nonzero < 2] = "insufficient"
    return cls


def series_profile(m5: M5Data, end_day: int = PROFILE_END_DAY) -> pd.DataFrame:
    """One row per series: activity, intermittency (ADI, CV²), class, volume, revenue."""
    sales = m5.sales[:, :end_day]
    first = first_sale_index(sales)
    cols = np.arange(sales.shape[1])
    active = cols[None, :] >= first[:, None]
    nz = (sales > 0) & active
    n_active = active.sum(axis=1)
    n_nonzero = nz.sum(axis=1)
    total = sales.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        adi = np.where(n_nonzero > 0, n_active / n_nonzero, np.inf)
        nz_vals = np.where(nz, sales, 0).astype(float)
        mean_nz = nz_vals.sum(axis=1) / n_nonzero
        var_nz = (np.where(nz, (sales - mean_nz[:, None]) ** 2, 0.0)).sum(axis=1) / n_nonzero
        cv2 = var_nz / mean_nz**2
    revenue_28 = np.zeros(len(sales))
    if m5.prices is not None:
        last28 = slice(end_day - 28, end_day)
        revenue_28 = np.nansum(sales[:, last28] * m5.prices[:, last28], axis=1)

    prof = m5.series.copy()
    prof["first_sale_day"] = np.where(first < sales.shape[1], first + 1, -1)
    prof["active_days"] = n_active
    prof["nonzero_days"] = n_nonzero
    prof["zero_share_active"] = np.where(n_active > 0, 1 - n_nonzero / np.maximum(n_active, 1), 1.0)
    prof["mean_daily_active"] = np.where(n_active > 0, total / np.maximum(n_active, 1), 0.0)
    prof["adi"] = adi
    prof["cv2"] = np.nan_to_num(cv2, nan=0.0)
    prof["demand_class"] = demand_class(adi, prof["cv2"].to_numpy(), n_nonzero)
    prof["revenue_last28"] = revenue_28
    prof["volume_tercile"] = pd.qcut(
        prof["mean_daily_active"].rank(method="first"), 3, labels=["low", "mid", "high"]
    ).astype(str)
    return prof


@dataclass(frozen=True)
class ZeroRuns:
    runs: pd.DataFrame  # series_idx, start_day, end_day (1-based, inclusive), length, rate, kind
    flagged_days: int  # days in stockout-kind runs
    series_affected: int  # series with at least one stockout-kind run
    dormant_days: int


def suspicious_zero_runs(
    m5: M5Data,
    end_day: int | None = None,
    method: Literal["empirical", "poisson"] = "empirical",
) -> ZeroRuns:
    """Zero runs that are implausible given the series' trailing history (likely stockouts).

    `empirical`: P = p0^L with p0 the smoothed share of zero days in the trailing window
    (independent days, but no Poisson assumption, so lumpy series are not over-flagged).
    `poisson`: P = exp(-rate * L) with the trailing mean rate.
    A run on days when the item has no price is a delisting, not a stockout, so those days
    break runs.
    """
    end = end_day or m5.n_days
    sales = m5.sales[:, :end]
    priced = ~np.isnan(m5.prices[:, :end]) if m5.prices is not None else np.ones_like(sales, bool)
    first = first_sale_index(sales)
    csum = np.concatenate([np.zeros((len(sales), 1)), np.cumsum(sales, axis=1)], axis=1)
    zsum = np.concatenate([np.zeros((len(sales), 1)), np.cumsum(sales == 0, axis=1)], axis=1)
    rows = []
    for i in range(len(sales)):
        f = first[i]
        if f >= end:
            continue
        zero = (sales[i, f:] == 0) & priced[i, f:]
        if not zero.any():
            continue
        padded = np.concatenate([[False], zero, [False]])
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        starts, stops = edges[::2] + f, edges[1::2] + f  # [start, stop)
        lengths = stops - starts
        keep = lengths >= MIN_RUN
        for s, e, n in zip(starts[keep], stops[keep], lengths[keep], strict=True):
            lo = max(f, s - TRAILING_DAYS)
            if s - lo < 28:  # not enough history to judge the rate
                continue
            rate = (csum[i, s] - csum[i, lo]) / (s - lo)
            if method == "poisson":
                log_p = -rate * n
            else:
                p0 = (zsum[i, s] - zsum[i, lo] + 1) / (s - lo + 2)
                log_p = n * np.log(p0)
            if log_p < np.log(P_MAX):
                rows.append((i, s + 1, e, n, rate))
    runs = pd.DataFrame(rows, columns=["series_idx", "start_day", "end_day", "length", "rate"])
    runs["kind"] = np.where(runs["length"] >= DORMANT_RUN, "dormant", "stockout")
    stockout = runs[runs["kind"] == "stockout"]
    return ZeroRuns(
        runs=runs,
        flagged_days=int(stockout["length"].sum()),
        series_affected=int(stockout["series_idx"].nunique()),
        dormant_days=int(runs.loc[runs["kind"] == "dormant", "length"].sum()),
    )


def sample_panel(profile: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Stratified sample over category × store × demand class, proportional allocation.

    Every non-empty stratum gets at least one series; the remainder is allocated by largest
    remainder. Series with fewer than 2 sales in the profiling window are excluded because
    their demand class (and RMSSE scale) is undefined.
    """
    eligible = profile[profile["demand_class"] != "insufficient"]
    keys = ["cat_id", "store_id", "demand_class"]
    sizes = eligible.groupby(keys).size()
    sizes = sizes[sizes > 0]
    quota = sizes / sizes.sum() * size
    alloc = (quota // 1).clip(lower=1).astype(int)
    remainder = size - int(alloc.sum())
    if remainder > 0:
        order = (quota - quota // 1).sort_values(ascending=False).index[:remainder]
        alloc.loc[order] += 1
    alloc = alloc.clip(upper=sizes)
    rng = np.random.default_rng(seed)
    picks = []
    for key, n in alloc.items():
        members = eligible.index[(eligible[keys] == pd.Series(key, index=keys)).all(axis=1)]
        picks.extend(rng.choice(members, size=int(n), replace=False).tolist())
    return profile.loc[sorted(picks)]
