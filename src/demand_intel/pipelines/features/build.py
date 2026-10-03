"""Supervised features for direct, horizon-bucketed forecasting (PROJECT_SPEC.md §4, §6.1).

A sample is (series i, target day t). For a bucket whose largest horizon is `hb`, every
sales-derived feature is computed at the *anchor* day a = t - hb, so at forecast time
(origin T, target T + h with h <= hb) the anchor a = T + h - hb <= T is always observed.
Calendar, SNAP and price features describe the target day itself: they are known in
advance (spec §2.2). Weather is past-only (anchored like sales) unless the explicitly
labelled oracle family is requested.

Features are grouped into families so the ablation ladder (§6.1, §6.2) can add them one at
a time.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_intel.data import M5Data, first_sale_index

FAMILIES = ("sales", "calendar", "snap", "price", "hierarchy", "weather", "weather_oracle")
DEFAULT_FAMILIES = ("sales", "calendar", "snap", "price", "hierarchy")
BUCKETS = ((1, 7), (8, 14), (15, 28))
CATEGORICAL = ("item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name", "event_type")


@dataclass
class FeatureFrame:
    X: pd.DataFrame
    y: np.ndarray | None  # targets (training only)
    series_idx: np.ndarray  # row → series
    day: np.ndarray  # row → target day (1-based)
    eligible: np.ndarray  # row → priced and on/after the first sale


class _Cum:
    """Prefix sums for O(1) window statistics; column k holds the sum of the first k days."""

    def __init__(self, sales: np.ndarray) -> None:
        s = sales.astype(np.float64)
        zero = np.zeros((len(s), 1))
        self.sum = np.concatenate([zero, np.cumsum(s, axis=1)], axis=1)
        self.sq = np.concatenate([zero, np.cumsum(s**2, axis=1)], axis=1)
        self.nz = np.concatenate([zero, np.cumsum(s > 0, axis=1)], axis=1)

    def mean(self, a: np.ndarray, w: int) -> np.ndarray:
        """Mean over days (a-w, a] for anchor days a (1-based), per series; a is (n_days,)."""
        lo = np.maximum(a - w, 0)
        cnt = np.maximum(a - lo, 1)
        out: np.ndarray = (self.sum[:, a] - self.sum[:, lo]) / cnt
        return out

    def std(self, a: np.ndarray, w: int) -> np.ndarray:
        lo = np.maximum(a - w, 0)
        cnt = np.maximum(a - lo, 1)
        m = (self.sum[:, a] - self.sum[:, lo]) / cnt
        v = (self.sq[:, a] - self.sq[:, lo]) / cnt - m**2
        out: np.ndarray = np.sqrt(np.maximum(v, 0))
        return out

    def window_mean(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Mean over days (lo, hi] (1-based, clipped at 0)."""
        lo, hi = np.maximum(lo, 0), np.maximum(hi, 0)
        out: np.ndarray = (self.sum[:, hi] - self.sum[:, lo]) / np.maximum(hi - lo, 1)
        return out

    def nonzero_share(self, a: np.ndarray, w: int) -> np.ndarray:
        lo = np.maximum(a - w, 0)
        cnt = np.maximum(a - lo, 1)
        out: np.ndarray = (self.nz[:, a] - self.nz[:, lo]) / cnt
        return out


def _rolling_extreme(sales: np.ndarray, a: np.ndarray, w: int, fn: str) -> np.ndarray:
    out = np.empty((len(sales), len(a)), dtype=np.float32)
    for j, day in enumerate(a):
        window = sales[:, max(day - w, 0) : day]
        out[:, j] = window.max(axis=1) if fn == "max" else window.min(axis=1)
    return out


def _sales_features(sales: np.ndarray, days: np.ndarray, hb: int) -> dict[str, np.ndarray]:
    """Sales-history features anchored at a = t - hb (never later than the origin)."""
    a = days - hb  # anchor day (1-based); sales[:, a-1] is the anchor day's value
    if (a < 1).any():
        raise ValueError("target days too early for this bucket's anchor")
    if a.max() > sales.shape[1]:
        raise ValueError("anchor beyond observed sales: would leak the future")
    cum = _Cum(sales)
    f: dict[str, np.ndarray] = {}
    for k in range(7):  # the 7 most recent observable days
        f[f"lag_{hb + k}"] = sales[:, a - 1 - k]
    for k in (28, 56):  # longer fixed lags (target-relative), when not already covered
        if k > hb + 6:
            f[f"lag_{k}"] = sales[:, days - k - 1]
    # same weekday as the target, in the 4 most recent observable weeks
    first_week = -(-hb // 7)
    for j in range(4):
        k = 7 * (first_week + j)
        f[f"lag_sameday_{j + 1}w"] = sales[:, days - k - 1]
    for w in (7, 14, 28, 56, 112):
        f[f"mean_{w}"] = cum.mean(a, w)
    for w in (7, 28):
        f[f"std_{w}"] = cum.std(a, w)
    f["max_28"] = _rolling_extreme(sales, a, 28, "max")
    f["min_28"] = _rolling_extreme(sales, a, 28, "min")
    f["nonzero_share_28"] = cum.nonzero_share(a, 28)
    f["nonzero_share_112"] = cum.nonzero_share(a, 112)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["momentum_7_28"] = np.where(f["mean_28"] > 0, f["mean_7"] / f["mean_28"], 0.0)
    week0 = cum.window_mean(a - 7, a)  # most recent observable week
    week1 = cum.window_mean(a - 14, a - 7)
    week2 = cum.window_mean(a - 21, a - 14)
    f["slope_14"] = cum.window_mean(a - 14, a) - cum.window_mean(a - 28, a - 14)
    f["accel_7"] = (week0 - week1) - (week1 - week2)
    last_sale = np.maximum.accumulate(
        np.where(sales > 0, np.arange(1, sales.shape[1] + 1, dtype=np.int32)[None, :], 0), axis=1
    )
    since = a[None, :] - last_sale[:, a - 1]
    f["days_since_sale"] = np.where(last_sale[:, a - 1] > 0, since, 9999)
    return f


def _calendar_features(cal: pd.DataFrame, days: np.ndarray) -> dict[str, np.ndarray]:
    c = cal.set_index("d").loc[days]
    date = pd.to_datetime(c["date"])
    return {
        "wday": c["wday"].to_numpy(),
        "mday": date.dt.day.to_numpy(),
        "month": c["month"].to_numpy(),
        "week": date.dt.isocalendar().week.to_numpy(dtype=np.int16),
        "year": c["year"].to_numpy(),
        "is_weekend": (c["wday"] <= 2).to_numpy().astype(np.int8),  # wday 1 = Sat, 2 = Sun
        "event_2": c["event_name_2"].notna().to_numpy().astype(np.int8),
    }


def _calendar_categories(cal: pd.DataFrame, days: np.ndarray) -> dict[str, pd.Categorical]:
    """Event name/type of the target day; categories span the whole calendar so codes are
    stable across folds."""
    out = {}
    for col, src in (("event_name", "event_name_1"), ("event_type", "event_type_1")):
        values = cal.set_index("d")[src].fillna("none")
        cats = pd.Index(sorted(values.unique()))
        out[col] = pd.Categorical.from_codes(cats.get_indexer(values.loc[days]), categories=cats)
    return out


def _price_features(
    prices: np.ndarray, days: np.ndarray, series: pd.DataFrame
) -> dict[str, np.ndarray]:
    p = prices[:, days - 1]
    prev_week = prices[:, np.maximum(days - 8, 0)]
    # trailing 8-week mean and max, over days up to the target (prices are known-future)
    roll_mean = np.empty_like(p)
    roll_max = np.empty_like(p)
    key = series["dept_id"].to_numpy() + "/" + series["store_id"].to_numpy()
    rel = np.empty_like(p)
    with warnings.catch_warnings():  # all-NaN windows (item not yet on sale) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, d in enumerate(days):
            window = prices[:, max(d - 56, 0) : d]
            roll_mean[:, j] = np.nanmean(window, axis=1)
            roll_max[:, j] = np.nanmax(window, axis=1)
        for k in np.unique(key):
            rows = key == k
            rel[rows] = p[rows] / np.nanmedian(p[rows], axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "price": p,
            "price_change_1w": p / prev_week - 1,
            "price_vs_mean_8w": p / roll_mean - 1,
            "discount_proxy": (p <= 0.9 * roll_max).astype(np.int8),
            "price_vs_dept_store": rel,
        }


def _weather_features(
    weather: pd.DataFrame,
    cal: pd.DataFrame,
    series: pd.DataFrame,
    days: np.ndarray,
    hb: int,
    oracle: bool,
) -> dict[str, np.ndarray]:
    """Weather by state. Past-only: 7-day means up to the anchor; oracle: target-day values."""
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    d_of = pd.Series(cal["d"].to_numpy(), index=pd.to_datetime(cal["date"]))
    w["d"] = d_of.reindex(w["date"]).to_numpy()
    # climatology from 2011-2014 only (before every backtest origin)
    w["doy"] = w["date"].dt.dayofyear
    clim = w[w["date"].dt.year <= 2014].groupby(["state_id", "doy"], observed=True)["tmax_c"].mean()
    w["tmax_anom"] = (
        w["tmax_c"] - clim.reindex(pd.MultiIndex.from_arrays([w["state_id"], w["doy"]])).to_numpy()
    )
    out: dict[str, np.ndarray] = {}
    state = series["state_id"].to_numpy()
    cols = ("tmax_c", "precip_mm", "snow_cm", "tmax_anom")
    for col in cols:
        out[f"wx_{col}{'_oracle' if oracle else '_7d'}"] = np.zeros(
            (len(series), len(days)), np.float32
        )
    for st in np.unique(state):
        ws = w[w["state_id"] == st].set_index("d").sort_index()
        rows = state == st
        for col in cols:
            vals = ws[col]
            if oracle:
                v = vals.reindex(days).to_numpy()
            else:
                roll = vals.rolling(7, min_periods=1).mean()
                v = roll.reindex(days - hb).to_numpy()
            out[f"wx_{col}{'_oracle' if oracle else '_7d'}"][rows] = v[None, :]
    return out


def build_features(
    m5: M5Data,
    rows: np.ndarray,
    days: np.ndarray,
    hb: int,
    families: Sequence[str] = DEFAULT_FAMILIES,
    weather: pd.DataFrame | None = None,
    with_target: bool = False,
) -> FeatureFrame:
    """Features for series `rows` × target `days` with anchor offset `hb`.

    `m5.sales` must be truncated at the forecast origin by the caller (the harness does);
    any feature that would need a later day raises instead of silently leaking.
    """
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown feature families {sorted(unknown)}")
    days = np.asarray(days)
    series = m5.series.iloc[rows].reset_index(drop=True)
    sales = m5.sales[rows]
    assert m5.prices is not None
    prices = m5.prices[rows]
    n, nd = len(rows), len(days)

    cols: dict[str, np.ndarray] = {}
    cats: dict[str, pd.Categorical] = {}
    if "sales" in families:
        cols.update(_sales_features(sales, days, hb))
    if "calendar" in families:
        for k, v in _calendar_features(m5.calendar, days).items():
            cols[k] = np.broadcast_to(v[None, :], (n, nd))
        for k, c in _calendar_categories(m5.calendar, days).items():
            cats[k] = pd.Categorical.from_codes(np.tile(c.codes, n), categories=c.categories)
    if "snap" in families:
        cal = m5.calendar.set_index("d").loc[days]
        snap = np.stack([cal[f"snap_{s}"].to_numpy() for s in ("CA", "TX", "WI")])
        state_pos = series["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
        cols["snap"] = snap[state_pos]
    if "price" in families:
        cols.update(_price_features(prices, days, series))
    if "hierarchy" in families:
        # categories come from the full series table so codes match across folds/subsets
        for k in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
            levels = pd.Index(sorted(m5.series[k].unique()))
            codes = levels.get_indexer(series[k])
            cats[k] = pd.Categorical.from_codes(np.repeat(codes, nd), categories=levels)
    for fam, oracle in (("weather", False), ("weather_oracle", True)):
        if fam in families:
            if weather is None:
                raise ValueError(f"family {fam!r} needs the weather table")
            cols.update(_weather_features(weather, m5.calendar, series, days, hb, oracle))

    frame = pd.DataFrame(
        {
            **{k: np.asarray(v, dtype=np.float32).ravel() for k, v in cols.items()},
            **cats,
        }
    )

    first = first_sale_index(sales)
    eligible = (~np.isnan(prices[:, days - 1])) & (days[None, :] - 1 >= first[:, None])
    y = sales[:, days - 1].astype(np.float32).ravel() if with_target else None
    return FeatureFrame(
        X=frame,
        y=y,
        series_idx=np.repeat(np.arange(n), nd),
        day=np.tile(days, n),
        eligible=eligible.ravel(),
    )
