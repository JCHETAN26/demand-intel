"""In-memory views of the validated M5 data.

`sales.parquet` is stored series-major (all days of series 0, then series 1, ...) in the
row order of `series.parquet`, so it reshapes into a series × day matrix without a pivot.
Prices are weekly; `price_matrix` expands them to series × calendar day (NaN = not on sale).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HIERARCHY = ("item_id", "dept_id", "cat_id", "store_id", "state_id")


@dataclass
class M5Data:
    series: pd.DataFrame  # one row per series: id + HIERARCHY, in matrix row order
    sales: np.ndarray  # (n_series, n_days) int32; column j is day d = j + 1
    calendar: pd.DataFrame  # one row per calendar day, d = 1..n_calendar_days
    prices: np.ndarray | None  # (n_series, n_calendar_days) float32, NaN = not on sale

    @property
    def n_days(self) -> int:
        return int(self.sales.shape[1])


def load_m5(validated_dir: Path, with_prices: bool = True) -> M5Data:
    series = pd.read_parquet(validated_dir / "series.parquet")
    for col in ("id", *HIERARCHY):
        series[col] = series[col].astype(str)
    long = pd.read_parquet(validated_dir / "sales.parquet")
    n_series = len(series)
    n_days = len(long) // n_series
    if n_series * n_days != len(long):
        raise ValueError("sales.parquet is not a complete series × day grid")
    d = long["d"].to_numpy().reshape(n_series, n_days)
    if not (d == np.arange(1, n_days + 1)).all():
        raise ValueError("sales.parquet is not ordered series-major by day")
    firsts = long.iloc[::n_days]
    same_item = firsts["item_id"].astype(str).to_numpy() == series["item_id"].to_numpy()
    same_store = firsts["store_id"].astype(str).to_numpy() == series["store_id"].to_numpy()
    if not (same_item.all() and same_store.all()):
        raise ValueError("sales.parquet series order differs from series.parquet")
    sales = long["sales"].to_numpy(dtype=np.int32).reshape(n_series, n_days)

    calendar = pd.read_parquet(validated_dir / "calendar.parquet")
    prices = price_matrix(validated_dir, series, calendar) if with_prices else None
    return M5Data(series=series, sales=sales, calendar=calendar, prices=prices)


def price_matrix(validated_dir: Path, series: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    p = pd.read_parquet(validated_dir / "sell_prices.parquet")
    row = pd.Series(np.arange(len(series)), index=series["item_id"] + "_" + series["store_id"])
    p_row = row.reindex(p["item_id"].astype(str) + "_" + p["store_id"].astype(str)).to_numpy()
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    p_week = np.searchsorted(weeks, p["wm_yr_wk"].to_numpy())
    weekly = np.full((len(series), len(weeks)), np.nan, dtype=np.float32)
    weekly[p_row, p_week] = p["sell_price"].to_numpy(dtype=np.float32)
    day_week = np.searchsorted(weeks, calendar.sort_values("d")["wm_yr_wk"].to_numpy())
    daily: np.ndarray = weekly[:, day_week]
    return daily


def first_sale_index(sales: np.ndarray) -> np.ndarray:
    """0-based column of each series' first nonzero sale (n_days if it never sells)."""
    nonzero = sales > 0
    first: np.ndarray = nonzero.argmax(axis=1)
    first[~nonzero.any(axis=1)] = sales.shape[1]
    return first
