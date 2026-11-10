"""A tiny M5-shaped dataset: 21 calendar days, 2 items x 2 stores, 14 evaluation days."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from demand_intel.data import M5Data
from demand_intel.pipelines.ingestion.m5 import M5Expectations

ITEMS = (("FOODS_1_001", "FOODS_1", "FOODS"), ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"))
STORES = (("CA_1", "CA"), ("TX_1", "TX"))
EXPECT = M5Expectations(
    calendar_days=21, evaluation_days=14, validation_days=7, series=4, items=2, stores=2
)


def calendar() -> pd.DataFrame:
    rows = []
    for i in range(EXPECT.calendar_days):
        d = date(2011, 1, 29) + timedelta(days=i)
        event = ("SuperBowl", "Sporting") if i == 2 else ("", "")
        rows.append(
            {
                "date": d.isoformat(),
                "wm_yr_wk": str(11101 + i // 7),
                "weekday": d.strftime("%A"),
                "wday": str(i % 7 + 1),
                "month": str(d.month),
                "year": str(d.year),
                "d": f"d_{i + 1}",
                "event_name_1": event[0],
                "event_type_1": event[1],
                "event_name_2": "",
                "event_type_2": "",
                "snap_CA": "1" if i < 10 else "0",
                "snap_TX": "0",
                "snap_WI": "0",
            }
        )
    return pd.DataFrame(rows)


def sales(days: int, suffix: str) -> pd.DataFrame:
    rows = []
    for k, (item, dept, cat) in enumerate(ITEMS):
        for j, (store, state) in enumerate(STORES):
            row = {
                "id": f"{item}_{store}_{suffix}",
                "item_id": item,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": store,
                "state_id": state,
            }
            row |= {f"d_{t}": str((t * (k + 1) + j) % 4) for t in range(1, days + 1)}
            rows.append(row)
    return pd.DataFrame(rows)


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"store_id": s, "item_id": i, "wm_yr_wk": str(w), "sell_price": "1.97"}
            for i, _, _ in ITEMS
            for s, _ in STORES
            for w in (11101, 11102, 11103)
        ]
    )


@dataclass
class M5Dirs:
    raw: Path
    out: Path
    quarantine: Path

    def write(self, name: str, df: pd.DataFrame) -> None:
        path = self.raw / name
        if path.exists():
            path.chmod(0o644)
        df.to_csv(path, index=False)


@pytest.fixture
def m5_dirs(tmp_path: Path) -> M5Dirs:
    dirs = M5Dirs(tmp_path / "raw", tmp_path / "validated", tmp_path / "quarantine")
    dirs.raw.mkdir()
    dirs.write("calendar.csv", calendar())
    dirs.write("sales_train_evaluation.csv", sales(EXPECT.evaluation_days, "evaluation"))
    dirs.write("sales_train_validation.csv", sales(EXPECT.validation_days, "validation"))
    dirs.write("sell_prices.csv", prices())
    return dirs


Mutator = Callable[[pd.DataFrame], pd.DataFrame]


def synthetic_m5(n_items: int = 6, n_days: int = 1941, seed: int = 0) -> M5Data:
    """A small but structurally complete M5Data: real calendar layout, weekly seasonality,
    price changes, two states, items appearing at different times."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cal_days = 1969
    dates = pd.date_range("2011-01-29", periods=cal_days, freq="D")
    cal = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": (11101 + np.arange(cal_days) // 7).astype("int32"),
            "weekday": dates.day_name(),
            "wday": ((dates.dayofweek + 2) % 7 + 1).astype("int16"),
            "month": dates.month.astype("int16"),
            "year": dates.year.astype("int16"),
            "d": np.arange(1, cal_days + 1, dtype="int16"),
            "event_name_1": np.where(
                np.arange(cal_days) % 97 == 0, "SuperBowl", np.array(None, dtype=object)
            ),
            "event_type_1": np.where(
                np.arange(cal_days) % 97 == 0, "Sporting", np.array(None, dtype=object)
            ),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (dates.day <= 10).astype("int8"),
            "snap_TX": (dates.day <= 12).astype("int8"),
            "snap_WI": (dates.day <= 14).astype("int8"),
        }
    )
    rows = []
    for k in range(n_items):
        cat = ("FOODS", "HOBBIES", "HOUSEHOLD")[k % 3]
        for store, state in (("CA_1", "CA"), ("TX_1", "TX")):
            item = f"{cat}_1_{k:03d}"
            rows.append(
                {
                    "id": f"{item}_{store}",
                    "item_id": item,
                    "dept_id": f"{cat}_1",
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                }
            )
    series = pd.DataFrame(rows)
    n = len(series)
    weekly = np.array([1.3, 1.2, 0.9, 0.8, 0.8, 0.9, 1.1])
    level = rng.uniform(0.5, 6, size=n)
    start = rng.integers(0, 400, size=n)
    t = np.arange(n_days)
    lam = level[:, None] * weekly[(t % 7)][None, :]
    sales = rng.poisson(lam).astype(np.int32)
    sales[t[None, :] < start[:, None]] = 0
    prices = np.repeat(rng.uniform(1, 10, size=(n, 1)), cal_days, axis=1).astype(np.float32)
    prices[:, 1000:] *= 0.9
    prices[np.arange(cal_days)[None, :] < start[:, None]] = np.nan
    return M5Data(series=series, sales=sales, calendar=cal, prices=prices)
