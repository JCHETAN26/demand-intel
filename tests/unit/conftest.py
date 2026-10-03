"""A tiny M5-shaped dataset: 21 calendar days, 2 items x 2 stores, 14 evaluation days."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

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
