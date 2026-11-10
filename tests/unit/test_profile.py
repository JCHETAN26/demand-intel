import numpy as np
import pandas as pd
import pytest

from demand_intel.analysis.profile import (
    demand_class,
    sample_panel,
    series_profile,
    suspicious_zero_runs,
)
from demand_intel.data import M5Data, first_sale_index, load_m5
from demand_intel.pipelines.ingestion.m5 import ingest_m5
from tests.conftest import EXPECT, M5Dirs, sales


def _m5(sales_rows: list[list[int]], priced: bool = True) -> M5Data:
    arr = np.array(sales_rows, dtype=np.int32)
    n = len(arr)
    series = pd.DataFrame(
        {
            "id": [f"I_{i}_CA_1" for i in range(n)],
            "item_id": [f"I_{i}" for i in range(n)],
            "dept_id": "D",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        }
    )
    prices = np.full(arr.shape, 2.0 if priced else np.nan, dtype=np.float32)
    cal = pd.DataFrame({"d": np.arange(1, arr.shape[1] + 1)})
    return M5Data(series=series, sales=arr, calendar=cal, prices=prices)


def test_demand_class_cutoffs() -> None:
    adi = np.array([1.0, 1.0, 2.0, 2.0, 5.0])
    cv2 = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
    nz = np.array([10, 10, 10, 10, 1])
    assert demand_class(adi, cv2, nz).tolist() == [
        "smooth",
        "erratic",
        "intermittent",
        "lumpy",
        "insufficient",
    ]


def test_series_profile_uses_days_after_first_sale() -> None:
    m5 = _m5([[0, 0, 2, 0, 2, 0], [1, 1, 1, 1, 1, 1]])
    prof = series_profile(m5, end_day=6)
    assert prof.loc[0, "first_sale_day"] == 3
    assert prof.loc[0, "active_days"] == 4
    assert prof.loc[0, "adi"] == pytest.approx(2.0)
    assert prof.loc[0, "cv2"] == pytest.approx(0.0)
    assert prof.loc[0, "demand_class"] == "intermittent"
    assert prof.loc[1, "demand_class"] == "smooth"
    assert first_sale_index(np.array([[0, 0], [0, 3]])).tolist() == [2, 1]


def test_zero_run_flagged_only_when_implausible_and_priced() -> None:
    busy = [5] * 60 + [0] * 20 + [5] * 10  # 20 zero days after selling 5/day: flagged
    slow = [1 if i % 10 == 0 else 0 for i in range(60)] + [0] * 20 + [1] * 10  # plausible
    zr = suspicious_zero_runs(_m5([busy, slow]))
    assert zr.runs["series_idx"].tolist() == [0]
    assert zr.runs[["start_day", "end_day", "length"]].iloc[0].tolist() == [61, 80, 20]
    assert zr.runs.loc[0, "kind"] == "stockout"
    dormant = suspicious_zero_runs(_m5([[5] * 60 + [0] * 70 + [5] * 5]))
    assert dormant.runs.loc[0, "kind"] == "dormant"
    assert dormant.flagged_days == 0 and dormant.dormant_days == 70
    assert suspicious_zero_runs(_m5([busy], priced=False)).runs.empty


def test_empirical_test_spares_lumpy_series_that_poisson_flags() -> None:
    # sells 7 units every 7th day: mean 1/day, but 6 of 7 days are zero
    lumpy = [7 if i % 7 == 0 else 0 for i in range(90)] + [0] * 20 + [7] * 5
    m5 = _m5([lumpy])
    assert len(suspicious_zero_runs(m5, method="poisson").runs) == 1
    assert suspicious_zero_runs(m5).runs.empty


def test_panel_is_stratified_deterministic_and_sized() -> None:
    rng = np.random.default_rng(0)
    n = 600
    prof = pd.DataFrame(
        {
            "cat_id": rng.choice(["FOODS", "HOBBIES", "HOUSEHOLD"], n, p=[0.6, 0.2, 0.2]),
            "store_id": rng.choice(["CA_1", "TX_1"], n),
            "demand_class": rng.choice(["smooth", "lumpy", "intermittent", "insufficient"], n),
        }
    )
    a = sample_panel(prof, 100, seed=7)
    b = sample_panel(prof, 100, seed=7)
    assert a.index.equals(b.index)
    assert len(a) == 100
    assert "insufficient" not in set(a["demand_class"])
    eligible = prof[prof["demand_class"] != "insufficient"]
    assert set(map(tuple, a[["cat_id", "store_id", "demand_class"]].to_numpy())) == set(
        map(tuple, eligible[["cat_id", "store_id", "demand_class"]].to_numpy())
    )


def test_load_m5_roundtrip(m5_dirs: M5Dirs) -> None:
    ingest_m5(m5_dirs.raw, m5_dirs.out, m5_dirs.quarantine, expectations=EXPECT)
    m5 = load_m5(m5_dirs.out)
    wide = sales(EXPECT.evaluation_days, "evaluation")
    for i, row in enumerate(m5.series.to_dict("records")):
        src = wide[(wide["item_id"] == row["item_id"]) & (wide["store_id"] == row["store_id"])]
        expected = src[[f"d_{t}" for t in range(1, 15)]].astype(int).to_numpy()[0]
        assert m5.sales[i].tolist() == expected.tolist()
    assert m5.prices is not None
    assert m5.prices.shape == (4, EXPECT.calendar_days)
    assert np.allclose(m5.prices, 1.97)
