import json

import pandas as pd
import pytest

from demand_intel.pipelines.ingestion.download import kaggle_env
from demand_intel.pipelines.ingestion.m5 import INGESTION_MANIFEST, IngestionError, ingest_m5
from demand_intel.pipelines.validation.contracts import (
    calendar_contract,
    prices_contract,
    sales_id_contract,
    validate,
)
from tests.conftest import EXPECT, M5Dirs, calendar, prices, sales


def run(d: M5Dirs, force: bool = False):  # type: ignore[no-untyped-def]
    return ingest_m5(d.raw, d.out, d.quarantine, force=force, expectations=EXPECT)


def test_happy_path_writes_typed_outputs(m5_dirs: M5Dirs) -> None:
    report = run(m5_dirs)
    assert report.status == "loaded"
    assert all(c.passed for c in report.checks if c.severity == "error")

    cal = pd.read_parquet(m5_dirs.out / "calendar.parquet")
    assert cal["d"].tolist() == list(range(1, 22))
    assert cal.loc[2, "event_type_1"] == "Sporting"
    assert pd.isna(cal.loc[3, "event_type_1"])

    long = pd.read_parquet(m5_dirs.out / "sales.parquet")
    assert len(long) == 4 * EXPECT.evaluation_days
    assert long["d"].min() == 1 and long["d"].max() == EXPECT.evaluation_days
    # spot-check the reshape against the wide source
    wide = sales(EXPECT.evaluation_days, "evaluation").set_index(["item_id", "store_id"])
    got = long.set_index(["item_id", "store_id", "d"])["sales"]
    assert got[("HOBBIES_1_001", "TX_1", 9)] == int(wide.loc[("HOBBIES_1_001", "TX_1"), "d_9"])

    series = pd.read_parquet(m5_dirs.out / "series.parquet")
    assert set(series["id"].astype(str)) == {
        f"{i}_{s}" for i in ("FOODS_1_001", "HOBBIES_1_001") for s in ("CA_1", "TX_1")
    }
    assert len(pd.read_parquet(m5_dirs.out / "sell_prices.parquet")) == len(prices())

    manifest = json.loads((m5_dirs.out / INGESTION_MANIFEST).read_text())
    f = {x["source_file"]: x for x in manifest["files"]}
    assert f["sell_prices.csv"]["row_count"] == 12
    assert f["sell_prices.csv"]["schema_version"] == "prices.v1"
    assert len(f["calendar.csv"]["checksum_sha256"]) == 64


def test_rerun_is_skipped_until_a_source_changes(m5_dirs: M5Dirs) -> None:
    first = run(m5_dirs)
    second = run(m5_dirs)
    assert second.status == "skipped"
    assert [f.run_id for f in second.files] == [f.run_id for f in first.files]

    changed = prices()
    changed.loc[0, "sell_price"] = "2.47"
    m5_dirs.write("sell_prices.csv", changed)
    assert run(m5_dirs).status == "loaded"
    assert run(m5_dirs, force=True).status == "loaded"


def test_negative_sales_fail_the_run_and_are_quarantined(m5_dirs: M5Dirs) -> None:
    bad = sales(EXPECT.evaluation_days, "evaluation")
    bad.loc[1, "d_5"] = "-3"
    m5_dirs.write("sales_train_evaluation.csv", bad)

    with pytest.raises(IngestionError) as err:
        run(m5_dirs)
    failed = {c.name for c in err.value.failed}
    assert "sales_train_evaluation.csv:reject_rate" in failed
    assert not (m5_dirs.out / "sales.parquet").exists()

    q = pd.read_parquet(m5_dirs.quarantine / "sales_train_evaluation.parquet")
    assert q["_source_row"].tolist() == [2]
    assert list(q["_reasons"].loc[0]) == ["sales:negative"]


def test_malformed_and_blank_sales_cells_are_flagged(m5_dirs: M5Dirs) -> None:
    bad = sales(EXPECT.evaluation_days, "evaluation")
    bad.loc[0, "d_2"] = "abc"
    bad.loc[3, "d_7"] = ""
    bad.loc[2, "d_1"] = "1.5"
    m5_dirs.write("sales_train_evaluation.csv", bad)
    with pytest.raises(IngestionError):
        run(m5_dirs)
    q = pd.read_parquet(m5_dirs.quarantine / "sales_train_evaluation.parquet").set_index(
        "_source_row"
    )
    assert "sales:unparseable" in q["_reasons"].loc[1]
    assert "sales:missing" in q["_reasons"].loc[4]
    assert "sales:non_integer" in q["_reasons"].loc[3]


def test_validation_file_must_match_evaluation_prefix(m5_dirs: M5Dirs) -> None:
    val = sales(EXPECT.validation_days, "validation")
    val.loc[0, "d_3"] = str(int(val["d_3"].loc[0]) + 1)
    m5_dirs.write("sales_train_validation.csv", val)
    with pytest.raises(IngestionError) as err:
        run(m5_dirs)
    assert {c.name for c in err.value.failed} == {
        "sales_train_validation:matches_evaluation_prefix"
    }


def test_missing_day_column_fails(m5_dirs: M5Dirs) -> None:
    m5_dirs.write(
        "sales_train_evaluation.csv",
        sales(EXPECT.evaluation_days, "evaluation").drop(columns="d_4"),
    )
    with pytest.raises(IngestionError) as err:
        run(m5_dirs)
    assert "sales_train_evaluation.csv:day_columns" in {c.name for c in err.value.failed}


def test_calendar_contract_rejects_inconsistent_rows() -> None:
    cal = calendar()
    cal.loc[0, "weekday"] = "Monday"
    cal.loc[1, "event_type_1"] = "Party"
    cal.loc[2, "event_type_1"] = ""  # name without type
    cal.loc[3, "snap_TX"] = "2"
    cal.loc[4, "d"] = "d_99"
    cal.loc[5, "date"] = cal.loc[6, "date"]
    result = validate(cal, calendar_contract())
    reasons = dict(zip(result.rejected.index, result.rejected["_reasons"], strict=True))
    assert "row:weekday_matches_date" in reasons[0]
    assert any(r.startswith("event_type_1:") for r in reasons[1])
    assert "row:event_1_name_iff_type" in reasons[2]
    assert any(r.startswith("snap_TX:") for r in reasons[3])
    assert "row:d_matches_date" in reasons[4]
    assert {5, 6} <= set(reasons)  # duplicated date rejects both rows
    assert len(result.valid) == len(cal) - len(reasons)


def test_sales_id_contract_checks_hierarchy() -> None:
    ids = sales(3, "evaluation")[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    ids.loc[0, "id"] = "FOODS_1_001_TX_1_evaluation"  # id does not match store
    ids.loc[1, "state_id"] = "WI"  # row 1 is store TX_1
    ids.loc[2, "dept_id"] = "HOUSEHOLD_1"  # item HOBBIES_1_001 not in this dept
    result = validate(ids, sales_id_contract())
    reasons = dict(zip(result.rejected.index, result.rejected["_reasons"], strict=True))
    assert "row:id_matches_item_store" in reasons[0]
    assert "row:store_in_state" in reasons[1]
    assert "row:item_in_dept" in reasons[2]
    assert "row:dept_in_cat" in reasons[2]


def test_prices_contract_checks_references_and_domain() -> None:
    p = prices()
    p.loc[0, "sell_price"] = "0"
    p.loc[1, "store_id"] = "WI_9"
    p.loc[2, "wm_yr_wk"] = "99999"
    p.loc[3, "item_id"] = "FOODS_9_999"
    p = pd.concat([p, p.iloc[[5]]], ignore_index=True)  # duplicate store-item-week
    contract = prices_contract(
        stores={"CA_1", "TX_1"},
        items={"FOODS_1_001", "HOBBIES_1_001"},
        weeks={11101, 11102, 11103},
    )
    result = validate(p, contract)
    reasons = dict(zip(result.rejected.index, result.rejected["_reasons"], strict=True))
    assert any(r.startswith("sell_price:") for r in reasons[0])
    assert any(r.startswith("store_id:") for r in reasons[1])
    assert any(r.startswith("wm_yr_wk:") for r in reasons[2])
    assert any(r.startswith("item_id:") for r in reasons[3])
    assert "row:store_item_week_unique" in reasons[5]
    assert "row:store_item_week_unique" in reasons[len(p) - 1]


def test_kaggle_env_maps_new_style_token() -> None:
    env = kaggle_env({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "KGAT_abc"})
    assert env["KAGGLE_API_TOKEN"] == "KGAT_abc"
    legacy = kaggle_env({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "0123abcd"})
    assert "KAGGLE_API_TOKEN" not in legacy
