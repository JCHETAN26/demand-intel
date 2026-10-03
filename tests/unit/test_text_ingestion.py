import json
import zipfile
from pathlib import Path

import pandas as pd

from demand_intel.pipelines.ingestion.text import build_recalls, parse_distribution


def test_parse_distribution_codes_names_and_nationwide() -> None:
    assert parse_distribution("CA, TX and WI") == (["CA", "TX", "WI"], False)
    assert parse_distribution("Distributed in California and Wisconsin") == (["CA", "WI"], False)
    assert parse_distribution("Nationwide in US") == ([], True)
    assert parse_distribution("Distributed throughout the United States") == ([], True)
    codes, _ = parse_distribution("CAN and Canada only")
    assert codes == []  # 'CA' inside 'CAN' is not a state


def _write(tmp: Path, results: list[dict[str, str]]) -> Path:
    path = tmp / "raw.zip"
    with zipfile.ZipFile(path, "w") as zf:
        payload = {"meta": {"last_updated": "x"}, "results": results}
        zf.writestr("food-enforcement.json", json.dumps(payload))
    return path


def test_build_recalls_validates_and_quarantines(tmp_path: Path) -> None:
    good = {
        "recall_number": "F-1-2014",
        "event_id": "1",
        "report_date": "20140110",
        "recall_initiation_date": "20140102",
        "classification": "Class I",
        "reason_for_recall": "Undeclared peanuts.",
        "product_description": "Granola bars, 12 oz",
        "distribution_pattern": "CA and TX",
    }
    bad_empty = {**good, "recall_number": "F-2-2014", "reason_for_recall": ""}
    bad_date = {**good, "recall_number": "F-3-2014", "report_date": "2014-13-45"}
    bad_order = {**good, "recall_number": "F-4-2014", "report_date": "20131201"}
    dup_a = {**good, "recall_number": "F-5-2014"}
    dup_b = {**good, "recall_number": "F-5-2014", "product_description": "Other"}
    raw = _write(tmp_path, [good, bad_empty, bad_date, bad_order, dup_a, dup_b])
    rep = build_recalls(raw, tmp_path / "out.parquet", tmp_path / "q.parquet")
    assert (rep.total, rep.loaded, rep.rejected) == (6, 1, 5)
    out = pd.read_parquet(tmp_path / "out.parquet")
    assert out.loc[0, "doc_id"] == "F-1-2014"
    assert list(out.loc[0, "states"]) == ["CA", "TX"]
    assert out.loc[0, "severity"] == "high"
    q = pd.read_parquet(tmp_path / "q.parquet")
    flat = {r for rs in q["_reasons"] for r in rs}
    assert {
        "reason_for_recall:empty",
        "report_date:unparseable",
        "report_before_initiation",
        "recall_number:duplicate",
    } <= flat
