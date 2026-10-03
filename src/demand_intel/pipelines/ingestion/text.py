"""Text corpus ingestion: openFDA food enforcement reports (PROJECT_SPEC.md §2.4).

Each document is one recall report. Timestamps: `report_date` is when the report became
public (the date a forecaster could have known it) and is the only date used for signals.

Validation (plan Phase 6, NLP): non-empty text, parseable dates (report on/after
initiation), unique recall numbers, source recorded, English text (ASCII-letter share),
and malformed records quarantined with reasons. Distribution patterns are parsed into
US state codes plus a nationwide flag.
"""

import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from demand_intel.pipelines.ingestion.download import MANIFEST, sha256_file

SOURCE = "openFDA food enforcement"
URL = "https://download.open.fda.gov/food/enforcement/food-enforcement-0001-of-0001.json.zip"
RAW_NAME = "openfda_food_enforcement.json.zip"
SCHEMA_VERSION = "recalls.v1"
FIELDS = (
    "recall_number",
    "event_id",
    "report_date",
    "recall_initiation_date",
    "classification",
    "reason_for_recall",
    "product_description",
    "recalling_firm",
    "distribution_pattern",
    "voluntary_mandated",
    "status",
)

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}  # fmt: skip
NATIONWIDE = re.compile(
    r"\b(nation\s*wide|nationally|all\s+(50\s+)?states"
    r"|throughout\s+the\s+(us|u\.s\.|united states))",
    re.I,
)
_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(STATES.values(), key=len, reverse=True)) + r")\b", re.I
)
_CODE_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(STATES) + r")(?![A-Za-z])")
_NAME_TO_CODE = {v.lower(): k for k, v in STATES.items()}


def parse_distribution(pattern: str) -> tuple[list[str], bool]:
    """US state codes mentioned in a distribution pattern, and whether it is nationwide."""
    text = pattern or ""
    codes = {_NAME_TO_CODE[m.group(1).lower()] for m in _NAME_RE.finditer(text)}
    codes |= {m.group(1) for m in _CODE_RE.finditer(text)}
    return sorted(codes), bool(NATIONWIDE.search(text))


def download_recalls(raw_dir: Path, force: bool = False) -> Path:
    out = raw_dir / "text"
    out.mkdir(parents=True, exist_ok=True)
    path = out / RAW_NAME
    manifest_path = out / MANIFEST
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    )
    recorded = manifest["files"].get(RAW_NAME)
    if force or not path.exists() or not recorded or recorded["sha256"] != sha256_file(path):
        with httpx.stream("GET", URL, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for block in r.iter_bytes(1 << 20):
                    f.write(block)
        manifest["files"][RAW_NAME] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        manifest |= {"source": SOURCE, "url": URL, "downloaded_at": datetime.now(UTC).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


@dataclass(frozen=True)
class TextReport:
    total: int
    loaded: int
    rejected: int
    reasons: dict[str, int]
    last_updated: str


def _english_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def build_recalls(raw_zip: Path, out_path: Path, quarantine_path: Path) -> TextReport:
    with zipfile.ZipFile(raw_zip) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        payload = json.load(io.TextIOWrapper(zf.open(name), encoding="utf-8"))
    records = pd.DataFrame([{f: r.get(f) for f in FIELDS} for r in payload["results"]])
    df = records.copy()
    reasons: dict[str, pd.Series] = {}
    for col in ("reason_for_recall", "product_description"):
        df[col] = df[col].fillna("").astype(str).str.strip()
        reasons[f"{col}:empty"] = df[col] == ""
    for col in ("report_date", "recall_initiation_date"):
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        reasons[f"{col}:unparseable"] = df[col].isna()
    reasons["report_before_initiation"] = df["report_date"] < df["recall_initiation_date"]
    reasons["recall_number:duplicate"] = (
        df["recall_number"].duplicated(keep=False) | df["recall_number"].isna()
    )
    reasons["classification:invalid"] = ~df["classification"].isin(
        ["Class I", "Class II", "Class III"]
    )
    text = df["reason_for_recall"] + " " + df["product_description"]
    reasons["language:not_english"] = text.map(_english_share) < 0.95

    flags = pd.DataFrame(reasons).fillna(False).astype(bool)
    bad = flags.any(axis=1)
    rejected = records[bad].copy()
    cols = flags.columns.to_numpy()
    rejected["_reasons"] = [list(cols[r]) for r in flags[bad].to_numpy()]

    good = df[~bad].copy()
    parsed = good["distribution_pattern"].fillna("").map(parse_distribution)
    good["states"] = parsed.map(lambda p: p[0])
    good["nationwide"] = parsed.map(lambda p: p[1])
    good["severity"] = good["classification"].map(
        {"Class I": "high", "Class II": "medium", "Class III": "low"}
    )
    good["text"] = good["reason_for_recall"] + "\nProduct: " + good["product_description"]
    good["source"] = SOURCE
    good["schema_version"] = SCHEMA_VERSION
    good = (
        good.rename(columns={"recall_number": "doc_id"})
        .sort_values("report_date")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    good.to_parquet(out_path, index=False)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    rejected.astype({c: str for c in rejected.columns if c != "_reasons"}).to_parquet(
        quarantine_path, index=False
    )
    return TextReport(
        total=len(records),
        loaded=len(good),
        rejected=int(bad.sum()),
        reasons={str(k): int(v.sum()) for k, v in flags.items()},
        last_updated=str(payload.get("meta", {}).get("last_updated")),
    )


def doc_length_stats(df: pd.DataFrame) -> dict[str, float]:
    n = df["text"].map(lambda t: len(str(t).split()))
    return {"median_words": float(np.median(n)), "p95_words": float(np.percentile(n, 95))}
