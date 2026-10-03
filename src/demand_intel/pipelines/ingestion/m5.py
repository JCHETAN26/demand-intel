"""Validate the raw M5 files and write typed Parquet to `data/validated/m5/`.

Outputs (all lossless with respect to the source):
- `calendar.parquet`     one row per day, `d` as an integer day index
- `series.parquet`       one row per item-store series with its hierarchy
- `sales.parquet`        long format (item_id, store_id, d, sales), from the evaluation file
- `sell_prices.parquet`  one row per store-item-week

`sales_train_validation.csv` is validated and checked to equal the first 1,913 days of the
evaluation file, but not written: it holds no additional data.

Rejected rows go to `data/quarantine/m5/<file>.parquet` with their reasons. File-level
checks (row counts, contiguous days, reject-rate limits) fail the whole run *before* any
validated output is written. A run is skipped when `_ingestion.json` shows the same source
checksums and schema versions and every output exists, so re-running is idempotent.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from demand_intel.pipelines.ingestion.download import sha256_file
from demand_intel.pipelines.validation.contracts import (
    SALES_ID_COLUMNS,
    calendar_contract,
    check_sales_matrix,
    prices_contract,
    sales_id_contract,
    validate,
)

DATASET = "m5"
CALENDAR = "calendar.csv"
SALES_EVAL = "sales_train_evaluation.csv"
SALES_VAL = "sales_train_validation.csv"
PRICES = "sell_prices.csv"

SCHEMA_VERSIONS = {
    CALENDAR: "calendar.v1",
    SALES_EVAL: "sales.v1",
    SALES_VAL: "sales.v1",
    PRICES: "prices.v1",
}
# Any rejected calendar or sales row means the source is not what the spec assumes.
MAX_REJECT_RATE = {CALENDAR: 0.0, SALES_EVAL: 0.0, SALES_VAL: 0.0, PRICES: 0.001}
OUTPUTS = ("calendar.parquet", "series.parquet", "sales.parquet", "sell_prices.parquet")
INGESTION_MANIFEST = "_ingestion.json"


@dataclass(frozen=True)
class M5Expectations:
    """Known shape of the source (PROJECT_SPEC.md §2.1). Tests pass smaller values."""

    calendar_days: int = 1969
    evaluation_days: int = 1941
    validation_days: int = 1913
    series: int = 30490
    items: int = 3049
    stores: int = 10


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: Literal["error", "warn"]
    detail: str = ""


@dataclass(frozen=True)
class FileResult:
    source_file: str
    run_id: str
    checksum_sha256: str
    schema_version: str
    row_count: int
    loaded_count: int
    rejected_count: int
    rejected_path: str | None
    outputs: list[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    dataset: str
    status: Literal["loaded", "skipped"]
    ingested_at: str
    files: list[FileResult]
    checks: list[Check]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], status: Literal["loaded", "skipped"]
    ) -> "IngestionReport":
        return cls(
            dataset=d["dataset"],
            status=status,
            ingested_at=d["ingested_at"],
            files=[FileResult(**f) for f in d["files"]],
            checks=[Check(**c) for c in d["checks"]],
        )


class IngestionError(RuntimeError):
    def __init__(self, failed: list[Check]) -> None:
        self.failed = failed
        super().__init__(
            "M5 ingestion failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed)
        )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _quarantine(rejected: pd.DataFrame, path: Path) -> str | None:
    """Write rejected rows (source values as strings) with 1-based source row numbers.

    `rejected` is indexed by 0-based position in the source file (excluding the header).
    """
    if path.exists():
        path.unlink()
    if rejected.empty:
        return None
    out = rejected.drop(columns="_reasons").astype(str)
    out.insert(0, "_source_row", np.asarray(rejected.index) + 1)
    out["_reasons"] = rejected["_reasons"]
    _write_parquet(out, path)
    return str(path)


def _reject_rate_check(name: str, rejected: int, total: int) -> Check:
    limit = MAX_REJECT_RATE[name]
    rate = rejected / total if total else 0.0
    return Check(
        f"{name}:reject_rate",
        rate <= limit,
        "error",
        f"{rejected}/{total} rejected, limit {limit:.2%}",
    )


def _count_check(name: str, actual: int, expected: int) -> Check:
    return Check(name, actual == expected, "error", f"{actual} (expected {expected})")


# --- calendar -----------------------------------------------------------------------------


def _ingest_calendar(
    path: Path, exp: M5Expectations, checks: list[Check]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    result = validate(raw, calendar_contract())
    checks.append(_reject_rate_check(CALENDAR, len(result.rejected), len(raw)))
    checks.append(_count_check("calendar:rows", len(raw), exp.calendar_days))

    cal = result.valid.copy()
    cal["d"] = cal["d"].str.removeprefix("d_").astype("int16")
    cal = cal.sort_values("d").reset_index(drop=True)
    contiguous = cal["d"].to_numpy().tolist() == list(range(1, len(cal) + 1))
    checks.append(Check("calendar:days_contiguous", contiguous, "error", "d_1..d_N without gaps"))
    weeks = cal.groupby("wm_yr_wk")["d"].agg(["size", "min"])
    checks.append(
        Check(
            "calendar:weeks_valid",
            bool((weeks["size"] <= 7).all() and weeks.sort_index()["min"].is_monotonic_increasing),
            "error",
            "each wm_yr_wk has <= 7 days and weeks increase with time",
        )
    )
    for col in ("wm_yr_wk", "wday", "month", "year"):
        cal[col] = cal[col].astype("int32" if col == "wm_yr_wk" else "int16")
    for s in ("CA", "TX", "WI"):
        cal[f"snap_{s}"] = cal[f"snap_{s}"].astype("int8")
    return cal, result.rejected, len(raw)


# --- sales --------------------------------------------------------------------------------


@dataclass
class _Sales:
    ids: pd.DataFrame  # valid identifier rows
    matrix: np.ndarray  # valid rows x days, int32
    rejected: pd.DataFrame
    total: int


def _read_sales(path: Path, expected_days: int, name: str, checks: list[Check]) -> _Sales:
    raw = pd.read_csv(path, dtype={c: str for c in SALES_ID_COLUMNS}, keep_default_na=False)
    day_cols = [c for c in raw.columns if c not in SALES_ID_COLUMNS]
    expected_cols = [f"d_{i}" for i in range(1, expected_days + 1)]
    checks.append(
        Check(
            f"{name}:day_columns",
            day_cols == expected_cols,
            "error",
            f"{len(day_cols)} day columns (expected d_1..d_{expected_days} in order)",
        )
    )
    ids = validate(raw[list(SALES_ID_COLUMNS)], sales_id_contract())
    flags = check_sales_matrix(raw[day_cols])
    bad = flags.any(axis=1)
    bad.loc[ids.rejected.index] = True
    id_reasons = dict(zip(ids.rejected.index, ids.rejected["_reasons"], strict=True))
    reasons = [[*id_reasons.get(i, []), *flags.columns[flags.loc[i]]] for i in bad[bad].index]
    rejected = raw.loc[bad, list(SALES_ID_COLUMNS)].copy()
    rejected["_reasons"] = reasons

    good = ~bad
    values = raw.loc[good, day_cols].to_numpy(dtype=np.int64) if good.any() else np.empty((0, 0))
    if values.size and values.max() > np.iinfo(np.int32).max:
        raise OverflowError(f"{name}: sales value exceeds int32")
    return _Sales(
        ids=ids.valid.loc[good[good].index].reset_index(drop=True),
        matrix=values.astype(np.int32),
        rejected=rejected,
        total=len(raw),
    )


def _series_frame(ids: pd.DataFrame) -> pd.DataFrame:
    series = ids.drop(columns="id").copy()
    series.insert(0, "id", series["item_id"] + "_" + series["store_id"])
    return series.astype("category")


def _long_sales(series: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    n_series, n_days = values.shape
    row = np.repeat(np.arange(n_series), n_days)
    item = pd.Categorical(series["item_id"].astype(str))
    store = pd.Categorical(series["store_id"].astype(str))
    return pd.DataFrame(
        {
            "item_id": pd.Categorical.from_codes(item.codes[row], categories=item.categories),
            "store_id": pd.Categorical.from_codes(store.codes[row], categories=store.categories),
            "d": np.tile(np.arange(1, n_days + 1, dtype=np.int16), n_series),
            "sales": values.ravel(),
        }
    )


def _matches_prefix(evaluation: _Sales, validation: _Sales, days: int) -> bool:
    def keyed(s: _Sales, n: int) -> pd.DataFrame:
        key = s.ids["item_id"].astype(str) + "_" + s.ids["store_id"].astype(str)
        return pd.DataFrame(s.matrix[:, :n], index=key).sort_index()

    ev, va = keyed(evaluation, days), keyed(validation, days)
    return ev.index.equals(va.index) and np.array_equal(ev.to_numpy(), va.to_numpy())


# --- orchestration ------------------------------------------------------------------------


def _is_current(out_dir: Path, checksums: dict[str, str]) -> dict[str, Any] | None:
    path = out_dir / INGESTION_MANIFEST
    if not path.exists() or not all((out_dir / o).exists() for o in OUTPUTS):
        return None
    manifest: dict[str, Any] = json.loads(path.read_text())
    recorded = {f["source_file"]: f for f in manifest.get("files", [])}
    for name, checksum in checksums.items():
        f = recorded.get(name)
        if f is None or f["checksum_sha256"] != checksum:
            return None
        if f["schema_version"] != SCHEMA_VERSIONS[name]:
            return None
    return manifest


def ingest_m5(
    raw_dir: Path,
    out_dir: Path,
    quarantine_dir: Path,
    force: bool = False,
    expectations: M5Expectations | None = None,
) -> IngestionReport:
    exp = expectations or M5Expectations()
    paths = {name: raw_dir / name for name in SCHEMA_VERSIONS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw M5 files: {missing}")
    checksums = {name: sha256_file(p) for name, p in paths.items()}
    if not force and (manifest := _is_current(out_dir, checksums)) is not None:
        return IngestionReport.from_dict(manifest, status="skipped")

    checks: list[Check] = []
    run_ids = {name: str(uuid.uuid4()) for name in paths}

    calendar, cal_rejected, cal_total = _ingest_calendar(paths[CALENDAR], exp, checks)

    evaluation = _read_sales(paths[SALES_EVAL], exp.evaluation_days, SALES_EVAL, checks)
    validation = _read_sales(paths[SALES_VAL], exp.validation_days, SALES_VAL, checks)
    for name, s in ((SALES_EVAL, evaluation), (SALES_VAL, validation)):
        checks.append(_reject_rate_check(name, len(s.rejected), s.total))
    series = _series_frame(evaluation.ids)
    checks += [
        _count_check("sales:series", len(series), exp.series),
        _count_check("sales:items", series["item_id"].nunique(), exp.items),
        _count_check("sales:stores", series["store_id"].nunique(), exp.stores),
        Check(
            "sales:within_calendar",
            exp.evaluation_days <= len(calendar),
            "error",
            f"{exp.evaluation_days} sales days vs {len(calendar)} calendar days",
        ),
        Check(
            "sales_train_validation:matches_evaluation_prefix",
            _matches_prefix(evaluation, validation, exp.validation_days),
            "error",
            f"validation file equals evaluation file on d_1..d_{exp.validation_days}",
        ),
    ]

    prices_raw = pd.read_csv(paths[PRICES], dtype=str, keep_default_na=False)
    prices_result = validate(
        prices_raw,
        prices_contract(
            stores=set(series["store_id"].astype(str)),
            items=set(series["item_id"].astype(str)),
            weeks=set(calendar["wm_yr_wk"].tolist()),
        ),
    )
    checks.append(_reject_rate_check(PRICES, len(prices_result.rejected), len(prices_raw)))
    prices = prices_result.valid.copy()
    priced = prices[["store_id", "item_id"]].drop_duplicates()
    checks.append(
        Check(
            "sell_prices:every_series_priced",
            len(priced) == len(series),
            "warn",
            f"{len(priced)} of {len(series)} series have at least one price",
        )
    )
    prices = prices.astype(
        {
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int32",
            "sell_price": "float64",
        }
    )

    rejected_frames = {
        CALENDAR: cal_rejected,
        SALES_EVAL: evaluation.rejected,
        SALES_VAL: validation.rejected,
        PRICES: prices_result.rejected,
    }
    quarantined = {
        name: _quarantine(df, quarantine_dir / f"{Path(name).stem}.parquet")
        for name, df in rejected_frames.items()
    }

    failed = [c for c in checks if c.severity == "error" and not c.passed]
    if failed:
        raise IngestionError(failed)

    _write_parquet(calendar, out_dir / "calendar.parquet")
    _write_parquet(series, out_dir / "series.parquet")
    _write_parquet(_long_sales(series, evaluation.matrix), out_dir / "sales.parquet")
    _write_parquet(prices, out_dir / "sell_prices.parquet")

    counts = {
        CALENDAR: (cal_total, len(calendar), len(cal_rejected)),
        SALES_EVAL: (evaluation.total, len(evaluation.ids), len(evaluation.rejected)),
        SALES_VAL: (validation.total, len(validation.ids), len(validation.rejected)),
        PRICES: (len(prices_raw), len(prices), len(prices_result.rejected)),
    }
    outputs = {
        CALENDAR: ["calendar.parquet"],
        SALES_EVAL: ["series.parquet", "sales.parquet"],
        SALES_VAL: [],
        PRICES: ["sell_prices.parquet"],
    }
    report = IngestionReport(
        dataset=DATASET,
        status="loaded",
        ingested_at=datetime.now(UTC).isoformat(),
        files=[
            FileResult(
                source_file=name,
                run_id=run_ids[name],
                checksum_sha256=checksums[name],
                schema_version=SCHEMA_VERSIONS[name],
                row_count=counts[name][0],
                loaded_count=counts[name][1],
                rejected_count=counts[name][2],
                rejected_path=quarantined[name],
                outputs=outputs[name],
            )
            for name in SCHEMA_VERSIONS
        ],
        checks=checks,
    )
    (out_dir / INGESTION_MANIFEST).write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return report
