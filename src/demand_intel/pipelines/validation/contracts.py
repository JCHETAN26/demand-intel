"""Row-level data contracts for source files (PROJECT_SPEC.md §2.1, plan Phase 5/6).

A contract has two stages:
1. Type coercion: string columns are parsed into their declared types. A non-empty value
   that fails to parse rejects the row with reason `<column>:unparseable`.
2. A Pandera schema, validated lazily, for nullability, domain and row-wise checks.

Rows that fail either stage are split off with their reasons so they can be quarantined.
Checks that span a whole file (row counts, contiguous days) live with the ingestion code,
because they fail the file rather than a row.
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa

from demand_intel.forecasting.backtesting.splits import M5_START

ColumnType = Literal["int", "float", "str", "date", "binary"]

STATES = ("CA", "TX", "WI")
STORES = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")
CATEGORIES = ("FOODS", "HOBBIES", "HOUSEHOLD")
EVENT_TYPES = ("Sporting", "Cultural", "National", "Religious")
SALES_ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


@dataclass(frozen=True)
class Contract:
    types: dict[str, ColumnType]
    schema: pa.DataFrameSchema


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame  # original string values plus a `_reasons` column


def _parse(s: pd.Series, kind: ColumnType) -> pd.Series:
    blank = s.isna() | (s.astype(str).str.strip() == "")
    if kind == "str":
        return s.astype(object).where(~blank, None)
    if kind == "date":
        return pd.to_datetime(s.where(~blank), format="%Y-%m-%d", errors="coerce")
    num = pd.to_numeric(s.where(~blank), errors="coerce")
    if kind == "float":
        return num.astype("Float64")
    # int / binary: reject non-integral numbers such as "1.5"
    num = num.where(np.isclose(num.fillna(0) % 1, 0))
    return num.astype("Int64")


def validate(df: pd.DataFrame, contract: Contract) -> ValidationResult:
    df = df.reset_index(drop=True)
    missing = set(contract.types) - set(df.columns)
    if missing:
        raise ValueError(f"source is missing columns {sorted(missing)}")

    typed = pd.DataFrame(index=df.index)
    reasons: dict[str, pd.Series] = {}
    for col, kind in contract.types.items():
        raw = df[col]
        parsed = _parse(raw, kind)
        had_value = raw.notna() & (raw.astype(str).str.strip() != "")
        reasons[f"{col}:unparseable"] = had_value & parsed.isna()
        typed[col] = parsed

    try:
        contract.schema.validate(typed, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        if fc["index"].isna().any():
            raise  # a table-level failure (e.g. wrong dtype) is a bug, not a bad row
        # DataFrame-level (row-wise) checks are reported once per column; collapse to "row".
        row_level = fc["schema_context"].eq("DataFrameSchema") | fc["column"].isna()
        fc = fc.assign(column=fc["column"].where(~row_level, "row"))
        for (column, check), group in fc.groupby(["column", "check"]):
            mask = pd.Series(False, index=df.index)
            mask.loc[group["index"].astype(int).unique()] = True
            reasons[f"{column}:{check}"] = mask

    flags = pd.DataFrame(reasons)
    bad = flags.any(axis=1)
    rejected = df.loc[bad].copy()
    cols = flags.columns.to_numpy()
    rejected["_reasons"] = [list(cols[row]) for row in flags.loc[bad].to_numpy(dtype=bool)]
    return ValidationResult(valid=typed.loc[~bad], rejected=rejected)


RowCheck = Callable[[pd.DataFrame], pd.Series]


def _row_check(name: str, fn: RowCheck) -> pa.Check:
    return pa.Check(fn, name=name, element_wise=False)


def _binary(name: str) -> pa.Column:
    return pa.Column(pd.Int64Dtype(), pa.Check.isin([0, 1]), nullable=False, name=name)


def calendar_contract() -> Contract:
    def day_number(d: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(d["d"].str.removeprefix("d_"), errors="coerce")

    return Contract(
        types={
            "date": "date",
            "wm_yr_wk": "int",
            "weekday": "str",
            "wday": "int",
            "month": "int",
            "year": "int",
            "d": "str",
            "event_name_1": "str",
            "event_type_1": "str",
            "event_name_2": "str",
            "event_type_2": "str",
            "snap_CA": "binary",
            "snap_TX": "binary",
            "snap_WI": "binary",
        },
        schema=pa.DataFrameSchema(
            {
                "date": pa.Column(None, nullable=False, unique=True),
                "wm_yr_wk": pa.Column(pd.Int64Dtype(), pa.Check.ge(11101), nullable=False),
                "weekday": pa.Column(object, nullable=False),
                "wday": pa.Column(pd.Int64Dtype(), pa.Check.in_range(1, 7), nullable=False),
                "month": pa.Column(pd.Int64Dtype(), pa.Check.in_range(1, 12), nullable=False),
                "year": pa.Column(pd.Int64Dtype(), nullable=False),
                "d": pa.Column(object, pa.Check.str_matches(r"^d_[1-9]\d*$"), nullable=False),
                "event_name_1": pa.Column(object, nullable=True),
                "event_type_1": pa.Column(object, pa.Check.isin(EVENT_TYPES), nullable=True),
                "event_name_2": pa.Column(object, nullable=True),
                "event_type_2": pa.Column(object, pa.Check.isin(EVENT_TYPES), nullable=True),
                **{f"snap_{s}": _binary(f"snap_{s}") for s in STATES},
            },
            checks=[
                _row_check("weekday_matches_date", lambda d: d["weekday"] == d["date"].dt.day_name()),
                # M5 weeks start on Saturday: wday 1 = Saturday ... 7 = Friday.
                _row_check(
                    "wday_matches_date", lambda d: d["wday"] == (d["date"].dt.dayofweek + 2) % 7 + 1
                ),
                _row_check("month_matches_date", lambda d: d["month"] == d["date"].dt.month),
                _row_check("year_matches_date", lambda d: d["year"] == d["date"].dt.year),
                _row_check(
                    "d_matches_date",
                    lambda d: (
                        day_number(d) == (d["date"] - pd.Timestamp(M5_START)).dt.days + 1
                    ).fillna(False),
                ),
                *(
                    _row_check(
                        f"event_{i}_name_iff_type",
                        lambda d, i=i: d[f"event_name_{i}"].isna() == d[f"event_type_{i}"].isna(),
                    )
                    for i in (1, 2)
                ),
                _row_check(
                    "event_2_requires_event_1",
                    lambda d: d["event_name_1"].notna() | d["event_name_2"].isna(),
                ),
            ],
        ),
    )


def sales_id_contract() -> Contract:
    """Identifier columns of a sales file. The day columns are checked by `check_sales_matrix`."""

    def prefixed(child: str, parent: str) -> RowCheck:
        return lambda d: pd.Series(
            [c.startswith(f"{p}_") for c, p in zip(d[child], d[parent], strict=True)],
            index=d.index,
        )

    return Contract(
        types={c: "str" for c in SALES_ID_COLUMNS},
        schema=pa.DataFrameSchema(
            {
                "id": pa.Column(
                    object, pa.Check.str_matches(r"_(validation|evaluation)$"), nullable=False
                ),
                "item_id": pa.Column(object, pa.Check.str_matches(r"^[A-Z]+_\d+_\d+$"), nullable=False),
                "dept_id": pa.Column(object, pa.Check.str_matches(r"^[A-Z]+_\d+$"), nullable=False),
                "cat_id": pa.Column(object, pa.Check.isin(CATEGORIES), nullable=False),
                "store_id": pa.Column(object, pa.Check.isin(STORES), nullable=False),
                "state_id": pa.Column(object, pa.Check.isin(STATES), nullable=False),
            },
            checks=[
                _row_check("item_in_dept", prefixed("item_id", "dept_id")),
                _row_check("dept_in_cat", prefixed("dept_id", "cat_id")),
                _row_check("store_in_state", prefixed("store_id", "state_id")),
                _row_check(
                    "id_matches_item_store",
                    lambda d: (
                        d["id"].str.replace(r"_(validation|evaluation)$", "", regex=True)
                        == d["item_id"] + "_" + d["store_id"]
                    ),
                ),
                _row_check("id_unique", lambda d: ~d["id"].duplicated(keep=False)),
            ],
        ),
    )


def check_sales_matrix(values: pd.DataFrame) -> pd.DataFrame:
    """Per-row reason flags for the day columns: every value must be an integer >= 0.

    Columns that pandas already parsed as numbers are checked directly; only columns that
    came back as strings (blank or malformed cells) are parsed, so the common case is fast.
    """
    text_cols = [c for c in values.columns if not pd.api.types.is_numeric_dtype(values[c])]
    present = values.notna().to_numpy()
    if text_cols:
        num = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        pos = [values.columns.get_loc(c) for c in text_cols]
        present[:, pos] &= (values[text_cols].astype(str).apply(lambda s: s.str.strip()) != "").to_numpy()
    else:
        num = values.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "sales:missing": (~present).any(axis=1),
            "sales:unparseable": (present & np.isnan(num)).any(axis=1),
            "sales:negative": (num < 0).any(axis=1),
            "sales:non_integer": (~np.isclose(np.nan_to_num(num) % 1, 0)).any(axis=1),
        },
        index=values.index,
    )


def prices_contract(
    stores: Collection[str], items: Collection[str], weeks: Collection[int]
) -> Contract:
    """Prices may only reference stores/items present in the sales file and calendar weeks."""
    return Contract(
        types={"store_id": "str", "item_id": "str", "wm_yr_wk": "int", "sell_price": "float"},
        schema=pa.DataFrameSchema(
            {
                "store_id": pa.Column(object, pa.Check.isin(list(stores)), nullable=False),
                "item_id": pa.Column(object, pa.Check.isin(list(items)), nullable=False),
                "wm_yr_wk": pa.Column(pd.Int64Dtype(), pa.Check.isin(list(weeks)), nullable=False),
                "sell_price": pa.Column(pd.Float64Dtype(), pa.Check.gt(0), nullable=False),
            },
            checks=[
                _row_check(
                    "store_item_week_unique",
                    lambda d: ~d.duplicated(["store_id", "item_id", "wm_yr_wk"], keep=False),
                )
            ],
        ),
    )
