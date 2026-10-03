"""The M5 12-level aggregation hierarchy as a sparse summing matrix.

Row r of `S` sums the bottom-level (item × store) series that belong to aggregate series r,
so `S @ bottom` gives every level at once. Levels follow the M5 competition guide.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total", ()),
    ("state", ("state_id",)),
    ("store", ("store_id",)),
    ("category", ("cat_id",)),
    ("department", ("dept_id",)),
    ("state_category", ("state_id", "cat_id")),
    ("state_department", ("state_id", "dept_id")),
    ("store_category", ("store_id", "cat_id")),
    ("store_department", ("store_id", "dept_id")),
    ("item", ("item_id",)),
    ("item_state", ("item_id", "state_id")),
    ("item_store", ("item_id", "store_id")),
)


@dataclass(frozen=True)
class Hierarchy:
    S: sparse.csr_matrix  # (n_aggregate, n_bottom)
    level: np.ndarray  # level index (0..11) of each aggregate row
    keys: list[str]  # human-readable key of each aggregate row

    @property
    def n_levels(self) -> int:
        return len(LEVELS)


def build_hierarchy(series: pd.DataFrame) -> Hierarchy:
    n = len(series)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    levels: list[int] = []
    keys: list[str] = []
    offset = 0
    for li, (name, by) in enumerate(LEVELS):
        if by:
            key = series[list(by)].astype(str).agg("/".join, axis=1)
        else:
            key = pd.Series(["total"] * n, index=series.index)
        codes, uniques = pd.factorize(key, sort=True)
        rows.append(codes + offset)
        cols.append(np.arange(n))
        levels.extend([li] * len(uniques))
        keys.extend(f"{name}:{u}" for u in uniques)
        offset += len(uniques)
    summing = sparse.csr_matrix(
        (np.ones(n * len(LEVELS), dtype=np.float32), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, n),
    )
    return Hierarchy(S=summing, level=np.array(levels), keys=keys)
