"""In-memory view of the serving bundle (see pipelines/inference/serving.py)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_intel.forecasting.backtesting.splits import HORIZON
from demand_intel.forecasting.evaluation.metrics import QCOLS


@dataclass
class ServingStore:
    series: pd.DataFrame
    index: dict[str, int]  # series id (item_store) → row
    mean: np.ndarray  # (n, H)
    quantiles: np.ndarray  # (n, H, Q)
    forecast_days: np.ndarray  # (H,)
    history_days: np.ndarray
    history_sales: np.ndarray  # (n, D)
    history_price: np.ndarray  # (n, D)
    future_price: np.ndarray  # (n, H)
    drivers: np.ndarray  # (n, H, F)
    driver_names: list[str]
    summary: dict[str, Any]
    risk: pd.DataFrame
    calendar: dict[int, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ServingStore":
        series = pd.read_parquet(path / "series.parquet")
        n = len(series)
        fc = pd.read_parquet(path / "forecast.parquet").sort_values(["series", "d"])
        hist = pd.read_parquet(path / "history.parquet").sort_values(["series", "d"])
        fp = pd.read_parquet(path / "future_price.parquet").sort_values(["series", "d"])
        dr = pd.read_parquet(path / "drivers.parquet").sort_values(["series", "d"])
        names = [c for c in dr.columns if c not in ("series", "d")]
        n_hist = len(hist) // n
        summary = json.loads((path / "summary.json").read_text())
        return cls(
            series=series,
            index={sid: i for i, sid in enumerate(series["id"])},
            mean=fc["mean"].to_numpy(np.float32).reshape(n, HORIZON),
            quantiles=np.stack([fc[c].to_numpy(np.float32) for c in QCOLS], axis=-1).reshape(
                n, HORIZON, len(QCOLS)
            ),
            forecast_days=fc["d"].to_numpy()[:HORIZON],
            history_days=hist["d"].to_numpy()[:n_hist],
            history_sales=hist["sales"].to_numpy().reshape(n, n_hist),
            history_price=hist["price"].to_numpy().reshape(n, n_hist),
            future_price=fp["price"].to_numpy().reshape(n, HORIZON),
            drivers=dr[names].to_numpy(np.float32).reshape(n, HORIZON, len(names)),
            driver_names=names,
            summary=summary,
            risk=pd.read_parquet(path / "risk.parquet")
            .sort_values("series")
            .reset_index(drop=True),
            calendar={int(k): v for k, v in summary.get("calendar", {}).items()},
        )

    def row(self, item_id: str, store_id: str) -> int | None:
        return self.index.get(f"{item_id}_{store_id}")

    def date(self, d: int) -> str:
        return self.calendar.get(int(d), str(d))
