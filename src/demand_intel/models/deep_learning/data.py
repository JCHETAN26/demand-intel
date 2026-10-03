"""Windowed sequence data for the deep-learning forecasters (plan Phases 14-15).

A sample is (series i, origin o): the encoder sees days (o - ENC, o] and the decoder the
HORIZON days after o. Inputs follow the TFT split (spec §2.2, §5.3):

- static:        item, dept, cat, store, state (embedded)
- observed past: scaled sales, nonzero indicator          (encoder only)
- known future:  weekday, month, day of month, event, SNAP, log price ratio, on-sale flag
                 (encoder and decoder)

Sales are divided by a per-sample scale (mean of the encoder window + 1) so one network
serves series of any volume; outputs are multiplied back.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from demand_intel.data import M5Data
from demand_intel.forecasting.backtesting.splits import HORIZON

ENC = 56
STATIC = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
N_KNOWN_CONT = 5  # mday/31, snap, log price ratio, on-sale flag, event-today flag


@dataclass
class SequenceArrays:
    """Arrays for a set of series over the full calendar; built once per fold."""

    sales: np.ndarray  # (n, n_obs) float32, observed up to the origin only
    static: np.ndarray  # (n, len(STATIC)) int64 codes
    static_cardinality: tuple[int, ...]
    wday: np.ndarray  # (n_cal,) int64, 0..6
    month: np.ndarray  # (n_cal,) int64, 0..11
    event: np.ndarray  # (n_cal,) int64, 0 = none
    n_events: int
    known: np.ndarray  # (n, n_cal, N_KNOWN_CONT) float32

    @classmethod
    def build(cls, m5: M5Data, rows: np.ndarray) -> "SequenceArrays":
        series = m5.series.iloc[rows]
        cal = m5.calendar.sort_values("d")
        static, card = [], []
        for col in STATIC:
            levels = pd.Index(sorted(m5.series[col].unique()))
            static.append(levels.get_indexer(series[col]))
            card.append(len(levels))
        events = cal["event_name_1"].fillna("none")
        ev_levels = pd.Index(["none", *sorted(set(events) - {"none"})])
        events = pd.Index(events)
        assert m5.prices is not None
        prices = m5.prices[rows]
        with np.errstate(all="ignore"):
            ref = np.nanmedian(prices, axis=1, keepdims=True)
            log_ratio = np.nan_to_num(np.log(prices / ref), nan=0.0)
        snap = np.stack([cal[f"snap_{s}"].to_numpy() for s in ("CA", "TX", "WI")])
        state = series["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
        n, n_cal = len(rows), len(cal)
        known = np.zeros((n, n_cal, N_KNOWN_CONT), dtype=np.float32)
        known[:, :, 0] = pd.to_datetime(cal["date"]).dt.day.to_numpy()[None, :] / 31.0
        known[:, :, 1] = snap[state]
        known[:, :, 2] = log_ratio
        known[:, :, 3] = ~np.isnan(prices)
        known[:, :, 4] = np.asarray(events != "none")[None, :]
        return cls(
            sales=m5.sales[rows].astype(np.float32),
            static=np.stack(static, axis=1).astype(np.int64),
            static_cardinality=tuple(card),
            wday=(cal["wday"].to_numpy() - 1).astype(np.int64),
            month=(cal["month"].to_numpy() - 1).astype(np.int64),
            event=ev_levels.get_indexer(events).astype(np.int64),
            n_events=len(ev_levels),
            known=known,
        )


def window(arr: SequenceArrays, i: int, origin: int, with_target: bool) -> dict[str, torch.Tensor]:
    """One sample; `origin` is the last observed day (1-based)."""
    if origin < ENC or origin > arr.sales.shape[1]:
        raise ValueError(f"origin {origin} outside the observed history")
    if with_target and origin + HORIZON > arr.sales.shape[1]:
        raise ValueError("target window extends past the observed data")
    enc = slice(origin - ENC, origin)  # 0-based columns of days origin-ENC+1 .. origin
    dec = slice(origin, origin + HORIZON)
    hist = arr.sales[i, enc]
    scale = np.float32(hist.mean() + 1.0)
    full = slice(origin - ENC, origin + HORIZON)
    sample = {
        "static": torch.from_numpy(arr.static[i]),
        "past": torch.from_numpy(np.stack([hist / scale, (hist > 0).astype(np.float32)], axis=1)),
        "wday": torch.from_numpy(arr.wday[full]),
        "month": torch.from_numpy(arr.month[full]),
        "event": torch.from_numpy(arr.event[full]),
        "known": torch.from_numpy(arr.known[i, full]),
        "scale": torch.tensor(scale),
    }
    if with_target:
        sample["y"] = torch.from_numpy(arr.sales[i, dec] / scale)
    return sample


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Training windows: every series × a set of origins (all targets observed)."""

    def __init__(
        self,
        arr: SequenceArrays,
        origins: np.ndarray,
        rng: np.random.Generator | None = None,
        samples: int | None = None,
    ) -> None:
        self.arr = arr
        pairs = np.array(np.meshgrid(np.arange(len(arr.sales)), origins)).T.reshape(-1, 2)
        if samples is not None and rng is not None and samples < len(pairs):
            pairs = pairs[rng.choice(len(pairs), samples, replace=False)]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        i, o = self.pairs[k]
        return window(self.arr, int(i), int(o), with_target=True)
