"""Rolling-origin splits fixed by PROJECT_SPEC.md §5.1.

M5 days are 1-indexed: `d_1` is 2011-01-29. An origin is the last observed day; a fold
forecasts the `HORIZON` days that follow it.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from demand_intel import SPEC_VERSION

M5_START = date(2011, 1, 29)
LAST_LABELED_DAY = 1941  # last day in sales_train_evaluation.csv
HORIZON = 28
REPORTED_HORIZONS = (7, 14, 28)

BACKTEST_ORIGINS = (1773, 1801, 1829, 1857, 1885)
TEST_ORIGIN = 1913


def day_to_date(d: int) -> date:
    if d < 1:
        raise ValueError(f"M5 days start at 1, got {d}")
    return M5_START + timedelta(days=d - 1)


def date_to_day(dt: date) -> int:
    d = (dt - M5_START).days + 1
    if d < 1:
        raise ValueError(f"{dt} is before the M5 start date {M5_START}")
    return d


@dataclass(frozen=True)
class Fold:
    name: str
    origin: int

    @property
    def first_target(self) -> int:
        return self.origin + 1

    @property
    def last_target(self) -> int:
        return self.origin + HORIZON

    def target_days(self, horizon: int = HORIZON) -> range:
        if not 1 <= horizon <= HORIZON:
            raise ValueError(f"horizon must be in 1..{HORIZON}, got {horizon}")
        return range(self.first_target, self.origin + horizon + 1)


BACKTEST_FOLDS = tuple(Fold(f"fold_{i}", o) for i, o in enumerate(BACKTEST_ORIGINS, start=1))
TEST_FOLD = Fold("test", TEST_ORIGIN)


def split_manifest() -> dict[str, Any]:
    """Serializable record of every split, written to evaluation/split_manifest.json."""
    return {
        "spec_version": SPEC_VERSION,
        "horizon": HORIZON,
        "reported_horizons": list(REPORTED_HORIZONS),
        "folds": [
            {
                "name": f.name,
                "origin_day": f.origin,
                "origin_date": day_to_date(f.origin).isoformat(),
                "target_days": [f.first_target, f.last_target],
                "target_dates": [
                    day_to_date(f.first_target).isoformat(),
                    day_to_date(f.last_target).isoformat(),
                ],
            }
            for f in (*BACKTEST_FOLDS, TEST_FOLD)
        ],
    }
