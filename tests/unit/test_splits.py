from datetime import date
from itertools import pairwise

import pytest

from demand_intel.forecasting.backtesting.splits import (
    BACKTEST_FOLDS,
    HORIZON,
    LAST_LABELED_DAY,
    TEST_FOLD,
    date_to_day,
    day_to_date,
    split_manifest,
)


def test_calendar_anchors() -> None:
    assert day_to_date(1) == date(2011, 1, 29)
    assert day_to_date(1913) == date(2016, 4, 24)
    assert day_to_date(1941) == date(2016, 5, 22)
    assert date_to_day(date(2016, 4, 25)) == 1914


def test_day_roundtrip_and_bounds() -> None:
    for d in (1, 500, 1969):
        assert date_to_day(day_to_date(d)) == d
    with pytest.raises(ValueError):
        day_to_date(0)


def test_folds_are_contiguous_and_end_at_test() -> None:
    folds = (*BACKTEST_FOLDS, TEST_FOLD)
    for prev, nxt in pairwise(folds):
        assert nxt.first_target == prev.last_target + 1
    assert TEST_FOLD.last_target == LAST_LABELED_DAY


def test_backtest_never_reaches_test_window() -> None:
    assert all(f.last_target <= TEST_FOLD.origin for f in BACKTEST_FOLDS)


def test_backtest_covers_christmas() -> None:
    xmas = date_to_day(date(2015, 12, 25))
    assert any(xmas in f.target_days() for f in BACKTEST_FOLDS)


def test_target_days_respects_horizon() -> None:
    assert len(TEST_FOLD.target_days(7)) == 7
    assert len(TEST_FOLD.target_days()) == HORIZON
    with pytest.raises(ValueError):
        TEST_FOLD.target_days(29)


def test_manifest_shape() -> None:
    m = split_manifest()
    assert [f["name"] for f in m["folds"]][-1] == "test"
    assert m["folds"][-1]["target_dates"] == ["2016-04-25", "2016-05-22"]
