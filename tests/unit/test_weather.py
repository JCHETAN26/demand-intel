import json
from pathlib import Path

import pandas as pd
import pytest

from demand_intel.config import REPO_ROOT
from demand_intel.pipelines.ingestion import weather
from demand_intel.pipelines.ingestion.weather import build_state_weather, load_locations

LOCATIONS = REPO_ROOT / "configs" / "weather_locations.yaml"


def _write_metro(path: Path, days: pd.DatetimeIndex, tmax: list[float | None]) -> None:
    n = len(days)
    daily = {
        "time": [d.strftime("%Y-%m-%d") for d in days],
        "temperature_2m_max": tmax,
        "temperature_2m_min": [0.0] * n,
        "temperature_2m_mean": [5.0] * n,
        "precipitation_sum": [1.0] * n,
        "snowfall_sum": [0.0] * n,
    }
    path.write_text(json.dumps({"daily": daily}))


@pytest.fixture
def short_period(monkeypatch: pytest.MonkeyPatch) -> pd.DatetimeIndex:
    monkeypatch.setattr(weather, "START", pd.Timestamp("2011-01-29").date())
    monkeypatch.setattr(weather, "END", pd.Timestamp("2011-02-07").date())
    return pd.date_range("2011-01-29", "2011-02-07")


def test_locations_cover_three_states() -> None:
    version, metros = load_locations(LOCATIONS)
    assert version == "weather_locations.v1"
    assert {m.state for m in metros} == {"CA", "TX", "WI"}
    assert all(m.weight > 0 for m in metros)


def test_weighted_average_and_short_gaps(tmp_path: Path, short_period: pd.DatetimeIndex) -> None:
    _, metros = load_locations(LOCATIONS)
    for m in metros:
        tmax: list[float | None] = [10.0] * len(short_period)
        if m.name == "los_angeles":
            tmax = [20.0] * len(short_period)
            tmax[3] = None  # one missing day: other metros carry it
        _write_metro(tmp_path / f"{m.state}_{m.name}.json", short_period, tmax)

    report = build_state_weather(tmp_path, LOCATIONS, tmp_path / "out.parquet")
    df = pd.read_parquet(tmp_path / "out.parquet")
    assert report.rows == 3 * len(short_period)
    ca = df[df["state_id"] == "CA"].reset_index(drop=True)
    w = {m.name: m.weight for m in metros if m.state == "CA"}
    others = w["san_francisco"] + w["san_diego"]
    expected = (20 * w["los_angeles"] + 10 * others) / sum(w.values())
    assert ca.loc[0, "tmax_c"] == pytest.approx(expected)
    assert ca.loc[3, "tmax_c"] == pytest.approx(10.0)  # LA missing -> others only
    assert report.max_gap_days == 1


def test_long_gap_fails(tmp_path: Path, short_period: pd.DatetimeIndex) -> None:
    _, metros = load_locations(LOCATIONS)
    for m in metros:
        tmax: list[float | None] = [10.0] * len(short_period)
        if m.name == "milwaukee":
            tmax[2:6] = [None] * 4
        _write_metro(tmp_path / f"{m.state}_{m.name}.json", short_period, tmax)
    with pytest.raises(RuntimeError, match="weather validation failed"):
        build_state_weather(tmp_path, LOCATIONS, tmp_path / "out.parquet")


def test_out_of_range_values_are_dropped(tmp_path: Path, short_period: pd.DatetimeIndex) -> None:
    _, metros = load_locations(LOCATIONS)
    for m in metros:
        tmax: list[float | None] = [10.0] * len(short_period)
        if m.name == "houston":
            tmax[0] = 99.0
        _write_metro(tmp_path / f"{m.state}_{m.name}.json", short_period, tmax)
    report = build_state_weather(tmp_path, LOCATIONS, tmp_path / "out.parquet")
    df = pd.read_parquet(tmp_path / "out.parquet")
    assert report.out_of_range_cells == 1
    assert df.loc[df["state_id"] == "TX", "tmax_c"].iloc[0] == pytest.approx(10.0)
