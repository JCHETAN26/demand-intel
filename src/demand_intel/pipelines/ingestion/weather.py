"""Historical daily weather per state (PROJECT_SPEC.md §2.3).

Raw Open-Meteo responses are stored per metro in `data/raw/weather/` with a checksum
manifest; the validated output is one row per state × date in
`data/validated/weather/weather_state_daily.parquet`, a population-weighted average of each
state's metros. Derived features (extremes, anomalies) are computed later from training
data only, so they are not part of ingestion.
"""

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml

from demand_intel.pipelines.ingestion.download import MANIFEST, sha256_file

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = date(2011, 1, 29), date(2016, 6, 19)  # M5 calendar d_1..d_1969
COLUMNS = {
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "temperature_2m_mean": "tmean_c",
    "precipitation_sum": "precip_mm",
    "snowfall_sum": "snow_cm",
}
# Plausible physical ranges; anything outside is a data error, not weather.
RANGES = {
    "tmax_c": (-45.0, 55.0),
    "tmin_c": (-50.0, 45.0),
    "tmean_c": (-45.0, 50.0),
    "precip_mm": (0.0, 500.0),
    "snow_cm": (0.0, 200.0),
}
MAX_GAP_DAYS = 2


@dataclass(frozen=True)
class Metro:
    state: str
    name: str
    lat: float
    lon: float
    weight: float
    timezone: str


def load_locations(path: Path) -> tuple[str, list[Metro]]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    metros = [
        Metro(state, m["name"], m["lat"], m["lon"], m["weight"], s["timezone"])
        for state, s in cfg["states"].items()
        for m in s["metros"]
    ]
    return str(cfg["version"]), metros


def _fetch(metro: Metro, client: httpx.Client) -> bytes:
    params: dict[str, str | float] = {
        "latitude": metro.lat,
        "longitude": metro.lon,
        "start_date": START.isoformat(),
        "end_date": END.isoformat(),
        "daily": ",".join(COLUMNS),
        "timezone": metro.timezone,
    }
    for attempt in range(5):
        r = client.get(ARCHIVE_URL, params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r.content
        time.sleep(2**attempt)
    r.raise_for_status()
    return r.content


def download_weather(raw_dir: Path, locations: Path, force: bool = False) -> Path:
    """Fetch each metro once; files whose manifest checksum matches are not refetched."""
    out = raw_dir / "weather"
    out.mkdir(parents=True, exist_ok=True)
    version, metros = load_locations(locations)
    manifest_path = out / MANIFEST
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    )
    with httpx.Client() as client:
        for m in metros:
            name = f"{m.state}_{m.name}.json"
            path = out / name
            recorded = manifest["files"].get(name)
            if not force and path.exists() and recorded and recorded["sha256"] == sha256_file(path):
                continue
            path.write_bytes(_fetch(m, client))
            manifest["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest |= {
        "dataset": "weather",
        "source": ARCHIVE_URL,
        "locations_version": version,
        "period": [START.isoformat(), END.isoformat()],
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _metro_frame(path: Path) -> pd.DataFrame:
    daily = json.loads(path.read_text())["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for src, dst in COLUMNS.items():
        df[dst] = pd.to_numeric(pd.Series(daily[src]), errors="coerce")
    return df


@dataclass(frozen=True)
class WeatherReport:
    rows: int
    missing_cells: int
    max_gap_days: int
    out_of_range_cells: int
    checks: dict[str, bool]


def _longest_gap(missing: pd.Series) -> int:
    run = best = 0
    for m in missing.to_numpy():
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def build_state_weather(raw_weather: Path, locations: Path, out_path: Path) -> WeatherReport:
    """Validate the per-metro files and write the state × day weighted average."""
    _, metros = load_locations(locations)
    expected = pd.date_range(START, END, freq="D")
    frames, missing_cells, gaps, out_of_range = [], 0, 0, 0
    for state in sorted({m.state for m in metros}):
        parts = [m for m in metros if m.state == state]
        weights = np.array([m.weight for m in parts], dtype=float)
        stacked = []
        for m in parts:
            df = _metro_frame(raw_weather / f"{m.state}_{m.name}.json").set_index("date")
            df = df.reindex(expected)
            for col, (lo, hi) in RANGES.items():
                bad = df[col].notna() & ~df[col].between(lo, hi)
                out_of_range += int(bad.sum())
                df.loc[bad, col] = np.nan
            missing = df.isna().any(axis=1)
            missing_cells += int(df.isna().sum().sum())
            gaps = max(gaps, _longest_gap(missing))
            stacked.append(df.to_numpy())
        arr = np.stack(stacked)  # metros × days × vars
        w = np.where(np.isnan(arr), 0.0, weights[:, None, None])
        avg = np.nansum(arr * w, axis=0) / np.where(w.sum(axis=0) > 0, w.sum(axis=0), np.nan)
        state_df = pd.DataFrame(avg, columns=list(RANGES), index=expected).rename_axis("date")
        state_df.insert(0, "state_id", state)
        frames.append(state_df.reset_index())

    result = pd.concat(frames, ignore_index=True)
    result["state_id"] = result["state_id"].astype("category")
    checks = {
        "coverage_complete": len(result) == 3 * len(expected),
        "no_gap_over_max": gaps <= MAX_GAP_DAYS,
        "no_missing_after_weighting": not result[list(RANGES)].isna().any().any(),
    }
    report = WeatherReport(len(result), missing_cells, gaps, out_of_range, checks)
    if not all(checks.values()):
        raise RuntimeError(f"weather validation failed: {report}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return report
