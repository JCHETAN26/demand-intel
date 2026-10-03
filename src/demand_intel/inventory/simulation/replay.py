"""Policy comparison over historical windows (PROJECT_SPEC.md §8, plan Phase 27).

- Calibration window: fold_1. Its out-of-sample forecasts set P3's copula correlation and
  P2's residual standard deviation; it is not scored.
- Evaluation windows: fold_2 .. fold_5 and test (5 × 28 days). Each window starts every
  policy at its own order-up-to level (steady state).
- Forecasts come from the backtest files of the chosen model; every policy faces the same
  demand, prices and scenario.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_intel import SEED
from demand_intel.data import M5Data
from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, HORIZON, TEST_FOLD
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.inventory.policies.policies import (
    calibrate_rho,
    fixed_policy,
    moving_average_policy,
    oracle_policy,
    point_forecast_policy,
    probabilistic_policy,
)
from demand_intel.inventory.simulation.simulator import Scenario, simulate

CALIBRATION = BACKTEST_FOLDS[0]
EVALUATION = (*BACKTEST_FOLDS[1:], TEST_FOLD)
DEFAULT = Scenario()
SENSITIVITIES: tuple[Scenario, ...] = (
    *(replace(DEFAULT, stockout_cost_multiple=p) for p in (0.5, 2.0, 5.0)),
    replace(DEFAULT, service_level=0.90),
    replace(DEFAULT, service_level=0.98),
    replace(DEFAULT, lead_time=2),
)


@dataclass
class FoldForecast:
    point: np.ndarray  # (n, H)
    quantiles: np.ndarray  # (n, H, Q)


def load_forecast(
    forecast_dir: Path, model: str, fold: str, rows: np.ndarray, source: str = "full"
) -> FoldForecast:
    """Load a backtest forecast from `source` population ("full" files cover every series)
    and select `rows`; falls back to the other population if needed."""
    for population in (source, "panel" if source == "full" else "full"):
        path = forecast_dir / model / population / f"{fold}.npz"
        if path.exists():
            z = np.load(path)
            pos = pd.Series(np.arange(len(z["rows"])), index=z["rows"]).reindex(rows)
            if pos.isna().any():
                continue
            idx = pos.to_numpy(dtype=int)
            q = np.stack([z[f"q{round(t * 100):02d}"][idx] for t in QUANTILES], axis=-1)
            return FoldForecast(point=z["point"][idx], quantiles=q)
    raise FileNotFoundError(f"no {model} forecast for {fold} covering the requested series")


def run_replay(
    m5: M5Data,
    rows: np.ndarray,
    model: str,
    forecast_dir: Path,
    scenarios: Sequence[Scenario] = (DEFAULT, *SENSITIVITIES),
    n_boot: int = 1000,
    source: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Returns (per window × scenario × policy metrics, paired cost differences, calibration)."""
    assert m5.prices is not None
    cal = load_forecast(forecast_dir, model, CALIBRATION.name, rows, source)
    o = CALIBRATION.origin
    cal_actual = m5.sales[rows, o : o + HORIZON].astype(float)
    residual = [cal_actual - cal.point]
    calibration: dict[str, Any] = {
        "model": model,
        "forecast_population": source,
        "calibration_fold": CALIBRATION.name,
        "rho": {},
    }

    metrics: list[dict[str, Any]] = []
    per_series: dict[tuple[str, str], np.ndarray] = {}
    for sc in scenarios:
        key = (sc.lead_time, sc.service_level)
        if key not in calibration["rho"]:
            rho, cov = calibrate_rho(cal.quantiles, cal_actual, sc.lead_time, sc.service_level)
            calibration["rho"][key] = {"rho": rho, "coverage": cov}
    resid_hist = list(residual)
    for fold in EVALUATION:
        o = fold.origin
        demand = m5.sales[rows, o : o + HORIZON].astype(float)
        history = m5.sales[rows, :o].astype(float)
        price = np.nanmean(m5.prices[rows, o : o + HORIZON], axis=1)
        price = np.nan_to_num(price, nan=float(np.nanmedian(price)))
        fc = load_forecast(forecast_dir, model, fold.name, rows, source)
        sigma = np.concatenate(resid_hist, axis=1).std(axis=1)  # earlier folds only
        for sc in scenarios:
            rho = calibration["rho"][(sc.lead_time, sc.service_level)]["rho"]
            levels = {
                "P0_fixed": fixed_policy(history, HORIZON, sc.lead_time),
                "P1_moving_average": moving_average_policy(
                    history, HORIZON, sc.lead_time, sc.service_level
                ),
                "P2_point_forecast": point_forecast_policy(
                    fc.point, sigma, sc.lead_time, sc.service_level
                ),
                "P3_probabilistic": probabilistic_policy(
                    fc.quantiles, sc.lead_time, sc.service_level, rho, seed=SEED
                ),
                "P*_oracle": oracle_policy(demand, sc.lead_time),
            }
            for policy, s in levels.items():
                res = simulate(demand, s, price, sc)
                metrics.append(
                    {
                        "scenario": sc.label,
                        "window": fold.name,
                        "policy": policy,
                        **res.summary(price),
                    }
                )
                k = (sc.label, policy)
                cost = res.holding_cost + res.stockout_cost
                per_series[k] = per_series.get(k, 0) + cost
        resid_hist.append(demand - fc.point)

    rng = np.random.default_rng(SEED)
    n = len(rows)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    diffs = []
    for sc in scenarios:
        for a, b in (
            ("P3_probabilistic", "P1_moving_average"),
            ("P3_probabilistic", "P2_point_forecast"),
            ("P2_point_forecast", "P1_moving_average"),
            ("P1_moving_average", "P0_fixed"),
        ):
            d = per_series[(sc.label, a)] - per_series[(sc.label, b)]
            boots = d[boot_idx].sum(axis=1)
            diffs.append(
                {
                    "scenario": sc.label,
                    "comparison": f"{a} - {b}",
                    "cost_difference": float(d.sum()),
                    "relative": float(d.sum() / per_series[(sc.label, b)].sum()),
                    "ci_low": float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                }
            )
    calibration["rho"] = {f"L={k[0]},SL={k[1]}": v for k, v in calibration["rho"].items()}
    return pd.DataFrame(metrics), pd.DataFrame(diffs), calibration
