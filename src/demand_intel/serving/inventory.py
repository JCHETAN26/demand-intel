"""Replenishment recommendation for one series (plan Phase 26).

Lead-time demand covers days 1..L+1 (lead time plus a one-day review). Its distribution is
simulated from the daily forecast quantiles with the one-factor copula used by policy P3;
the correlation comes from the backtest calibration when available.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from demand_intel.inventory.policies.policies import _inverse_cdf

DEFAULT_RHO = 0.3


@dataclass(frozen=True)
class Recommendation:
    lead_time_days: int
    service_level: float
    expected_demand: float
    p50: float
    p90: float
    target_level: float  # order-up-to = service-level quantile of lead-time demand
    safety_stock: float
    inventory_position: float
    recommended_order: int
    stockout_probability: float  # P(lead-time demand > inventory position) without ordering
    rho: float


def recommend(
    daily_q: np.ndarray,
    daily_mean: np.ndarray,
    on_hand: float,
    on_order: float,
    lead_time: int,
    service_level: float,
    rho: float = DEFAULT_RHO,
    n_paths: int = 2000,
    seed: int = 0,
) -> Recommendation:
    """daily_q: (H, Q) quantiles; daily_mean: (H,)."""
    k = min(lead_time + 1, daily_q.shape[0])
    rng = np.random.default_rng(seed)
    # One vectorized draw for all k days (the loop over days dominated request time).
    z = np.sqrt(rho) * rng.standard_normal((1, n_paths)) + np.sqrt(1 - rho) * rng.standard_normal(
        (k, n_paths)
    )
    total = _inverse_cdf(daily_q[:k], stats.norm.cdf(z)).sum(axis=0)
    target = float(np.quantile(total, service_level))
    position = on_hand + on_order
    expected = float(daily_mean[:k].sum())
    return Recommendation(
        lead_time_days=lead_time,
        service_level=service_level,
        expected_demand=expected,
        p50=float(np.quantile(total, 0.5)),
        p90=float(np.quantile(total, 0.9)),
        target_level=target,
        safety_stock=max(target - expected, 0.0),
        inventory_position=position,
        recommended_order=int(max(0, np.ceil(target - position))),
        stockout_probability=float((total > position).mean()),
        rho=rho,
    )


def calibrated_rho(summary: dict[str, object], lead_time: int, service_level: float) -> float:
    cal = summary.get("inventory_calibration")
    if isinstance(cal, dict):
        entry = cal.get("rho", {}).get(f"L={lead_time},SL={service_level}")
        if isinstance(entry, dict) and "rho" in entry:
            return float(entry["rho"])
    return DEFAULT_RHO
