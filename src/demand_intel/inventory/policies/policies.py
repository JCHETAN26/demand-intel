"""Order-up-to policies P0-P3 and the oracle (PROJECT_SPEC.md §8.2).

The level S_t covers demand over the protection interval: the lead time L plus the
one-day review period, i.e. days t+1 .. t+L+1.

- P0 fixed:          S = 28-day historical mean × (L + 7)
- P1 moving average: S = mean × (L+1) + z·sd·sqrt(L+1) from the 28 days before the window
- P2 point forecast: S = Σ forecast over the interval + z·σ_res·sqrt(L+1), σ_res from
                     out-of-sample residuals of earlier folds
- P3 probabilistic:  S = service-level quantile of simulated lead-time demand, from paths
                     drawn from the daily forecast quantiles with a calibrated day-to-day
                     correlation (sums of daily quantiles are not quantiles of sums)
- P* oracle:         S = actual demand over the interval (upper bound, not deployable)
"""

import numpy as np
from scipy import stats

from demand_intel.forecasting.evaluation.metrics import QUANTILES


def protection_days(lead_time: int) -> int:
    return lead_time + 1


def _interval(arr: np.ndarray, t: int, k: int) -> np.ndarray:
    """Columns t+1 .. t+k of arr (n, T); beyond the end, repeat the last 7 columns."""
    t_days = arr.shape[1]
    idx = np.arange(t + 1, t + 1 + k)
    over = idx >= t_days
    idx[over] = t_days - 7 + (idx[over] - t_days) % 7
    return arr[:, idx]


def fixed_policy(history: np.ndarray, t_days: int, lead_time: int) -> np.ndarray:
    level = history[:, -28:].mean(axis=1) * (lead_time + 7)
    return np.repeat(level[:, None], t_days, axis=1)


def moving_average_policy(
    history: np.ndarray, t_days: int, lead_time: int, service: float
) -> np.ndarray:
    k = protection_days(lead_time)
    z = stats.norm.ppf(service)
    h = history[:, -28:]
    level = h.mean(axis=1) * k + z * h.std(axis=1) * np.sqrt(k)
    return np.repeat(level[:, None], t_days, axis=1)


def point_forecast_policy(
    point: np.ndarray, sigma: np.ndarray, lead_time: int, service: float
) -> np.ndarray:
    k = protection_days(lead_time)
    z = stats.norm.ppf(service)
    t_days = point.shape[1]
    return np.stack(
        [_interval(point, t, k).sum(axis=1) + z * sigma * np.sqrt(k) for t in range(t_days)], axis=1
    )


def oracle_policy(demand: np.ndarray, lead_time: int) -> np.ndarray:
    k = protection_days(lead_time)
    return np.stack([_interval(demand, t, k).sum(axis=1) for t in range(demand.shape[1])], axis=1)


def _inverse_cdf(quantiles: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Piecewise-linear inverse CDF through the forecast quantiles, flat in the tails.

    quantiles: (n, Q) at levels QUANTILES; u: (n, m) uniforms → (n, m) demand draws.
    """
    taus = np.array([0.0, *QUANTILES, 1.0])
    qs = np.concatenate([quantiles[:, :1], quantiles, quantiles[:, -1:]], axis=1)
    pos = np.clip(np.searchsorted(taus, u, side="right") - 1, 0, len(taus) - 2)
    lo, hi = taus[pos], taus[pos + 1]
    frac = (u - lo) / np.maximum(hi - lo, 1e-12)
    rows = np.arange(len(quantiles))[:, None]
    draws: np.ndarray = qs[rows, pos] + frac * (qs[rows, pos + 1] - qs[rows, pos])
    return draws


def lead_time_quantile(
    daily_q: np.ndarray, rho: float, level: float, n_paths: int = 400, seed: int = 0
) -> np.ndarray:
    """Quantile `level` of the sum over k days, by Monte Carlo with a one-factor Gaussian
    copula: u_d = Φ(√ρ·Z + √(1-ρ)·ε_d). daily_q: (n, k, Q)."""
    rng = np.random.default_rng(seed)
    n, k, _ = daily_q.shape
    common = rng.standard_normal((n, n_paths))
    total = np.zeros((n, n_paths))
    for d in range(k):
        eps = rng.standard_normal((n, n_paths))
        u = stats.norm.cdf(np.sqrt(rho) * common + np.sqrt(1 - rho) * eps)
        total += _inverse_cdf(daily_q[:, d, :], u)
    out: np.ndarray = np.quantile(total, level, axis=1)
    return out


def probabilistic_policy(
    quantiles: np.ndarray, lead_time: int, service: float, rho: float, seed: int = 0
) -> np.ndarray:
    """quantiles: (n, T, Q) daily forecast quantiles over the window."""
    k = protection_days(lead_time)
    t_days = quantiles.shape[1]
    levels = []
    for t in range(t_days):
        idx = np.arange(t + 1, t + 1 + k)
        over = idx >= t_days
        idx[over] = t_days - 7 + (idx[over] - t_days) % 7
        levels.append(lead_time_quantile(quantiles[:, idx, :], rho, service, seed=seed + t))
    return np.stack(levels, axis=1)


def calibrate_rho(
    quantiles: np.ndarray,
    actual: np.ndarray,
    lead_time: int,
    level: float,
    grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> tuple[float, dict[float, float]]:
    """Pick ρ whose lead-time-sum `level` quantile is closest to nominal coverage on a
    calibration window (an earlier, out-of-sample fold)."""
    k = protection_days(lead_time)
    starts = np.arange(0, actual.shape[1] - k + 1, 3)
    coverage = {}
    for rho in grid:
        hits, total = 0.0, 0
        for s in starts:
            q = lead_time_quantile(quantiles[:, s : s + k, :], rho, level, n_paths=300, seed=int(s))
            hits += float((actual[:, s : s + k].sum(axis=1) <= q).sum())
            total += len(q)
        coverage[rho] = hits / total
    best = min(coverage, key=lambda r: abs(coverage[r] - level))
    return best, coverage
