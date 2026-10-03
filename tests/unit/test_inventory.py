import numpy as np
import pytest

from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.inventory.policies.policies import (
    _inverse_cdf,
    calibrate_rho,
    fixed_policy,
    lead_time_quantile,
    oracle_policy,
    point_forecast_policy,
)
from demand_intel.inventory.simulation.simulator import Scenario, simulate


def test_units_are_conserved_and_orders_arrive_after_lead_time() -> None:
    demand = np.array([[3.0, 0, 5, 2, 0, 4, 1, 0, 2, 3]])
    s = np.full_like(demand, 6.0)
    res = simulate(
        demand, s, np.array([2.0]), Scenario(lead_time=2), initial_on_hand=np.array([4.0])
    )
    assert np.allclose(res.sales + res.lost, demand)
    # day 0: stock 4, demand 3 → on hand 1, position 1 → order 5, arriving at day 2
    assert res.orders[0, 0] == 5 and res.on_hand[0, 0] == 1
    assert res.on_hand[0, 1] == 1  # nothing arrives on day 1
    assert res.on_hand[0, 2] == pytest.approx(1 + 5 - 5)  # arrival on day 2 covers demand 5
    assert (res.on_hand >= 0).all()


def test_zero_demand_costs_only_holding() -> None:
    demand = np.zeros((2, 14))
    res = simulate(demand, np.full_like(demand, 3.0), np.array([1.0, 2.0]), Scenario())
    assert res.stockout_cost.sum() == 0 and res.holding_cost.sum() > 0
    assert res.summary(np.array([1.0, 2.0]))["fill_rate"] == 1.0


def test_oracle_beats_fixed_policy_on_lumpy_demand() -> None:
    rng = np.random.default_rng(0)
    demand = rng.poisson(2, size=(50, 56)).astype(float) * rng.integers(0, 2, size=(50, 56))
    price = np.full(50, 3.0)
    sc = Scenario(lead_time=2, stockout_cost_multiple=2.0)
    oracle = simulate(demand, oracle_policy(demand, 2), price, sc).summary(price)
    fixed = simulate(demand, fixed_policy(demand, 56, 2), price, sc).summary(price)
    assert oracle["total_cost"] < fixed["total_cost"]
    assert oracle["fill_rate"] >= 0.95


def test_inverse_cdf_passes_through_quantiles() -> None:
    q = np.array([[0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]])
    for j, tau in enumerate(QUANTILES):
        assert _inverse_cdf(q, np.array([[tau]]))[0, 0] == pytest.approx(q[0, j])


def test_correlation_widens_lead_time_sums() -> None:
    daily = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (20, 7, 1))
    q_ind = lead_time_quantile(daily, rho=0.0, level=0.95, n_paths=4000)
    q_cor = lead_time_quantile(daily, rho=0.8, level=0.95, n_paths=4000)
    assert (q_cor > q_ind).mean() > 0.9


def test_point_policy_adds_safety_stock() -> None:
    point = np.ones((3, 28))
    s = point_forecast_policy(point, np.array([0.0, 1.0, 2.0]), lead_time=6, service=0.95)
    assert np.allclose(s[0], 7.0)
    assert (s[2] > s[1]).all() and (s[1] > s[0]).all()


def test_calibrate_rho_picks_correlation_for_correlated_truth() -> None:
    rng = np.random.default_rng(1)
    n, t = 300, 28
    level_shift = rng.gamma(2.0, 1.0, size=(n, 1))  # series-level shock → correlated days
    actual = rng.poisson(level_shift * np.ones((n, t)))
    mean = np.full((n, t), 2.0)
    quant = np.stack(
        [
            np.quantile(rng.poisson(rng.gamma(2.0, 1.0, 20000)), q) * np.ones_like(mean)
            for q in QUANTILES
        ],
        axis=-1,
    )
    rho, cov = calibrate_rho(quant, actual.astype(float), lead_time=6, level=0.9)
    assert rho > 0.0
    assert abs(cov[rho] - 0.9) < abs(cov[0.0] - 0.9)
