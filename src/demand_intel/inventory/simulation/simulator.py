"""Historical inventory replay (PROJECT_SPEC.md §8, plan Phases 25-27).

All costs, lead times and starting stock are SCENARIO assumptions, not observed data.

Mechanics per series and day (lost sales, daily review, lead time L):
  receive the order placed L days ago → observe demand → sell min(stock, demand), the rest
  is lost → decide: order max(0, S_t - inventory position) with inventory position =
  on hand + on order.

Policies differ only in how they set the order-up-to level S_t (see `policies`). Every
policy sees the same demand path, the same costs and only information available on day t.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    lead_time: int = 7
    holding_rate_per_year: float = 0.25  # of unit price, charged on ending stock
    stockout_cost_multiple: float = 1.0  # × unit price per lost unit
    service_level: float = 0.95

    @property
    def label(self) -> str:
        return (
            f"L={self.lead_time} p={self.stockout_cost_multiple}x SL={self.service_level:.0%} "
            "(SCENARIO — not observed data)"
        )


@dataclass
class SimResult:
    demand: np.ndarray
    sales: np.ndarray
    lost: np.ndarray
    on_hand: np.ndarray  # end-of-day stock
    orders: np.ndarray
    holding_cost: np.ndarray  # per series
    stockout_cost: np.ndarray  # per series

    def summary(self, price: np.ndarray) -> dict[str, float]:
        demand_total = self.demand.sum()
        cycles = self.demand > 0
        return {
            "total_cost": float(self.holding_cost.sum() + self.stockout_cost.sum()),
            "holding_cost": float(self.holding_cost.sum()),
            "stockout_cost": float(self.stockout_cost.sum()),
            "fill_rate": float(self.sales.sum() / demand_total) if demand_total else 1.0,
            "cycle_service_level": float(
                1 - ((self.lost > 0) & cycles).sum() / max(cycles.sum(), 1)
            ),
            "stockout_days": float((self.lost > 0).sum()),
            "lost_units": float(self.lost.sum()),
            "avg_on_hand_units": float(self.on_hand.mean()),
            "avg_on_hand_value": float((self.on_hand * price[:, None]).mean()),
            "turnover": float(self.sales.sum() / max(self.on_hand.mean(axis=1).sum(), 1e-9)),
        }


def simulate(
    demand: np.ndarray,
    order_up_to: np.ndarray,
    price: np.ndarray,
    scenario: Scenario,
    initial_on_hand: np.ndarray | None = None,
) -> SimResult:
    """Replay `demand` (n, T) with order-up-to levels `order_up_to` (n, T) set on each day."""
    n, t_days = demand.shape
    lt = scenario.lead_time
    on_hand = (
        (order_up_to[:, 0] if initial_on_hand is None else initial_on_hand).astype(float).copy()
    )
    pipeline = np.zeros((n, lt))  # pipeline[:, k] arrives k+1 days from now
    out = {k: np.zeros((n, t_days)) for k in ("sales", "lost", "on_hand", "orders")}
    for t in range(t_days):
        on_hand += pipeline[:, 0]
        pipeline = np.roll(pipeline, -1, axis=1)
        pipeline[:, -1] = 0
        sold = np.minimum(on_hand, demand[:, t])
        out["sales"][:, t] = sold
        out["lost"][:, t] = demand[:, t] - sold
        on_hand -= sold
        out["on_hand"][:, t] = on_hand
        position = on_hand + pipeline.sum(axis=1)
        order = np.maximum(0.0, np.ceil(order_up_to[:, t] - position))
        pipeline[:, -1] += order
        out["orders"][:, t] = order
    daily_holding = scenario.holding_rate_per_year / 365 * price
    return SimResult(
        demand=demand.astype(float),
        sales=out["sales"],
        lost=out["lost"],
        on_hand=out["on_hand"],
        orders=out["orders"],
        holding_cost=(out["on_hand"] * daily_holding[:, None]).sum(axis=1),
        stockout_cost=(out["lost"] * (scenario.stockout_cost_multiple * price)[:, None]).sum(
            axis=1
        ),
    )
