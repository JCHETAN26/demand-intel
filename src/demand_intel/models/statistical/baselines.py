"""Naive baselines F0a–F0c (PROJECT_SPEC.md §5.3). Every later model must beat these."""

from dataclasses import dataclass

import numpy as np

from demand_intel.forecasting.backtesting.harness import FoldContext, Forecast
from demand_intel.forecasting.backtesting.splits import HORIZON


@dataclass
class Naive:
    """F0a: ŷ_{t+h} = y_t."""

    name: str = "naive"

    def forecast(self, ctx: FoldContext) -> Forecast:
        last = ctx.history[:, -1].astype(np.float32)
        return Forecast(point=np.repeat(last[:, None], HORIZON, axis=1))


@dataclass
class SeasonalNaive:
    """F0b: ŷ_{t+h} = y_{t+h-7k}, the last observed week repeated."""

    name: str = "seasonal_naive"
    season: int = 7

    def forecast(self, ctx: FoldContext) -> Forecast:
        week = ctx.history[:, -self.season :].astype(np.float32)
        reps = -(-HORIZON // self.season)
        return Forecast(point=np.tile(week, (1, reps))[:, :HORIZON])


@dataclass
class MovingAverage:
    """F0c: mean of the last `window` days, flat over the horizon."""

    name: str = "moving_average_28"
    window: int = 28

    def forecast(self, ctx: FoldContext) -> Forecast:
        mean = ctx.history[:, -self.window :].mean(axis=1).astype(np.float32)
        return Forecast(point=np.repeat(mean[:, None], HORIZON, axis=1))
