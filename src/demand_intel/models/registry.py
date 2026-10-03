"""Name → forecaster factory. Imports are lazy so heavy dependencies load only when used."""

from collections.abc import Callable

from demand_intel.forecasting.backtesting.harness import Forecaster


def _baselines() -> dict[str, Callable[[], Forecaster]]:
    from demand_intel.models.statistical.baselines import MovingAverage, Naive, SeasonalNaive

    return {
        "naive": Naive,
        "seasonal_naive": SeasonalNaive,
        "moving_average_28": MovingAverage,
    }


def available() -> dict[str, Callable[[], Forecaster]]:
    return {**_baselines()}


def make(name: str) -> Forecaster:
    factories = available()
    if name not in factories:
        raise KeyError(f"unknown model {name!r}; available: {sorted(factories)}")
    return factories[name]()
