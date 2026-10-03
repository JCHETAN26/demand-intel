"""Name → forecaster factory. Imports are lazy so heavy dependencies load only when used."""

import contextlib
from collections.abc import Callable

from demand_intel.forecasting.backtesting.harness import Forecaster


def _baselines() -> dict[str, Callable[[], Forecaster]]:
    from demand_intel.models.statistical.baselines import MovingAverage, Naive, SeasonalNaive

    return {
        "naive": Naive,
        "seasonal_naive": SeasonalNaive,
        "moving_average_28": MovingAverage,
    }


def _statistical() -> dict[str, Callable[[], Forecaster]]:
    from demand_intel.models.statistical.classical import statistical_models

    return statistical_models()


# Feature-family ablation ladder (spec §6.1): each step adds one family.
ABLATION_LADDER: dict[str, tuple[str, ...]] = {
    "lgbm_abl1_sales": ("sales",),
    "lgbm_abl2_calendar": ("sales", "calendar"),
    "lgbm_abl3_snap": ("sales", "calendar", "snap"),
    "lgbm_abl4_price": ("sales", "calendar", "snap", "price"),
    "lgbm_abl5_hierarchy": ("sales", "calendar", "snap", "price", "hierarchy"),
    "lgbm_abl6_weather": ("sales", "calendar", "snap", "price", "hierarchy", "weather"),
}


def _lgbm() -> dict[str, Callable[[], Forecaster]]:
    from demand_intel.config import settings
    from demand_intel.models.lgbm.model import LightGBMForecaster
    from demand_intel.pipelines.features.build import DEFAULT_FAMILIES

    model_dir = settings.data_dir / "models" / "lgbm"
    out: dict[str, Callable[[], Forecaster]] = {
        "lgbm": lambda: LightGBMForecaster("lgbm", DEFAULT_FAMILIES, model_dir=model_dir),
        "lgbm_weather_oracle": lambda: LightGBMForecaster(
            "lgbm_weather_oracle", (*DEFAULT_FAMILIES, "weather_oracle")
        ),
    }
    full = ABLATION_LADDER["lgbm_abl6_weather"]
    out["lgbm_mm_E_text_embed"] = lambda: LightGBMForecaster(
        "lgbm_mm_E_text_embed", (*full, "text_embed")
    )
    out["lgbm_mm_F_text_events"] = lambda: LightGBMForecaster(
        "lgbm_mm_F_text_events", (*full, "text_events")
    )
    for name, fams in ABLATION_LADDER.items():
        out[name] = lambda name=name, fams=fams: LightGBMForecaster(name, fams)  # type: ignore[misc]
    return out


def _deep() -> dict[str, Callable[[], Forecaster]]:
    from demand_intel.config import settings
    from demand_intel.models.deep_learning.lstm import Seq2SeqLSTM
    from demand_intel.models.deep_learning.tft import TemporalFusionTransformer
    from demand_intel.models.deep_learning.train import DeepForecaster

    ckpt = settings.data_dir / "models" / "deep"
    return {
        "lstm": lambda: DeepForecaster("lstm", Seq2SeqLSTM, checkpoint_dir=ckpt),
        "tft": lambda: DeepForecaster("tft", TemporalFusionTransformer, checkpoint_dir=ckpt),
    }


def available() -> dict[str, Callable[[], Forecaster]]:
    factories = {**_baselines(), **_statistical(), **_lgbm()}
    with contextlib.suppress(ImportError):  # torch is an optional extra
        factories.update(_deep())
    return factories


def make(name: str) -> Forecaster:
    factories = available()
    if name not in factories:
        raise KeyError(f"unknown model {name!r}; available: {sorted(factories)}")
    return factories[name]()
