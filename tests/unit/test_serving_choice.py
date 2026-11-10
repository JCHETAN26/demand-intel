import pytest

from demand_intel.pipelines.inference.serving import (
    FALLBACK_MODEL,
    ServingSelectionError,
    choose_serving_model,
)


def lgbm_only(name: str) -> bool:
    return name.startswith("lgbm")


def test_servable_champion_is_served_without_fallback() -> None:
    c = choose_serving_model("lgbm_bc", lgbm_only)
    assert (c.served, c.champion, c.fallback, c.reason) == ("lgbm_bc", "lgbm_bc", False, None)


def test_unservable_champion_falls_back_with_a_reason() -> None:
    c = choose_serving_model("seasonal_naive", lgbm_only)
    assert c.served == FALLBACK_MODEL and c.fallback
    assert c.reason is not None and "seasonal_naive" in c.reason
    assert c.provenance() == {
        "registry_champion": "seasonal_naive",
        "fallback": True,
        "fallback_reason": c.reason,
    }


def test_strict_refuses_to_fall_back() -> None:
    with pytest.raises(ServingSelectionError, match="seasonal_naive"):
        choose_serving_model("seasonal_naive", lgbm_only, allow_fallback=False)


def test_explicit_override_is_recorded_as_fallback() -> None:
    c = choose_serving_model("seasonal_naive", lgbm_only, requested="lgbm_bc")
    assert c.served == "lgbm_bc" and c.fallback
    assert c.reason is not None and "--model lgbm_bc" in c.reason
    same = choose_serving_model("lgbm_bc", lgbm_only, requested="lgbm_bc")
    assert not same.fallback


def test_unservable_override_is_rejected() -> None:
    with pytest.raises(ServingSelectionError):
        choose_serving_model("lgbm", lgbm_only, requested="tft")
