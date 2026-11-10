"""End-to-end API contract: build a real (small) bundle, start the app, call the endpoints.

Unlike tests/unit/test_api.py, this goes through the same path `forecast-batch` takes:
the model is chosen against a registry whose champion is not servable, so the bundle is a
recorded fallback. The tests pin the response schemas of /v1/forecast and
/v1/inventory/recommend so a field rename or removal fails here before it breaks the web app.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("lightgbm")
from fastapi.testclient import TestClient

from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.models.lgbm.model import LightGBMForecaster
from demand_intel.pipelines.inference.serving import build_bundle, choose_serving_model
from tests.conftest import synthetic_m5

pytestmark = pytest.mark.integration

PARAMS = {
    "objective": "tweedie",
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "num_threads": 2,
    "seed": 0,
}
SERIES = {"item_id": "FOODS_1_000", "store_id": "CA_1"}
QKEYS = {f"p{round(t * 100):02d}" for t in QUANTILES}

FORECAST_FIELDS = {
    "item_id",
    "store_id",
    "origin_date",
    "dates",
    "mean",
    "p10",
    "p50",
    "p90",
    "quantiles",
    "model",
    "registry_champion",
    "fallback",
}
INVENTORY_FIELDS = {
    "item_id",
    "store_id",
    "lead_time_days",
    "service_level",
    "expected_demand",
    "p50",
    "p90",
    "target_level",
    "safety_stock",
    "inventory_position",
    "recommended_order",
    "stockout_probability",
    "rho",
    "note",
}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("e2e")
    m5 = synthetic_m5(n_items=3)
    profile = pd.DataFrame(
        {"demand_class": "smooth", "volume_tercile": "mid"}, index=range(len(m5.series))
    )
    choice = choose_serving_model("seasonal_naive", servable=lambda n: n.startswith("lgbm"))
    model = LightGBMForecaster(
        choice.served, train_days=120, params=PARAMS, max_rounds=40, model_dir=tmp / "models"
    )
    build_bundle(
        m5, tmp / "serving", tmp / "models", tmp / "eval", profile, forecaster=model, choice=choice
    )
    return tmp / "serving"


@pytest.fixture(scope="module")
def client(bundle: Path) -> Iterator[TestClient]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DI_SERVING_DIR", str(bundle))
        from demand_intel.serving.api import app

        with TestClient(app) as c:
            yield c


def test_bundle_records_the_fallback(client: TestClient) -> None:
    health = client.get("/v1/health").json()
    assert health["status"] == "ok"
    assert health["forecast_model"] == "lgbm"
    assert health["registry_champion"] == "seasonal_naive"
    assert health["fallback"] is True
    summary = client.get("/v1/summary").json()
    assert summary["fallback"] is True
    assert "seasonal_naive" in summary["fallback_reason"]


def test_forecast_schema_and_values(client: TestClient) -> None:
    r = client.post("/v1/forecast", json={**SERIES, "horizon": 28})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == FORECAST_FIELDS
    assert set(body["quantiles"]) == QKEYS
    assert body["model"] == "lgbm" and body["fallback"] is True
    n = len(body["dates"])
    assert n == 28
    assert all(len(body[k]) == n for k in ("mean", "p10", "p50", "p90"))
    assert all(len(v) == n for v in body["quantiles"].values())
    q = np.array([body["quantiles"][f"p{round(t * 100):02d}"] for t in QUANTILES])
    assert (q >= 0).all() and (np.diff(q, axis=0) >= -1e-9).all()  # non-negative, monotone
    assert body["p50"] == body["quantiles"]["p50"]
    assert pd.to_datetime(body["dates"]).is_monotonic_increasing
    assert pd.Timestamp(body["dates"][0]) > pd.Timestamp(body["origin_date"])


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({**SERIES, "horizon": 1}, 200),
        ({**SERIES, "horizon": 0}, 422),
        ({**SERIES, "horizon": 29}, 422),
        ({"item_id": "FOODS_1_000"}, 422),
        ({"item_id": "NOPE", "store_id": "CA_1"}, 404),
    ],
)
def test_forecast_validation(client: TestClient, payload: dict[str, object], status: int) -> None:
    assert client.post("/v1/forecast", json=payload).status_code == status


def test_inventory_schema_and_consistency(client: TestClient) -> None:
    r = client.post(
        "/v1/inventory/recommend",
        json={**SERIES, "current_inventory": 0, "lead_time_days": 7, "service_level": 0.95},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == INVENTORY_FIELDS
    assert body["lead_time_days"] == 7 and body["service_level"] == 0.95
    assert isinstance(body["recommended_order"], int)
    assert 0 <= body["stockout_probability"] <= 1
    assert body["p50"] <= body["p90"] <= body["target_level"] + 1e-6
    assert body["recommended_order"] >= body["target_level"] - body["inventory_position"] - 1

    # Lead-time demand comes from the same forecast the forecast endpoint serves: its
    # expected value is the sum of daily means over lead time plus the one-day review.
    fc = client.post("/v1/forecast", json={**SERIES, "horizon": 8}).json()
    assert body["expected_demand"] == pytest.approx(sum(fc["mean"]), rel=1e-2, abs=1e-2)


def test_inventory_responds_to_stock_and_service_level(client: TestClient) -> None:
    def rec(**kw: float) -> dict[str, float]:
        base = {**SERIES, "current_inventory": 0, "lead_time_days": 7, "service_level": 0.95}
        r = client.post("/v1/inventory/recommend", json={**base, **kw})
        assert r.status_code == 200
        return r.json()  # type: ignore[no-any-return]

    empty, full = rec(current_inventory=0), rec(current_inventory=10_000)
    assert empty["recommended_order"] > 0 and full["recommended_order"] == 0
    assert full["stockout_probability"] <= empty["stockout_probability"]
    assert rec(service_level=0.99)["target_level"] >= rec(service_level=0.8)["target_level"]
    assert rec(on_order=10_000)["recommended_order"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"current_inventory": -1},
        {"lead_time_days": 0},
        {"lead_time_days": 22},
        {"service_level": 0.5},
        {"service_level": 1.0},
    ],
)
def test_inventory_validation(client: TestClient, bad: dict[str, float]) -> None:
    payload = {**SERIES, "current_inventory": 5, **bad}
    assert client.post("/v1/inventory/recommend", json=payload).status_code == 422
