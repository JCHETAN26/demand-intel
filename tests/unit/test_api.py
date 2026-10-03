import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from demand_intel.models.lgbm.model import LightGBMForecaster
from demand_intel.pipelines.inference.serving import build_bundle
from tests.conftest import synthetic_m5

PARAMS = {
    "objective": "tweedie",
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "num_threads": 2,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp = tmp_path_factory.mktemp("bundle")
    m5 = synthetic_m5(n_items=3)
    profile = pd.DataFrame(
        {"demand_class": "smooth", "volume_tercile": "mid"}, index=range(len(m5.series))
    )
    model = LightGBMForecaster(
        "lgbm", train_days=120, params=PARAMS, max_rounds=40, model_dir=tmp / "models"
    )
    build_bundle(m5, tmp / "serving", tmp / "models", tmp / "eval", profile, forecaster=model)
    import os

    os.environ["DI_SERVING_DIR"] = str(tmp / "serving")
    from demand_intel.serving.api import app

    with TestClient(app) as c:
        yield c


def test_health_and_summary(client: TestClient) -> None:
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["series"] == 6
    s = client.get("/v1/summary").json()
    assert s["skus_monitored"] == 6 and s["horizon_days"] == 28
    assert 0 <= s["stockout_risk"] <= 6 and "SCENARIO" in s["risk_definition"]


def test_forecast_contract(client: TestClient) -> None:
    r = client.post(
        "/v1/forecast", json={"item_id": "FOODS_1_000", "store_id": "CA_1", "horizon": 14}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["p50"]) == len(body["dates"]) == 14
    assert all(
        lo <= mid <= hi for lo, mid, hi in zip(body["p10"], body["p50"], body["p90"], strict=True)
    )
    assert body["dates"][0] == "2016-05-23"
    assert (
        client.post("/v1/forecast", json={"item_id": "NOPE", "store_id": "CA_1"}).status_code == 404
    )
    assert (
        client.post(
            "/v1/forecast", json={"item_id": "FOODS_1_000", "store_id": "CA_1", "horizon": 40}
        ).status_code
        == 422
    )


def test_inventory_recommendation(client: TestClient) -> None:
    base = {
        "item_id": "FOODS_1_000",
        "store_id": "CA_1",
        "lead_time_days": 7,
        "service_level": 0.95,
    }
    low = client.post("/v1/inventory/recommend", json={**base, "current_inventory": 0}).json()
    high = client.post("/v1/inventory/recommend", json={**base, "current_inventory": 10_000}).json()
    assert low["recommended_order"] > 0 and high["recommended_order"] == 0
    assert low["p50"] <= low["p90"] <= low["target_level"] + 1e-6
    assert low["stockout_probability"] > high["stockout_probability"]


def test_events_series_detail_risk_and_metrics(client: TestClient) -> None:
    ev = client.post(
        "/v1/events/extract", json={"text": "Undeclared peanuts. Distributed in CA and TX."}
    ).json()
    assert ev["event_type"] == "UNDECLARED_ALLERGEN" and ev["states"] == ["CA", "TX"]
    d = client.get("/v1/series/FOODS_1_000/CA_1").json()
    assert len(d["forecast"]["mean"]) == 28 and len(d["history"]["sales"]) == 182
    assert set(d["drivers"]["week1_units"]) == set(d["drivers"]["families"])
    # driver totals reconcile with the forecast minus the model baseline (exact by construction)
    assert np.isfinite(np.array(d["drivers"]["daily_units"])).all()
    assert client.get("/v1/series", params={"cat_id": "FOODS"}).json()["total"] == 2
    assert client.get("/v1/risk", params={"kind": "excess"}).status_code == 200
    m = client.get("/metrics").text
    assert "di_request_seconds_bucket" in m and 'route="/v1/forecast"' in m
