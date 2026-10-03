"""Forecast, inventory and event-extraction API (plan Phase 30).

Run: `uvicorn demand_intel.serving.api:app` with DI_SERVING_DIR pointing at the bundle.
Every route is instrumented for Prometheus at /metrics (latency histogram, request and
error counters); the bundle is loaded once at startup and served from memory.
"""

import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from demand_intel.config import settings
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.serving.events import EventModel, RuleModel, extract
from demand_intel.serving.inventory import calibrated_rho, recommend
from demand_intel.serving.store import ServingStore

LATENCY = Histogram(
    "di_request_seconds",
    "Request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
REQUESTS = Counter("di_requests_total", "Requests", ["route", "status"])
FORECAST_P50 = Histogram(
    "di_forecast_p50_units",
    "Day-1 P50 of served forecasts",
    buckets=(0, 0.5, 1, 2, 5, 10, 25, 50, 100),
)
EVENT_TYPES = Counter("di_events_total", "Extracted events by type", ["event_type", "model"])


class State:
    store: ServingStore | None = None
    events: EventModel = RuleModel()


def serving_dir() -> Path:
    return Path(os.environ.get("DI_SERVING_DIR", settings.data_dir / "serving"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    path = serving_dir()
    if (path / "summary.json").exists():
        State.store = ServingStore.load(path)
    from demand_intel.serving.nlp_model import load_champion

    champion = load_champion()
    if champion is not None:
        State.events = champion
    yield


app = FastAPI(title="Demand Intelligence API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    route = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(route, "500").inc()
        raise
    if route != "/metrics":
        LATENCY.labels(route).observe(time.perf_counter() - start)
        REQUESTS.labels(route, str(response.status_code)).inc()
    return response


def _store() -> ServingStore:
    if State.store is None:
        raise HTTPException(503, "No forecast bundle loaded. Run `demand-intel forecast-batch`.")
    return State.store


def _row(store: ServingStore, item_id: str, store_id: str) -> int:
    i = store.row(item_id, store_id)
    if i is None:
        raise HTTPException(404, f"Unknown series {item_id} at store {store_id}.")
    return i


class ForecastRequest(BaseModel):
    item_id: str = Field(examples=["FOODS_3_090"])
    store_id: str = Field(examples=["CA_1"])
    horizon: int = Field(28, ge=1, le=28)


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    origin_date: str
    dates: list[str]
    mean: list[float]
    p10: list[float]
    p50: list[float]
    p90: list[float]
    quantiles: dict[str, list[float]]
    model: str


@app.get("/v1/health")
def health() -> dict[str, Any]:
    s = State.store
    return {
        "status": "ok" if s else "no_bundle",
        "series": len(s.series) if s else 0,
        "forecast_model": s.summary.get("model") if s else None,
        "event_model": State.events.name,
    }


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    s = _store()
    i = _row(s, req.item_id, req.store_id)
    h = req.horizon
    q = s.quantiles[i, :h]
    FORECAST_P50.observe(float(q[0, QUANTILES.index(0.5)]))
    qmap = {f"p{round(t * 100):02d}": q[:, j].round(3).tolist() for j, t in enumerate(QUANTILES)}
    return ForecastResponse(
        item_id=req.item_id,
        store_id=req.store_id,
        origin_date=s.summary["origin_date"],
        dates=[s.date(d) for d in s.forecast_days[:h]],
        mean=s.mean[i, :h].round(3).tolist(),
        p10=qmap["p10"],
        p50=qmap["p50"],
        p90=qmap["p90"],
        quantiles=qmap,
        model=s.summary["model"],
    )


class InventoryRequest(BaseModel):
    item_id: str
    store_id: str
    current_inventory: float = Field(ge=0)
    on_order: float = Field(0, ge=0)
    lead_time_days: int = Field(7, ge=1, le=21)
    service_level: float = Field(0.95, gt=0.5, lt=1)


@app.post("/v1/inventory/recommend")
def inventory_recommend(req: InventoryRequest) -> dict[str, Any]:
    s = _store()
    i = _row(s, req.item_id, req.store_id)
    rho = calibrated_rho(s.summary, req.lead_time_days, req.service_level)
    rec = recommend(
        s.quantiles[i],
        s.mean[i],
        req.current_inventory,
        req.on_order,
        req.lead_time_days,
        req.service_level,
        rho,
    )
    return {
        "item_id": req.item_id,
        "store_id": req.store_id,
        **rec.__dict__,
        "note": "Lead-time demand covers the lead time plus a one-day review period.",
    }


class EventRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)
    distribution: str | None = None


@app.post("/v1/events/extract")
def events_extract(req: EventRequest) -> dict[str, Any]:
    ev = extract(State.events, [req.text], [req.distribution or req.text])[0]
    EVENT_TYPES.labels(ev.event_type, ev.model).inc()
    return ev.__dict__


@app.get("/v1/series")
def list_series(
    cat_id: str | None = None, store_id: str | None = None, q: str | None = None, limit: int = 200
) -> dict[str, Any]:
    s = _store()
    df = s.series
    if cat_id:
        df = df[df["cat_id"] == cat_id]
    if store_id:
        df = df[df["store_id"] == store_id]
    if q:
        df = df[df["item_id"].str.contains(q.upper(), regex=False)]
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "demand_class"]
    return {"total": len(df), "items": df[cols].head(min(limit, 1000)).to_dict("records")}


@app.get("/v1/series/{item_id}/{store_id}")
def series_detail(item_id: str, store_id: str) -> dict[str, Any]:
    s = _store()
    i = _row(s, item_id, store_id)
    meta = s.series.iloc[i].to_dict()
    hist_price = s.history_price[i]
    fut_price = s.future_price[i]
    return {
        "series": meta,
        "history": {
            "dates": [s.date(d) for d in s.history_days],
            "sales": s.history_sales[i].tolist(),
            "price": [None if np.isnan(p) else round(float(p), 2) for p in hist_price],
        },
        "forecast": {
            "dates": [s.date(d) for d in s.forecast_days],
            "mean": s.mean[i].round(3).tolist(),
            **{
                f"p{round(t * 100):02d}": s.quantiles[i, :, j].round(3).tolist()
                for j, t in enumerate(QUANTILES)
            },
            "price": [None if np.isnan(p) else round(float(p), 2) for p in fut_price],
        },
        "drivers": {
            "families": s.driver_names,
            "daily_units": s.drivers[i].round(3).tolist(),
            "week1_units": dict(
                zip(s.driver_names, s.drivers[i, :7].sum(axis=0).round(2).tolist(), strict=True)
            ),
            "method": "TreeSHAP in log space, allocated to units in proportion (exact in total).",
        },
    }


@app.get("/v1/summary")
def summary() -> dict[str, Any]:
    """Homepage KPIs. Store inventory is not in M5: risk counts use the SCENARIO stock level
    defined in the bundle summary."""
    s = _store()
    return {
        "origin_date": s.summary["origin_date"],
        "forecast_dates": s.summary["forecast_dates"],
        "horizon_days": int(s.mean.shape[1]),
        "skus_monitored": len(s.series),
        "stockout_risk": int(s.risk["stockout_risk"].sum()),
        "excess_inventory_risk": int(s.risk["excess_risk"].sum()),
        "risk_definition": s.summary.get("risk", {}).get("definition"),
        "evaluation": s.summary.get("evaluation", {}),
        "model": s.summary.get("model"),
    }


@app.get("/v1/risk")
def risk(kind: Literal["stockout", "excess"] = "stockout", limit: int = 50) -> dict[str, Any]:
    """Series flagged at risk under the SCENARIO stock, largest gap first."""
    s = _store()
    r = s.risk.join(s.series[["item_id", "store_id", "cat_id"]], on="series")
    if kind == "stockout":
        r = r[r["stockout_risk"]].assign(gap=lambda x: x["lead_time_p90"] - x["scenario_stock"])
    else:
        r = r[r["excess_risk"]].assign(gap=lambda x: x["scenario_stock"] - x["demand28_p95"])
    r = r.sort_values("gap", ascending=False).head(min(limit, 500))
    cols = [
        "item_id",
        "store_id",
        "cat_id",
        "scenario_stock",
        "lead_time_p90",
        "demand28_p95",
        "gap",
    ]
    return {"kind": kind, "items": r[cols].round(2).to_dict("records")}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
