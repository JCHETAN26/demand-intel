"""Serving benchmark (PROJECT_SPEC.md §11.3, plan Phase 32) → evaluation/serving_benchmark.json.

Starts the API on the real serving bundle (2 workers, as deployed), warms it, then per
endpoint:
- latency phase: 4 concurrent clients, client-side p50/p95/p99 (includes HTTP + JSON)
- throughput phase: 4 client processes × 8 concurrent requests, total requests per second
- server-side p50/p95 from the API's own Prometheus histogram (excludes the client)

`--label` names the run (e.g. before / after an optimization); results from earlier
labels in the output file are kept so before/after can be compared.

Usage: uv run python scripts/benchmark_api.py --label after
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "serving"
OUT = ROOT / "evaluation" / "serving_benchmark.json"
PORT = 8790


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


async def _drive(
    url: str, payloads: list[dict[str, object]], concurrency: int
) -> tuple[list[float], int]:
    lat: list[float] = []
    errors = 0
    it = iter(payloads)
    async with httpx.AsyncClient(timeout=30) as client:

        async def worker() -> None:
            nonlocal errors
            for body in it:
                t0 = time.perf_counter()
                r = await client.post(url, json=body)
                lat.append(time.perf_counter() - t0)
                errors += r.status_code >= 500

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return lat, errors


def _proc(args: tuple[str, list[dict[str, object]], int]) -> tuple[list[float], int]:
    return asyncio.run(_drive(*args))


def server_quantiles(route: str) -> dict[str, float]:
    text = httpx.get(f"http://127.0.0.1:{PORT}/metrics", timeout=10).text
    buckets = []
    for line in text.splitlines():
        if line.startswith("di_request_seconds_bucket") and f'route="{route}"' in line:
            le = line.split('le="')[1].split('"')[0]
            buckets.append(
                (float("inf") if le == "+Inf" else float(le), float(line.rsplit(" ", 1)[1]))
            )
    buckets.sort()
    total = buckets[-1][1] if buckets else 0.0

    def q(p: float) -> float:
        target, prev_le, prev_c = p * total, 0.0, 0.0
        for le, c in buckets:
            if c >= target:
                if le == float("inf"):
                    return prev_le * 1000
                frac = (target - prev_c) / max(c - prev_c, 1e-9)
                return (prev_le + frac * (le - prev_le)) * 1000
            prev_le, prev_c = le, c
        return float("nan")

    return {"server_p50_ms": q(0.5), "server_p95_ms": q(0.95), "server_count": total}


def start(workers: int) -> subprocess.Popen[bytes]:
    env = {**os.environ, "DI_SERVING_DIR": str(BUNDLE), "PROMETHEUS_MULTIPROC_DIR": ""}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demand_intel.serving.api:app",
            "--port",
            str(PORT),
            "--workers",
            str(workers),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    for _ in range(180):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/v1/health", timeout=1).json()["status"] == "ok":
                return proc
        except (httpx.HTTPError, KeyError, ValueError):
            time.sleep(1)
    proc.kill()
    raise RuntimeError("API did not become healthy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--latency-requests", type=int, default=1000)
    ap.add_argument("--throughput-requests", type=int, default=4000)
    args = ap.parse_args()
    series = pd.read_parquet(BUNDLE / "series.parquet")[["item_id", "store_id"]].to_dict("records")
    rng = random.Random(0)
    picks = [rng.choice(series) for _ in range(args.throughput_requests)]
    endpoints = {
        "/v1/forecast": [{**p, "horizon": 28} for p in picks],
        "/v1/inventory/recommend": [
            {**p, "current_inventory": 20, "lead_time_days": 7, "service_level": 0.95}
            for p in picks
        ],
        "/v1/events/extract": [
            {"text": "Undeclared milk and soy in granola bars. Distributed in CA and TX."}
        ]
        * args.throughput_requests,
    }
    run: dict[str, object] = {"workers": args.workers}
    proc = start(args.workers)
    try:
        for path, payloads in endpoints.items():
            url = f"http://127.0.0.1:{PORT}{path}"
            asyncio.run(_drive(url, payloads[:100], 4))  # warm-up
            lat, err = asyncio.run(_drive(url, payloads[: args.latency_requests], 4))
            ms = np.array(lat) * 1000
            chunks = [payloads[i::4] for i in range(4)]
            t0 = time.perf_counter()
            with mp.get_context("spawn").Pool(4) as pool:
                outs = pool.map(_proc, [(url, c, 8) for c in chunks])
            wall = time.perf_counter() - t0
            done = sum(len(o[0]) for o in outs)
            run[path] = {
                "p50_ms": float(np.percentile(ms, 50)),
                "p95_ms": float(np.percentile(ms, 95)),
                "p99_ms": float(np.percentile(ms, 99)),
                "throughput_rps": done / wall,
                "error_rate": (err + sum(o[1] for o in outs)) / (len(lat) + done),
                **server_quantiles(path),
            }
            print(path, json.dumps(run[path]))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    results: dict[str, object] = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "runs" not in results:
        results = {"runs": {}}
    results["hardware"] = {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "note": "client and server share the machine",
    }
    results["method"] = (
        "latency: 4 concurrent clients, client-side; throughput: 4 processes x 8 "
        "concurrent; server-side quantiles from the Prometheus histogram"
    )
    results["slo"] = {
        "/v1/forecast": {"p50_ms": 30, "p95_ms": 100, "p99_ms": 250},
        "/v1/inventory/recommend": {"p95_ms": 150},
        "/v1/events/extract": {"p95_ms": 300},
    }
    results["runs"][args.label] = run  # type: ignore[index]
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
