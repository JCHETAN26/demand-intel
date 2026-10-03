"""Serving benchmark (PROJECT_SPEC.md §11.3, plan Phase 32) → evaluation/serving_benchmark.json.

Starts the API on the real serving bundle, warms it, then drives each endpoint with a fixed
number of concurrent clients. Latency is measured client-side (includes HTTP and JSON).
Two configurations are compared: 1 worker (baseline) and N workers (the optimization).

Usage: uv run python scripts/benchmark_api.py [--requests 2000] [--concurrency 16]
"""

import argparse
import asyncio
import json
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


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


async def drive(url: str, payloads: list[dict[str, object]], concurrency: int) -> dict[str, float]:
    lat: list[float] = []
    errors = 0
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    for p in payloads:
        queue.put_nowait(p)

    async with httpx.AsyncClient(timeout=30) as client:

        async def worker() -> None:
            nonlocal errors
            while not queue.empty():
                body = queue.get_nowait()
                t0 = time.perf_counter()
                r = await client.post(url, json=body)
                lat.append(time.perf_counter() - t0)
                if r.status_code >= 500:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    ms = np.array(lat) * 1000
    return {
        "requests": len(lat),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "rps": len(lat) / wall,
        "error_rate": errors / max(len(lat), 1),
    }


def start(workers: int, port: int) -> subprocess.Popen[bytes]:
    env = {**os.environ, "DI_SERVING_DIR": str(BUNDLE)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demand_intel.serving.api:app",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    for _ in range(120):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=1).json()["status"] == "ok":
                return proc
        except (httpx.HTTPError, KeyError, ValueError):
            time.sleep(1)
    proc.kill()
    raise RuntimeError("API did not become healthy")


def rss_mb(proc: subprocess.Popen[bytes]) -> float:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "--ppid", str(proc.pid)], capture_output=True, text=True
        )
        own = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True
        )
        return sum(int(x) for x in (out.stdout + own.stdout).split()) / 1024
    except (OSError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    series = pd.read_parquet(BUNDLE / "series.parquet")[["item_id", "store_id"]].to_dict("records")
    rng = random.Random(0)
    picks = [rng.choice(series) for _ in range(args.requests)]
    endpoints = {
        "/v1/forecast": [{**p, "horizon": 28} for p in picks],
        "/v1/inventory/recommend": [
            {**p, "current_inventory": 20, "lead_time_days": 7, "service_level": 0.95}
            for p in picks
        ],
        "/v1/events/extract": [
            {"text": "Undeclared milk and soy in granola bars. Distributed in CA and TX."}
        ]
        * args.requests,
    }
    results: dict[str, object] = {
        "hardware": {
            "cpu": cpu_model(),
            "cores": os.cpu_count(),
            "python": platform.python_version(),
        },
        "settings": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "client": "httpx AsyncClient, same machine (includes HTTP + JSON)",
        },
        "slo": {
            "/v1/forecast": {"p50_ms": 30, "p95_ms": 100, "p99_ms": 250},
            "/v1/inventory/recommend": {"p95_ms": 150},
            "/v1/events/extract": {"p95_ms": 300},
        },
        "configs": {},
    }
    for label, workers in (
        ("baseline_1_worker", 1),
        (f"optimized_{args.workers}_workers", args.workers),
    ):
        port = 8765 + workers
        proc = start(workers, port)
        try:
            for path, payloads in endpoints.items():  # warm-up
                asyncio.run(drive(f"http://127.0.0.1:{port}{path}", payloads[:50], 4))
            cfg: dict[str, object] = {"workers": workers}
            for path, payloads in endpoints.items():
                cfg[path] = asyncio.run(
                    drive(f"http://127.0.0.1:{port}{path}", payloads, args.concurrency)
                )
            cfg["rss_mb"] = rss_mb(proc)
            results["configs"][label] = cfg  # type: ignore[index]
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    out = ROOT / "evaluation" / "serving_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["configs"], indent=2))


if __name__ == "__main__":
    main()
