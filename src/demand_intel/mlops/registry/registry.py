"""Champion/challenger registry (plan Phase 29).

Two records, kept consistent:
- `evaluation/registry.json`: the current champion per task and every promotion decision
  with its gate report. It is the source of truth that `make evaluate` and the API read.
- MLflow: each evaluated model is a run (params, fold-averaged and test metrics, results
  file); a promoted forecasting model is registered with its trained boosters and gets the
  `champion` alias, the previous champion keeps a version without it.

MLflow is reached at DI_MLFLOW_TRACKING_URI; when that server is down, runs go to a local
SQLite store so the pipeline still completes (and says so).
"""

import contextlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from demand_intel.mlops.registry.gates import GateReport

INITIAL_CHAMPIONS = {"forecast": "seasonal_naive", "nlp_event": "rules-v1"}


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        reg: dict[str, Any] = json.loads(path.read_text())
        return reg
    return {
        "tasks": {
            t: {"champion": m, "since": None, "challengers": []}
            for t, m in INITIAL_CHAMPIONS.items()
        },
        "decisions": [],
    }


def record_decision(
    path: Path, task: str, report: GateReport, mlflow_version: str | None = None
) -> dict[str, Any]:
    reg = load_registry(path)
    now = datetime.now(UTC).isoformat()
    entry = {"task": task, "at": now, **report.to_dict(), "mlflow_version": mlflow_version}
    reg["decisions"].append(entry)
    t = reg["tasks"].setdefault(
        task, {"champion": report.champion, "since": None, "challengers": []}
    )
    if report.promote:
        t["challengers"] = [c for c in t["challengers"] if c != report.challenger]
        t["previous_champion"] = t["champion"]
        t["champion"], t["since"] = report.challenger, now
    elif report.challenger not in t["challengers"]:
        t["challengers"].append(report.challenger)
    path.write_text(json.dumps(reg, indent=2))
    return reg


def tracking_uri(preferred: str, fallback_dir: Path) -> str:
    """Use the MLflow server if it answers, else a local SQLite store."""
    if preferred.startswith("http"):
        try:
            with urllib.request.urlopen(preferred.rstrip("/") + "/health", timeout=3):
                return preferred
        except OSError:
            pass
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback_dir / 'mlflow.db'}"


def log_model_runs(
    uri: str, summary: pd.DataFrame, backtest_dir: Path, experiment: str = "forecasting"
) -> int:
    """One MLflow run per (model, population) row of the comparison summary."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    n = 0
    for row in summary.to_dict("records"):
        with mlflow.start_run(run_name=f"{row['model']}__{row['population']}"):
            mlflow.set_tags({"model": row["model"], "population": row["population"]})
            mlflow.log_metrics(
                {
                    str(k): float(v)
                    for k, v in row.items()
                    if isinstance(v, (int, float)) and pd.notna(v) and k != "folds"
                }
            )
            results = backtest_dir / f"{row['model']}__{row['population']}.csv"
            if results.exists():
                mlflow.log_artifact(str(results), artifact_path="backtest")
        n += 1
    return n


def register_forecaster(
    uri: str, name: str, model_dir: Path, metrics: dict[str, float], alias: str
) -> str:
    """Register a trained LightGBM bundle (boosters + quantile maps + metadata)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("registry")
    with mlflow.start_run(run_name=f"register-{name}") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        source = f"runs:/{run.info.run_id}/model"
    client = MlflowClient()
    with contextlib.suppress(mlflow.exceptions.MlflowException):  # already exists
        client.create_registered_model(name)
    version = client.create_model_version(name=name, source=source, run_id=run.info.run_id)
    client.set_registered_model_alias(name, alias, version.version)
    return str(version.version)
