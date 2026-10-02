# Demand Intelligence

A platform for SKU-level demand forecasting under uncertainty, and for turning those
forecasts into inventory decisions. It also tests whether external signals, including
signals extracted from text, actually improve forecasts. It is built on the Walmart M5
dataset, historical weather, and a public event/recall text corpus.

**Start with [`PROJECT_SPEC.md`](PROJECT_SPEC.md).** It fixes every definition, metric,
split, ablation, and promotion gate before any modeling begins.

Success is measured at two levels, and both are reported:

- **ML:** better forecasts (WRMSSE, WAPE, bias, calibration)
- **Business:** better simulated inventory outcomes (cost, service level)

## Layout

```
src/demand_intel/
  pipelines/       ingestion, validation, features, training, inference
  models/          statistical, lgbm, deep_learning, nlp
  forecasting/     evaluation (metrics), backtesting (splits), reconciliation
  inventory/       policies, simulation, optimization
  mlops/           registry, monitoring
apps/api           FastAPI serving
apps/web           Next.js app
orchestration/     Airflow DAGs (own image)
infrastructure/    Docker images, Postgres init, AWS IaC
monitoring/        Prometheus / Grafana / Evidently config
configs/           Versioned configs (weather locations, model configs)
data/raw/          Immutable source files (git-ignored)
data/samples/      Small committed samples for tests
evaluation/        Output of `make evaluate`, the only source of reported numbers
```

## Quickstart

```bash
cp .env.example .env
make install      # uv sync + pre-commit hooks
make check        # ruff + mypy + pytest
make up           # Postgres + MLflow (http://localhost:5000)
uv run demand-intel splits   # writes evaluation/split_manifest.json
```

Development runs in **GitHub Codespaces**: the devcontainer runs `make install`
on creation. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Codespaces secrets so M5 can be
downloaded.

The Makefile puts the virtualenv at `~/.venvs/demand-intel` (via `UV_PROJECT_ENVIRONMENT`)
so that the same commands also work from a local checkout on an exFAT drive, which has no
symlink support. If you run `uv` directly instead of through `make`, export that variable
first.

Each phase installs its heavier dependencies as an extra, for example
`uv sync --extra ml` or `uv sync --extra deep --extra nlp`.
