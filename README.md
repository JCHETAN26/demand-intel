<div align="center">

# Demand Intelligence

**Probabilistic demand forecasting for 30,490 retail SKU × store series, turned into
replenishment decisions and served behind an API with promotion gates and monitoring.**

[![CI](https://github.com/JCHETAN26/demand-intel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JCHETAN26/demand-intel/actions/workflows/ci.yml)
[![Deploy](https://img.shields.io/badge/deploy-ECS%20Fargate%20(manual)-FF9900?logo=amazonaws&logoColor=white)](.github/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/JCHETAN26/demand-intel)](https://github.com/JCHETAN26/demand-intel/commits/main)

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](.python-version)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-2A6DB2)](pyproject.toml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)

[![LightGBM](https://img.shields.io/badge/LightGBM-quantile-9ACD32)](src/demand_intel/models/lgbm)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%7C%20TFT-EE4C2C?logo=pytorch&logoColor=white)](src/demand_intel/models/deep_learning)
[![MLflow](https://img.shields.io/badge/MLflow-registry-0194E2?logo=mlflow&logoColor=white)](src/demand_intel/mlops/registry)
[![Airflow](https://img.shields.io/badge/Airflow-DAGs-017CEE?logo=apacheairflow&logoColor=white)](orchestration/airflow)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)](src/demand_intel/serving)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)](apps/web)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Terraform](https://img.shields.io/badge/Terraform-AWS-7B42BC?logo=terraform&logoColor=white)](infrastructure/aws)
[![Prometheus](https://img.shields.io/badge/Prometheus%20%2B%20Grafana-monitoring-E6522C?logo=prometheus&logoColor=white)](monitoring)

[Results](#results-at-a-glance) ·
[Architecture](#architecture) ·
[Quickstart](#quickstart) ·
[API](#api) ·
[Methodology](#methodology) ·
[Limitations](#limitations)

</div>

---

Built on the Walmart M5 data (2011–2016), Open-Meteo weather and openFDA food recall reports.
Every number below is produced by `make evaluate` and committed in [`evaluation/`](evaluation/);
none is typed by hand. The experimental contract (splits, metrics, gates, anti-cherry-picking
rules) was fixed in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) before modelling started, and every
later change is recorded in its change log.

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Services and ports](#services-and-ports)
- [CLI](#cli)
- [API](#api)
- [Testing and quality](#testing-and-quality)
- [CI/CD](#cicd)
- [Deployment](#deployment)
- [Orchestration and monitoring](#orchestration-and-monitoring)
- [Methodology](#methodology)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License and data](#license-and-data)

---

## Results at a glance

| Question | Answer | Evidence |
|---|---|---|
| Does the model beat a seasonal naive baseline? | **Yes.** Test WRMSSE 0.622 vs 0.870 (−28%) | Δ −0.30 (95% CI −0.31 to −0.29), wins 5/5 backtest folds; promoted through every gate |
| Does ML beat the best statistical model? | **Yes on all series, not significant on the 3,000-series panel** | vs per-class stats model: wins 5/5 folds; vs ETS/ARIMA on panel: CI crosses 0 |
| Do LSTM / TFT justify their cost? | **No** | No significant gain over LightGBM; TFT takes 924 s/fold vs 63 s |
| Are the P10 / P90 quantiles calibrated? | **Yes** | Inside the discrete-count calibration bracket on 5/5 folds |
| What drives demand? | **Calendar ≫ price > weather** | Adding calendar features: backtest WRMSSE 0.987 → 0.728; weather adds ~0.003 |
| Does recall text improve forecasts? | **No (null result, reported as such)** | Embeddings arm is significantly *worse* (Δ +0.004, 0/5 folds) |
| Does a probabilistic policy cut inventory cost? | **Yes, in simulation** | −46% cost vs moving-average reorder rule, −28% vs point forecast; stockout days 3,319 → 919 |
| Can it be served fast? | **Yes** | `/v1/forecast` p95 9 ms; `/v1/inventory/recommend` p95 38 → 16 ms after vectorising (SLO 150 ms) |

Full verdicts with confidence intervals: [`docs/acceptance.md`](docs/acceptance.md).
Inventory costs, lead times and stock levels are **scenarios** (M5 has no stock data) and are
labelled that way everywhere they appear.

### The promotion gate said no, then a pre-registered fix passed it

Plain LightGBM (`lgbm`) passed four of the five promotion gates: improvement, test window, no
slice regression and data quality. It failed the fifth: test-window bias was **−0.0503 against
a ±0.05 limit**, so it was not promoted and seasonal naive stayed champion. Loosening the limit
after seeing the result would be the cherry-picking the spec forbids.

Instead, a fix was written into the spec change log **before** it was run (spec v0.5):
`lgbm_bc` scales the point forecast by a per-category factor Σy/Σŷ learned on each fold's
28-day pre-origin window, clipped to ±10%, with the gates unchanged. Result:

| Model | Backtest WRMSSE | Test WRMSSE | Backtest bias | Test bias | Gate |
|---|---|---|---|---|---|
| `lgbm` | 0.674 | 0.640 | −0.041 | **−0.0503** | rejected |
| `lgbm_bc` | 0.644 | 0.622 | −0.013 | **−0.0462** | **promoted** |

`lgbm_bc` is now the registered and served champion
([`evaluation/registry.json`](evaluation/registry.json) records both decisions with every
gate). The correction removes most of the backtest bias, but on the test window it clears the
limit by only 0.004, so the under-forecast is reduced, not solved. The live monitor still flags
FOODS (−5.5%) and a few stores (see
[`evaluation/monitoring_report.json`](evaluation/monitoring_report.json)).

---

## Architecture

```mermaid
flowchart TD
    subgraph Sources
        M5[M5 sales · prices · calendar]
        WX[Open-Meteo weather]
        FDA[openFDA recalls]
    end

    M5 & WX & FDA --> ING[Ingest + validate<br/>checksums · Pandera contracts · quarantine · ledger]

    ING --> BT[Rolling-origin backtest<br/>5 folds + test window]
    BT --> STAT[Seasonal naive · ETS · ARIMA · TSB]
    BT --> LGB[Global LightGBM<br/>direct, 3 horizon buckets]
    BT --> DL[LSTM · Temporal Fusion Transformer]

    FDA --> NLP[Taxonomy · weak labels · human gold set<br/>TF-IDF · embeddings · fine-tuned encoder]
    NLP --> SIG[State × week text signals] --> BT

    STAT & LGB & DL --> GATE{Promotion gates<br/>MLflow registry}
    GATE --> Q[7 quantiles / day]
    Q --> LTD[Lead-time demand<br/>correlated sample paths]
    LTD --> OUL[Order-up-to level]

    GATE --> API[FastAPI]
    OUL --> API
    API --> WEB[Next.js app]
    API --> PROM[Prometheus / Grafana]

    AF[Airflow DAGs] -.schedules.-> BT
    TF[Terraform: S3 · ECR · ECS Fargate · ALB · RDS · CloudWatch] -.hosts.-> API
```

## Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, Polars, PyArrow, Pandera, checksum manifests, ingestion ledger (Postgres) |
| Models | LightGBM (quantile, SHAP), statsforecast, PyTorch (LSTM, TFT), scikit-learn, transformers, sentence-transformers |
| MLOps | MLflow registry, champion/challenger gates, Airflow, PSI / coverage / JS-divergence monitors |
| Serving | FastAPI, Uvicorn, in-memory feature store, Prometheus metrics, Docker |
| Front end | Next.js 15, React 19, TypeScript (overview, forecast, drivers, events, inventory, lab) |
| Infra | Terraform on AWS (S3, ECR, ECS Fargate, ALB, RDS, CloudWatch), GitHub Actions with OIDC |
| Quality | uv, ruff, mypy `--strict` (source and tests), pytest + coverage, pre-commit |

---

## Quickstart

**Requirements:** Python 3.11, [uv](https://docs.astral.sh/uv/), Docker, Node 20.
Development targets GitHub Codespaces (4 cores / 16 GB); a [dev container](.devcontainer/) is
included.

```bash
git clone https://github.com/JCHETAN26/demand-intel && cd demand-intel
cp .env.example .env
make install          # uv sync + pre-commit hooks
make check            # ruff + mypy (strict) + unit tests
make up               # Postgres + MLflow + API + web via docker compose
```

Or run the web app outside Docker with hot reload:

```bash
cd apps/web && npm install && npm run dev     # http://localhost:3000
```

### Reproducing every result from raw data

Accept the [M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
on Kaggle, export `KAGGLE_USERNAME` and `KAGGLE_KEY`, then:

```bash
make evaluate         # ingest → backtest → NLP → multimodal → gates → inventory → benchmark (hours on 4 cores)
make report           # rebuild tables and docs/acceptance.md from existing runs (minutes)
```

Optional dependency groups are installed as needed:

| Extra | Installs | Command |
|---|---|---|
| `ml` | scikit-learn, statsforecast, LightGBM, Optuna, MLflow, SHAP | `uv sync --extra ml` |
| `deep` | PyTorch (CPU wheels by default) | `uv sync --extra deep` |
| `nlp` | transformers, PEFT, sentence-transformers, ONNX Runtime | `uv sync --extra nlp` |
| `api` | FastAPI, Uvicorn, prometheus-client, boto3 | `uv sync --extra api` |
| `monitoring` | Evidently | `uv sync --extra monitoring` |
| `versioning` | DVC (S3) | `uv sync --extra versioning` |

---

## Configuration

Settings are read from the environment (or `.env`) with the `DI_` prefix
([`src/demand_intel/config.py`](src/demand_intel/config.py)).

| Variable | Default | Purpose |
|---|---|---|
| `DI_DATABASE_URL` | `postgresql+psycopg://demand_intel:demand_intel@localhost:5432/demand_intel` | Ingestion ledger and metadata |
| `DI_MLFLOW_TRACKING_URI` | `http://localhost:5000` | Experiment tracking and model registry |
| `DI_S3_BUCKET` | unset | Serving bundle location in AWS |
| `DI_DATA_DIR` | `./data` | Raw and processed data |
| `DI_EVALUATION_DIR` | `./evaluation` | Generated results |
| `DI_SERVING_DIR` | `./data/serving` | Serving bundle the API loads at startup |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | unset | M5 download (only for `make evaluate`) |

## Services and ports

| Service | Port | Started by |
|---|---|---|
| API (FastAPI) | `8000` | `make up` |
| Web (Next.js) | `3000` | `make up` |
| MLflow | `5000` | `make up` |
| Postgres | `5432` | `make up` |
| Airflow | `8080` | `docker compose --profile airflow up -d` |
| Prometheus | `9090` | `docker compose --profile monitoring up -d` |
| Grafana | `3001` | `docker compose --profile monitoring up -d` |

## CLI

Every pipeline step is a command: `uv run demand-intel --help`.

| Stage | Commands |
|---|---|
| Data | `ingest m5`, `ingest weather`, `ingest text`, `splits`, `eda` |
| Forecasting | `backtest`, `explain`, `forecast-batch` |
| NLP | `nlp-prepare`, `nlp-train`, `nlp-signals`, `nlp-benchmark`, `nlp-import-gold` |
| Decisions | `inventory`, `bundle-risk`, `bundle-extras` |
| MLOps | `promote`, `monitor`, `report`, `acceptance` |

---

## API

| Method | Endpoint | Returns |
|---|---|---|
| `POST` | `/v1/forecast` | 7 quantiles per day for one series, 1–28 days ahead |
| `POST` | `/v1/inventory/recommend` | Order-up-to level and order quantity from lead-time demand |
| `POST` | `/v1/events/extract` | Hazard type, severity and affected states from recall text |
| `GET` | `/v1/risk` | Stockout-risk KPIs against a simple reorder rule |
| `GET` | `/v1/series`, `/v1/series/{item}/{store}` | Series catalogue and history |
| `GET` | `/v1/summary`, `/v1/lab`, `/v1/events/recent` | Data behind the web app |
| `GET` | `/v1/health`, `/metrics` | Liveness and Prometheus metrics |

Interactive docs are served at `http://localhost:8000/docs`.

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' \
  -d '{"item_id": "FOODS_3_090", "store_id": "CA_1", "horizon": 28}'
```

Latency from [`evaluation/serving_benchmark.json`](evaluation/serving_benchmark.json) (2
workers, 4 vCPU, client and server on the same machine):

| Endpoint | p50 | p95 | p99 | Throughput | SLO (p95) |
|---|---|---|---|---|---|
| `/v1/forecast` | 6 ms | 9 ms | 12 ms | 749 rps | 100 ms |
| `/v1/inventory/recommend` | 11 ms | 16 ms | 19 ms | 382 rps | 150 ms |
| `/v1/events/extract` | 6 ms | 8 ms | 9 ms | 759 rps | 300 ms |

---

## Testing and quality

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict over src/ and tests/
make test        # unit tests (excludes integration, data and gpu markers)
make check       # all three
```

| Marker | Needs | Runs in CI |
|---|---|---|
| *(none)* | nothing | yes, with coverage |
| `integration` | Postgres / MLflow (docker services) | yes, against a Postgres 16 service |
| `data` | raw datasets in `data/raw` | no |
| `gpu` | a CUDA device | no |

Pre-commit runs the same ruff and mypy from the project environment, so hooks, CI and editors
use one version.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, every PR | ruff lint + format check, strict mypy, unit tests with coverage; integration tests against Postgres |
| [`deploy.yml`](.github/workflows/deploy.yml) | manual | builds API and web images, pushes to ECR, rolls the ECS services (OIDC, no long-lived keys) |

## Deployment

The AWS stack lives in [`infrastructure/aws`](infrastructure/aws): S3, ECR, ECS Fargate, ALB,
RDS and CloudWatch, with no Kubernetes and no GPUs (the API serves precomputed forecasts and a
CPU event model).

```bash
cd infrastructure/aws
terraform init && terraform apply
```

Then sync the serving bundle to S3 and run the **deploy** workflow, which needs the
`AWS_ROLE_ARN` and `AWS_REGION` repository secrets. Rough cost at the defaults is $90–130 per
month; see [`infrastructure/aws/README.md`](infrastructure/aws/README.md) for the full steps
and `terraform destroy` when you're done.

## Orchestration and monitoring

| DAG | Schedule | Steps |
|---|---|---|
| `forecast_training` | weekly | validate → profile → backtest → train → evaluate → gate/register → publish → monitor |
| `nlp_training` | monthly | ingest/validate → preprocess → baselines → fine-tune → evaluate → benchmark → register |

The monitoring profile ships Prometheus alert rules ([`monitoring/alerts.yml`](monitoring/alerts.yml))
and a provisioned Grafana API dashboard. The `monitor` command checks feature drift (PSI),
quantile coverage and forecast bias by slice, and writes
[`evaluation/monitoring_report.json`](evaluation/monitoring_report.json).

---

## Methodology

**Backtesting.** Five rolling 28-day folds (Dec 2015 – Apr 2016, Christmas included) are used
for model selection. One test window, the M5 public validation period, is touched once per
model family. Seasonal naive and naive reproduce the published M5 WRMSSE on the test window,
which validates the metric code.

**No leakage by construction.** Models see sales only up to the forecast origin. Tests show
that sales features stay the same when later data is removed. Known-future inputs (calendar,
prices, SNAP) are kept separate from past-only inputs.

**Probabilistic forecasts.** LightGBM is trained directly on 3 horizon buckets (1–7, 8–14,
15–28 days) and outputs 7 quantiles. Calibration for count data uses the bracket
P(y < q) ≤ τ ≤ P(y ≤ q), because P(y ≤ q) = τ cannot hold for discrete demand.

**Lead-time demand is a sum, not a sum of quantiles.** It is built from sample paths whose
day-to-day correlation is calibrated on an earlier fold. Adding daily P90s would overstate
safety stock.

**Statistics.** Every model comparison reports a paired difference with a 95% bootstrap CI,
fold wins and a Wilcoxon test ([`paired_tests.csv`](evaluation/paired_tests.csv)). Complexity
(deep models, text) is added only when it beats the simpler model under these rules.

**Text mapping.** M5 items are anonymised, so recalls can map only to FOODS × state × week.
The multimodal arms are compared against the weather arm on the same panel. The null result
is reported as a finding.

**Labels.** Training labels for recall hazard type come from rules (weak supervision). The
test set is human-validated, and agreement with weak labels is never used to choose a model.

---

## Repository layout

```
src/demand_intel/
  pipelines/ingestion   M5, weather, recall corpus: download, checksum manifests, ledger
  pipelines/validation  Pandera contracts with row-level rejection reasons
  pipelines/features    horizon-bucketed features (leakage-tested), text signals
  pipelines/inference   batch forecasts and the serving bundle
  analysis/             data quality, EDA, demand classes, panel sample
  forecasting/          splits, harness, metrics (WRMSSE, pinball, calibration), comparison
  models/               baselines, statsforecast, LightGBM (+SHAP), LSTM, TFT, NLP
  inventory/            lost-sales simulator, policies P0–P3, historical replay
  mlops/                promotion gates, MLflow registry, monitoring (PSI, coverage, JS)
  serving/              FastAPI app, in-memory store, inventory and event endpoints
apps/web                Next.js app
apps/api                API container
orchestration/airflow   forecast_training (weekly) and nlp_training (monthly) DAGs
infrastructure/aws      Terraform
monitoring/             Prometheus rules, Grafana dashboard
evaluation/             generated results (committed)
docs/                   generated EDA and acceptance reports
tests/                  unit and integration tests
```

---

## Limitations

- **Inventory results are simulated.** M5 has no on-hand stock, lead times or costs. The policy
  ranking holds across the stockout-cost scenarios tested, but the dollar figures are not real
  savings.
- **Bias is reduced, not solved.** `lgbm_bc` clears the ±5% gate by 0.004 on the test window;
  the monitor still warns on FOODS and six stores (−5% to −8.5%).
- **NLP verdict pending.** The fine-tuned encoder cannot be judged until the human-labelled test
  set is complete. Weak-label scores are not reported as accuracy.
- **Serving benchmark** runs client and server on one 4-vCPU machine, not the 1 vCPU / 2 GB
  target container in the spec.

## Roadmap

- [x] Pre-registered bias correction for LightGBM, then rerun the promotion gate (`lgbm_bc` promoted)
- [ ] Finish the human-labelled recall test set and close the NLP question
- [ ] Rerun the serving benchmark on the target container size
- [ ] Public demo deployment

## Contributing

1. Branch from `main` and run `make install` to set up the hooks.
2. Keep `make check` green; CI runs the same checks on every pull request.
3. Results in `evaluation/` are generated, so regenerate them with `make report` (or
   `make evaluate`) rather than editing them by hand.
4. Changes to splits, metrics or promotion gates go into the
   [`PROJECT_SPEC.md`](PROJECT_SPEC.md) change log **before** they are run.

## License and data

Code is released under the [MIT License](LICENSE). The M5 data is subject to the
[Kaggle competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy) and is
not redistributed here. Weather comes from [Open-Meteo](https://open-meteo.com/) and recall
reports from [openFDA](https://open.fda.gov/).
