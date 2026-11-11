# Demand Intelligence

**Probabilistic demand forecasting for 30,490 retail SKU × store series, turned into
replenishment decisions, served behind an API with promotion gates and monitoring.**

Built on the Walmart M5 data (2011–2016), Open-Meteo weather and openFDA food recall reports.
Every number below is produced by `make evaluate` and committed in [`evaluation/`](evaluation/);
none is typed by hand. The experimental contract (splits, metrics, gates, anti-cherry-picking
rules) was fixed in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) before modelling started, and every later
change is in its change log.

[![CI](https://github.com/JCHETAN26/demand-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/JCHETAN26/demand-intel/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![mypy strict](https://img.shields.io/badge/mypy-strict-blue)

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

### The promotion gate said no, then the fix was pre-registered and it said yes

Plain LightGBM (`lgbm`) passed four of the five promotion gates: improvement, test window, no
slice regression and data quality. It failed the fifth: test-window bias was **−0.0503 against a
±0.05 limit**, so it was not promoted and seasonal naive stayed champion. Loosening the limit
after seeing the result would be the cherry-picking the spec forbids.

Instead, a fix was written into the spec change log **before** it was run (spec v0.5):
`lgbm_bc` scales the point forecast by a per-category factor Σy/Σŷ learned on each fold's 28-day
pre-origin window, clipped to ±10%, with the gates unchanged. Result:

| | backtest WRMSSE | test WRMSSE | backtest bias | test bias | gate |
|---|---|---|---|---|---|
| `lgbm` | 0.674 | 0.640 | −0.041 | **−0.0503** | rejected |
| `lgbm_bc` | 0.644 | 0.622 | −0.013 | **−0.0462** | **promoted** |

`lgbm_bc` is now the registered and served champion
([`evaluation/registry.json`](evaluation/registry.json) records both decisions with every gate).
The honest reading: the correction removes most of the backtest bias, but on the test window it
clears the limit by only 0.004, so the under-forecast is reduced, not solved. The live monitor
still flags FOODS (−5.5%) and a few stores (see
[`evaluation/monitoring_report.json`](evaluation/monitoring_report.json)).

---

## Architecture

```
 M5 sales · prices · calendar      weather (Open-Meteo)      openFDA recalls
            │                              │                       │
            └──── ingest + validate (checksums, Pandera contracts, quarantine, ledger) ┐
                                                                                       │
  rolling-origin backtest (5 folds + test) ── seasonal naive · ETS · ARIMA · TSB ──────┤
                                           ── global LightGBM (direct, 3 horizon buckets)
                                           ── LSTM · Temporal Fusion Transformer ──────┤
                                                                                       │
  recall text ── taxonomy · weak labels · human gold set ── TF-IDF · embeddings ───────┤
                                                        ── fine-tuned encoder ─────────┤
                                                        ── state × week signals ───────┘
                                     │
          7 quantiles/day  →  lead-time demand (correlated sample paths)  →  order-up-to level
                                     │
   MLflow registry + promotion gates · Airflow DAGs · FastAPI · Next.js · Prometheus/Grafana
                                     │
                 Terraform: S3 · ECR · ECS Fargate · ALB · RDS · CloudWatch
```

| Layer | Tools |
|---|---|
| Data | pandas, Pandera, checksum manifests, ingestion ledger (Postgres) |
| Models | LightGBM (quantile, SHAP), statsforecast, PyTorch (LSTM, TFT), scikit-learn, transformers |
| MLOps | MLflow registry, champion/challenger gates, Airflow, PSI / coverage / JS-divergence monitors |
| Serving | FastAPI, in-memory feature store, Prometheus metrics, Docker |
| Front end | Next.js (overview, forecast, drivers, events, inventory, lab) |
| Infra | Terraform on AWS, GitHub Actions (CI + OIDC deploy) |
| Quality | ruff, mypy `--strict`, pytest, pre-commit |

---

## Quickstart

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Docker, Node 20. Development
targets GitHub Codespaces (4 cores / 16 GB).

```bash
git clone https://github.com/JCHETAN26/demand-intel && cd demand-intel
make install          # uv sync + pre-commit hooks
make check            # ruff + mypy (strict) + unit tests
make up               # Postgres + MLflow + API via docker compose
cd apps/web && npm install && npm run dev     # web app on :3000
```

To reproduce every result from raw data, accept the
[M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy) on Kaggle,
export `KAGGLE_USERNAME` and `KAGGLE_KEY`, then:

```bash
make evaluate         # ingest → backtest → NLP → multimodal → gates → inventory → benchmark (hours)
make report           # rebuild tables and docs/acceptance.md from existing runs (minutes)
```

Each step is also a CLI command: `uv run demand-intel --help`.
Monitoring stack: `docker compose --profile monitoring up -d` (Prometheus :9090, Grafana :3001).

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

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' \
  -d '{"item_id": "FOODS_3_090", "store_id": "CA_1", "horizon": 28}'
```

Latency comes from [`evaluation/serving_benchmark.json`](evaluation/serving_benchmark.json): 2
workers, 4 vCPU, with client and server on the same machine.

| Endpoint | p50 | p95 | p99 | Throughput | SLO (p95) |
|---|---|---|---|---|---|
| `/v1/forecast` | 6 ms | 9 ms | 12 ms | 749 rps | 100 ms |
| `/v1/inventory/recommend` | 11 ms | 16 ms | 19 ms | 382 rps | 150 ms |
| `/v1/events/extract` | 6 ms | 8 ms | 9 ms | 759 rps | 300 ms |

---

## Methodology

**Backtesting.** Five rolling 28-day folds (Dec 2015 – Apr 2016, Christmas included) are used
for model selection. One test window, the M5 public validation period, is touched once per
model family. Seasonal naive and naive reproduce the published M5 WRMSSE on the test window,
which validates the metric code.

**No leakage by construction.** Models see sales only up to the forecast origin. Tests show
that sales features stay the same when later data is removed. Known-future inputs (calendar,
prices, SNAP) are kept separate from past-only inputs.

**Probabilistic forecasts.** LightGBM is trained directly on 3 horizon buckets (1–7, 8–14, 15–28 days) and outputs 7
quantiles. Calibration for count data uses the bracket P(y < q) ≤ τ ≤ P(y ≤ q), because
P(y ≤ q) = τ cannot hold for discrete demand.

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
orchestration/airflow   forecast_training (weekly) and nlp_training (monthly) DAGs
infrastructure/aws      Terraform
monitoring/             Prometheus rules, Grafana dashboard
evaluation/             generated results (committed)
docs/                   generated EDA and acceptance reports
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

---

Released under the MIT License (see [`LICENSE`](LICENSE)).
