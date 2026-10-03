# Demand Intelligence

SKU-level demand forecasting under uncertainty, turned into replenishment decisions, with a
test of whether text signals from FDA food recalls improve the forecasts. Built on the
Walmart M5 data (30,490 item × store series, 2011–2016), historical weather, and openFDA
food enforcement reports.

**Results:** [`docs/acceptance.md`](docs/acceptance.md) answers the project's ten acceptance
questions with the numbers that decided them. Every figure there, in the web app and in
`evaluation/` is produced by `make evaluate`; none is typed by hand
([spec §10](PROJECT_SPEC.md#10-reproducible-evaluation)).

**Contract:** [`PROJECT_SPEC.md`](PROJECT_SPEC.md) fixes the definitions, splits, metrics,
ablations and promotion gates. Changes made after modelling started are listed, with
reasons, in its change log.

## What it does

```
 M5 sales · prices · calendar      weather (Open-Meteo)      openFDA recalls
            │                              │                       │
            └──── ingest + validate (checksums, contracts, quarantine, ledger) ────┐
                                                                                   │
  rolling-origin backtest (5 folds + test) ── seasonal naive · ETS · ARIMA · TSB ──┤
                                           ── global LightGBM (direct horizons) ───┤
                                           ── LSTM · Temporal Fusion Transformer ──┤
                                                                                   │
  recall text ── taxonomy · weak labels · human gold set ── TF-IDF · embeddings ───┤
                                                        ── fine-tuned encoder ─────┤
                                                        ── state × week signals ───┘
                                     │
          P10 / P50 / P90  →  lead-time demand (sample paths)  →  order-up-to level
                                     │
     MLflow registry + promotion gates · Airflow DAGs · FastAPI · Next.js · Prometheus
```

| Question | Where the answer comes from |
|---|---|
| Expected demand per SKU for 7/14/28 days | `POST /v1/forecast`, `evaluation/lightgbm_results.csv` |
| How uncertain | 7 quantiles per day; `evaluation/probabilistic_calibration.csv` |
| What drives demand | feature ablation (`forecast_ablation.csv`) and TreeSHAP (`evaluation/shap/`) |
| Does deep learning beat LightGBM | `model_comparison.csv`, `paired_tests.csv` |
| Does text help | `multimodal_ablation.csv` (arms A–F, paired tests, FOODS slice) |
| Does fine-tuning beat generic embeddings | `nlp_results.csv`, `nlp_inference_benchmark.csv` |
| Stockout risk / how much to order | `POST /v1/inventory/recommend`, `/v1/risk` |
| Does better forecasting improve inventory | `inventory_simulation.csv`, `inventory_policy_differences.csv` |

## Layout

```
src/demand_intel/
  pipelines/ingestion   M5, weather, recall corpus: download, checksum manifests, ledger
  pipelines/validation  Pandera contracts with row-level rejection reasons
  pipelines/features    horizon-bucketed features (leakage-tested), text signals
  pipelines/inference   batch forecasts and the serving bundle
  analysis/             data quality, EDA, demand classes, PANEL sample
  forecasting/          splits, harness, metrics (WRMSSE, pinball, calibration), comparison
  models/               baselines, statsforecast, LightGBM (+SHAP), LSTM, TFT, NLP
  inventory/            lost-sales simulator, policies P0–P3, historical replay
  mlops/                promotion gates, registry (MLflow), monitoring (PSI, coverage, JS)
  serving/              FastAPI app, in-memory store, inventory and event endpoints
apps/web                Next.js app: overview, forecast, drivers, events, inventory, lab
orchestration/airflow   forecast_training (weekly) and nlp_training (monthly) DAGs
infrastructure/aws      Terraform: S3, ECR, ECS Fargate, ALB, RDS, CloudWatch
monitoring/             Prometheus rules, Grafana dashboard
scripts/                experiment runner, serving benchmark
evaluation/             generated results (committed)
docs/                   generated EDA and acceptance reports
```

## Running it

Development runs in GitHub Codespaces (4 cores / 16 GB). Add `KAGGLE_USERNAME` and
`KAGGLE_KEY` as Codespaces secrets and accept the M5 competition rules on Kaggle once.

```bash
make install                      # uv sync + pre-commit
make check                        # ruff + mypy (strict) + pytest
make up                           # Postgres + MLflow + API (docker compose)
make evaluate                     # everything, from raw data (hours)
make report                       # rebuild tables and acceptance from existing runs
docker compose --profile monitoring up -d   # Prometheus :9090, Grafana :3001
cd apps/web && npm install && npm run dev   # web app on :3000
```

Individual steps are CLI commands: `uv run demand-intel --help`.

## Design choices worth knowing

- **Backtesting:** 5 rolling 28-day folds (Dec 2015 – Apr 2016, Christmas included) for
  selection, and one test window (the M5 public validation period) touched once per model
  family. The test-window WRMSSE of naive and seasonal naive reproduces the published M5
  values, which checks the metric code.
- **No leakage by construction:** models receive sales only up to the origin; tests prove
  that sales features don't change when later data is removed.
- **Calibration for counts:** a calibrated quantile of discrete demand satisfies
  P(y<q) ≤ τ ≤ P(y≤q), not P(y≤q) = τ (spec v0.3).
- **Lead-time demand is a sum:** quantiles come from sample paths with a day-to-day
  correlation calibrated on an earlier fold, never from adding daily quantiles.
- **Text mapping:** M5 items are anonymized, so recalls map only to FOODS × state (named
  or nationwide) × publication date. A null multimodal result is a valid outcome.
- **Labels:** recall hazard types for training come from rules (weak supervision); the test
  set is human-validated, and weak-label agreement is never used to pick a model.
- **Inventory:** M5 has no stock data; costs, lead times and stock levels are scenarios
  and are labelled as such wherever they appear.
