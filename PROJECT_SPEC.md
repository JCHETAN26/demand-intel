# Demand Intelligence — Project Specification & Experimental Contract

Status: **v0.5** (v0.1 frozen before any modeling; see the Change Log).
Any change to a definition, metric, split, or gate after modeling begins must be
recorded in the [Change Log](#12-change-log) with the reason and date. Results
produced under an older spec version must say which version they used.

---

## 1. Purpose

Demand Intelligence forecasts SKU-level demand under uncertainty and turns those
forecasts into replenishment decisions. It also tests whether external signals,
including signals extracted from text, actually improve those forecasts.

> history → forecast distribution (P10/P50/P90) → explain drivers →
> detect external events → stockout / excess risk → replenishment quantity →
> simulated inventory outcome.

The project has **two success levels, and both must be reported**:

| Level | Question | Primary evidence |
|-------|----------|------------------|
| ML | Are the forecasts better? | WRMSSE / WAPE / bias / calibration under rolling-origin backtesting (§5) |
| Business | Are the decisions better? | Total simulated cost and service level under identical replay conditions (§8) |

**Guiding principle:** every sophisticated component is evaluated against a
simpler baseline that sits underneath it. A component that does not beat its
baseline under the metrics defined here is still reported, but it is not
promoted.

---

## 2. Datasets and scope boundaries

| ID | Dataset | Role | Grain |
|----|---------|------|-------|
| A | M5 Forecasting (Walmart) | Demand history, prices, calendar, SNAP, hierarchy | item × store × day |
| B | Historical weather (§2.3) | External structured covariates | state × day |
| C | Public event/recall text corpus (§2.4) | NLP training and evaluation; candidate forecasting signal | document (timestamped) |

Raw files are immutable. They live in `data/raw/<dataset>/` (later
`s3://demand-intel/raw/<dataset>/`) and are never edited in place.

### 2.1 Known facts about M5 that constrain the design

- **Hierarchy:** 3,049 items × 10 stores (CA_1–4, TX_1–3, WI_1–3) = **30,490
  bottom-level series**. There are 3 categories (FOODS, HOBBIES, HOUSEHOLD) and 7
  departments (FOODS_1–3, HOBBIES_1–2, HOUSEHOLD_1–2). The official evaluation
  uses 12 aggregation levels, which give 42,840 series in total.
- **Calendar:** `d_1` = 2011-01-29 through `d_1969` = 2016-06-19. Day `d_n` is
  `2011-01-29 + (n−1)` days.
- **Sales files:** `sales_train_validation.csv` covers `d_1..d_1913`
  (to 2016-04-24). `sales_train_evaluation.csv` covers `d_1..d_1941`
  (to 2016-05-22). Labels for `d_1942..d_1969` are not used in this project,
  even if a copy is found, so that the test period is the same for everyone (§5.1).
- **Prices** are weekly (`wm_yr_wk`) per store × item. A missing price means
  the item was not yet sold in that store. Those days are **not** zero demand;
  they are excluded from training and evaluation.
- **No promotion flag exists.** "Promotion" in this project always means a
  **price-derived discount proxy** (§4.3) and is labeled as a proxy wherever
  it appears.
- **Sales ≠ demand.** M5 records units sold. Stockouts are unobserved and show up
  as zeros. Every result treats sales as a demand proxy. Phase 7 flags
  *suspicious zero runs*: runs of zeros of length ≥ 14 days that are unlikely
  given the item's trailing-90-day share of zero days, p0^L < 0.001 (v0.2).
  Only runs of 14–55 days count as suspected stockouts. Runs of 56+ days are
  *dormant periods* and are reported separately, not masked.
  The forecasting and inventory headlines are re-run with those days masked as
  a sensitivity check.
- **Store locations are given only at state level.** External covariates can
  therefore be aligned only at state × day (§2.3).
- The bottom level is highly intermittent: most item-store-days are zero. The
  exact share is measured in Phase 7, not assumed.

### 2.2 Known-future vs. past-only inputs

The input classification decides what a model may see for a target day
`t + h` when the forecast is made at origin `t`.

| Input | Class | Justification |
|-------|-------|---------------|
| Calendar fields, events, SNAP days | **Known future** | Published in advance |
| Sell price | **Known future** (assumption A1) | A retailer sets its own prices. Ablation C (§6.2) also reports a past-only price variant |
| Sales and all derived lags/rolling stats | **Past only** | — |
| Weather | **Past only** in every deployable model | Historical weather *forecasts* are not available |
| Text-derived event signals | **Past only**, indexed by publication/report date, not event date | Prevents look-ahead |

Any experiment that feeds actual future weather is labeled **ORACLE — not
deployable**. It is reported only as an upper bound.

### 2.3 Weather (Dataset B)

- **Source:** Open-Meteo historical archive (ERA5 reanalysis), daily. NOAA
  GHCN-Daily is the fallback.
- **Spatial alignment:** for each state (CA, TX, WI), take a population-weighted
  average of 3 major metros. The metro list and weights are versioned in
  `configs/weather_locations.yaml`. This is a coarse proxy because store
  locations are unknown, and every weather result says so.
- **Variables:** daily max/min/mean temperature, precipitation, snowfall. Derived
  variables are extreme heat (Tmax > state's 95th percentile for that calendar
  month), extreme cold (Tmin < 5th percentile), and anomaly (deviation from the
  2011–2015 day-of-year climatology, computed from the training period only).
- **Coverage:** every state × day in `d_1..d_1941` must be present. Gaps longer
  than 2 days fail validation.

### 2.4 Text corpus (Dataset C): selection criteria and scientific constraint

M5 items are anonymized (`FOODS_3_090`), so **no item-level text mapping is
possible.** The most a text signal can map to is **category or department ×
state × week**. We do not invent finer alignment.

A corpus is eligible only if it meets all of these:

1. It is public and its license allows redistribution of derived labels.
2. Each document has a reliable timestamp, and the corpus overlaps 2011–2016.
3. Documents can be mapped to M5 categories (food / household / hobbies-toys)
   and, ideally, to US states.
4. It has some existing structured fields that can serve as partial labels.

**Leading candidates**, to be confirmed by ingestion in Phase 3:

- **openFDA food enforcement reports.** These include the recall reason, product
  description, distribution pattern (states), FDA classification I/II/III, and
  dates. They map to FOODS.
- **CPSC recalls.** These include the title, hazard, product, and recall date.
  They map to HOUSEHOLD and HOBBIES (toys, household goods).

**Consequence for the taxonomy:** Phase 18 derives the event taxonomy from the
corpus that is actually chosen. If that corpus contains recalls, the taxonomy
will be recall-centric, for example hazard type, product category, severity, and
geographic scope. We will not create classes such as `PRODUCT_LAUNCH` that the
corpus cannot support. If a second corpus is added to broaden the taxonomy, it
is held to the same four criteria.

**Pre-registered expectation:** recall events are unlikely to move store-level
demand for anonymized departments measurably. The multimodal experiment (§6.2)
is designed so that a null result is a valid, publishable outcome. If the
mapping is too weak, the NLP system is evaluated as a standalone component
(§7) and the forecasting ablation reports "not integrated, with reason".

---

## 3. Questions the system answers

The artifacts listed here are produced by `make evaluate` (§10).

| # | Question | Method | Answer artifact |
|---|----------|--------|-----------------|
| Q1 | How much demand should we expect per item-store over the next 7/14/28 days? | Champion model under rolling-origin backtest (§5) | `lightgbm_results.csv`, `model_comparison.csv` |
| Q2 | How uncertain is that prediction? | Quantile forecasts with calibration (§5.5) | `probabilistic_calibration.csv` |
| Q3 | How much do price, the promo proxy, holidays, and weather affect demand? | Feature ablation (§6.1) and SHAP by category | `forecast_ablation.csv`, SHAP artifacts |
| Q4 | Can deep learning beat statistical/ML baselines? | Model ladder F0–F5 (§5.3) | `model_comparison.csv` |
| Q5 | Can text improve forecasts? | Multimodal ablation A–F (§6.2) | `multimodal_ablation.csv` |
| Q6 | Does fine-tuning beat generic text embeddings? | NLP ladder N0–N4 (§7.3) | `nlp_baselines.csv`, `nlp_finetuning.csv` |
| Q7 | Which SKUs are at risk of stockout? | P(lead-time demand > on-hand + on-order) from the forecast distribution | API + `inventory_simulation.csv` |
| Q8 | How much should be replenished? | Order-up-to policy on lead-time demand quantiles (§8.2) | `/v1/inventory/recommend` |
| Q9 | Does better forecasting improve inventory outcomes? | Historical replay of policies P0–P3 vs oracle (§8) | `inventory_simulation.csv` |

---

## 4. Definitions

### 4.1 Forecast origin and horizon

- **Origin `t`:** the last day whose sales are observed. Features may use only
  data dated `≤ t`, except for the known-future inputs in §2.2.
- **Horizon `h ∈ {1..28}`:** a forecast for day `t + h`.
- **Reported horizons:** windows **H7 = h 1–7, H14 = h 1–14, H28 = h 1–28**.
  Metrics are also reported per `h`.
- **Strategy for tabular models:** use **direct, horizon-bucketed** models for
  buckets {1–7, 8–14, 15–28}. A bucket's model uses only sales lags ≥ the
  bucket's maximum `h`. This removes recursive error feedback and any chance of
  using unobserved lags. A recursive variant may be reported, but it is not the
  default.

### 4.2 Eligible series-days

A series-day is eligible for training and evaluation if the item has a price in
that store-week. Days before an item's first sale in a store are excluded from
the scaling denominators (§5.4), as in the M5 rules.

### 4.3 Price and promotion proxy

- `price_change` = price / previous week's price − 1.
- `price_vs_rolling` = price / mean price over the trailing 8 weeks − 1.
- `discount_proxy` = 1 if price ≤ 0.9 × the max price over the trailing 8
  weeks, else 0.
- `price_rel_dept` = price / median price of the item's department in the
  same store-week.

### 4.4 Text-derived temporal signals

A document maps to (category, state set, publication date). Signals are
aggregated over a trailing window ending at the origin:
`events_7d[type]`, `severity_sum_7d`, `positive_7d`, `negative_7d`, and the same
over 28d. Generic-embedding signals are the mean document embedding over the
trailing 28d, reduced by PCA. **The PCA is fit only on training-period
documents.**

---

## 5. Forecasting contract

### 5.1 Splits and rolling-origin backtest

```
           fold 1     fold 2     fold 3     fold 4     fold 5      TEST
origin:    d_1773     d_1801     d_1829     d_1857     d_1885     d_1913
window:  12/07–01/03 01/04–01/31 02/01–02/28 02/29–03/27 03/28–04/24 04/25–05/22
           (2015–16, each 28 days)                                  (2016)
```

- **Backtest folds (5):** origins `d_1773, d_1801, d_1829, d_1857, d_1885`, each
  forecasting the next 28 days. Together they cover 2015-12-07 → 2016-04-24,
  including Christmas/New Year, the Super Bowl, and Easter. They are used for
  model selection, hyperparameter tuning, early stopping, and conformal
  calibration.
- **Test:** origin `d_1913`, forecasting `d_1914..d_1941`
  (2016-04-25 → 2016-05-22). It is **touched once per model family** after all
  tuning is finished. Any re-evaluation is logged and disclosed.
- Training windows expand. Every model may use history only up to its fold's
  origin. A model's training window start, for example "from `d_1000`" for
  memory reasons, is part of its config and is identical across folds.
- The exact origins are written to `evaluation/split_manifest.json` and
  versioned.

### 5.2 Series populations

| Population | Definition | Used by |
|------------|------------|---------|
| **FULL** | all 30,490 bottom-level series | F0, F1, F3 (and F4/F5 if compute allows) |
| **PANEL** | stratified sample of 3,000 series, fixed seed: stratified by category × store × demand class | every model, so that all model families can be compared on identical series |

Demand classes follow Syntetos–Boylan on the training period of the first fold:
smooth, erratic, intermittent, lumpy (ADI cut-off 1.32, CV² cut-off 0.49).
Every metric is reported **overall** and **by category, store, demand class,
volume tercile, and horizon**.

### 5.3 Model ladder

| ID | Model | Scope |
|----|-------|-------|
| F0a | Naive: ŷ_{t+h} = y_t | FULL |
| F0b | Seasonal naive (7): ŷ_{t+h} = y_{t+h−7k} | FULL |
| F0c | Moving average over 28 days | FULL |
| F1 | Statistical: ETS / Holt-Winters (weekly), Croston/TSB for intermittent series | FULL via a vectorized implementation (statsforecast) |
| F2 | ARIMA / SARIMA | PANEL (cost-limited) |
| F3 | Global LightGBM, direct horizon-bucketed (§4.1), Tweedie objective | FULL |
| F4 | LSTM/GRU (PyTorch) | PANEL; FULL if it fits the compute budget (§11) |
| F5 | Temporal Fusion Transformer | PANEL; only if F4 is competitive with F3 or quantile quality motivates it |

### 5.4 Point metrics

| Metric | Role |
|--------|------|
| **WRMSSE** (M5 definition: 12 levels; weights = dollar sales over the 28 days before the origin; scale = in-sample one-step naive MSE from the first nonzero sale) | **Primary**: used for selection and promotion |
| RMSSE (bottom level, unweighted mean) | Secondary |
| WAPE = Σ\|y−ŷ\| / Σ\|y\| | Secondary: the business-facing number |
| MAE, RMSE | Secondary |
| **Bias** = Σ(ŷ−y) / Σy | **Guardrail**: negative means under-forecasting (stockout risk) |

Metrics are computed on PANEL for cross-family comparisons. They are also
computed on FULL for models that ran on FULL.

### 5.5 Probabilistic forecasting

- **Quantiles:** τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}. The UI shows
  P10/P50/P90. P95 exists because the default service level (§8) requires it.
- **Primary metric:** scaled pinball loss averaged over τ (WSPL, the M5 Uncertainty
  definition, applied to our τ set). It is computed at the bottom level, where models
  produce quantiles (v0.3).
- **Calibration:** empirical coverage of each quantile (target ≈ τ). The 80%
  interval [P10, P90] is checked for coverage and mean width. The reliability
  is reported per category and demand class.
- **Acceptance (v0.3, discrete demand):** a τ-quantile is calibrated if
  P(y < q_τ) − 0.03 ≤ τ ≤ P(y ≤ q_τ) + 0.03 on the backtest folds, for τ = 0.10 and
  0.90 overall and per category. P(y ≤ q) = τ alone is not attainable for counts: when
  60% of days are zero, any correct P10 is 0, so 60% of actuals fall at or below it. If raw quantiles fail, split
  conformal adjustment fit on the folds is applied before serving, and both the
  raw and adjusted numbers are reported.
- **Lead-time demand is a sum.** Quantiles of a multi-day sum must **not** be
  computed by adding daily quantiles. Lead-time quantiles come from sample paths
  or from a model trained directly on the cumulative target.

### 5.6 Rules for adding complexity

- **F3 must beat F0b and F1** on fold-averaged WRMSSE. The paired-difference CI
  (§9.2) must exclude 0.
- **F4/F5 are kept only if they beat F3** on WRMSSE or WSPL under the same test,
  **and** the gain is judged worth the cost reported in `model_comparison.csv`
  (training time, inference time, hardware). Otherwise they are reported but
  not promoted. **Deep learning does not win by default.**

---

## 6. Ablations

### 6.1 Feature-family ablation (fixed model F3)

1. Sales history (lags, rolling stats, momentum)
2. + calendar and events
3. + SNAP
4. + price and promotion proxy
5. + hierarchy identifiers
6. + weather (past only)

Each step reports ΔWRMSSE, ΔWAPE, and Δbias with a CI.

### 6.2 Multimodal experiment (fixed model F3; also F5 if promoted)

| Arm | Inputs |
|-----|--------|
| A | Historical demand |
| B | A + calendar/events/SNAP |
| C | B + price/promotion proxy |
| D | C + weather (past only) |
| D* | C + actual weather (**ORACLE — not deployable**) |
| E | D + generic text-embedding summary (§4.4) |
| F | D + fine-tuned event signals (§4.4) |

The comparisons that answer Q5 are **E vs D, F vs D, and F vs E**. They are
evaluated on the departments and states that the text signal can map to, and
also overall. A result counts as consistent only if the paired CI excludes 0
**and** the sign of the improvement holds in at least 4 of 5 folds.

---

## 7. NLP contract

### 7.1 Labels

- **Sources:** existing structured fields (for example FDA classification as
  severity), weak-supervision rules, and teacher-LLM candidate labels.
  **Teacher-LLM output is never ground truth.**
- **Gold set:** at least 600 documents verified by a human, stratified by
  class. Inter-annotator agreement is reported if a second pass exists;
  otherwise there is a self-consistency re-annotation of 100 documents after
  2 weeks, and Cohen's κ is reported.
- **Splits:** by time (train < validation < test by publication date), with
  near-duplicate documents (MinHash Jaccard ≥ 0.8) grouped into the same split.
  The **test set contains only gold labels.**

### 7.2 Output schema

`{event_type, category, direction ∈ {positive, negative, neutral},
severity ∈ [0,1], geography: [state], confidence ∈ [0,1]}`. The output is
validated against a Pydantic schema. *Structured-output validity rate* is a
reported metric.

### 7.3 Model ladder

| ID | Model |
|----|-------|
| N0 | Majority class / rules only |
| N1 | TF-IDF + logistic regression |
| N2 | Generic sentence embeddings + logistic regression |
| N3 | Fine-tuned encoder classifier (for example DeBERTa-v3-small) |
| N4 | LoRA/QLoRA fine-tuned small generative LM (≤ 1.5B parameters given §11) producing the JSON schema |
| T  | Teacher LLM (zero/few-shot): the reference point for the compression study |

### 7.4 Metrics

The primary metric is **macro-F1 on `event_type`** on the gold test set.
Secondary metrics are per-class precision/recall, the confusion matrix, ECE of
confidence, severity MAE, and structured-output validity. The compression
study (Phase 22) reports **macro-F1 × p50/p95 latency × peak memory ×
throughput × $ per 1k documents** for T, N3, and N4 (fp16 / int8 / int4 / ONNX
where applicable).

N4 is promoted over N3 only if its macro-F1 CI excludes N3's, or if it ties on
F1 and adds schema fields that N3 cannot produce.

---

## 8. Inventory contract

All costs, lead times, and initial stock levels are
**SCENARIO — not observed data**. M5 contains no inventory, so these are
simulation assumptions, and they are labeled as such everywhere.

### 8.1 Simulation setup

- **Population:** the PANEL series, over the test window and backtest folds.
- **Mechanics:** a daily review. Lead time `L ∈ {2, 7}` days (default 7).
  Unmet demand is **lost** (no backorders). Each day runs in this order:
  receive arrivals → observe demand → record sales and lost sales → decide the
  order.
- **Initial inventory:** order-up-to level of each policy at the start of the
  window, so all policies start from their own steady state. A sensitivity
  check uses a common start.
- **Costs (default scenario):** holding cost h = 25% of price per year, charged
  daily on ending inventory. Stockout cost p = 1× the unit price per lost unit.
  Sensitivity runs use p ∈ {0.5, 2, 5}× price.
- **Information:** each day's decision uses only forecasts made at origins ≤
  that day. Forecasts are refreshed every 7 days.

### 8.2 Policies

| ID | Policy |
|----|--------|
| P0 | Fixed order-up-to level = 28-day historical mean × (L + 7) |
| P1 | Moving-average forecast + normal safety stock from historical demand std |
| P2 | Champion point forecast (P50), with safety stock from backtest residual std |
| P3 | Champion **probabilistic** forecast: order-up-to = lead-time-demand quantile at the target service level (§5.5 last rule) |
| P* | Perfect-foresight oracle (**ORACLE**: an upper bound only) |

Target cycle service level: 95% by default, with a sensitivity check at 90%
and 98%.

### 8.3 Metrics

Total simulated cost (**primary**), fill rate, cycle service level, stockout
days, lost-sales units, average on-hand, excess inventory (units above
lead-time P95), and inventory turnover. Policies are compared on identical
demand paths. CIs come from a bootstrap clustered by series.

**Q9 is answered by P3 vs P2 vs P1.** Forecasting "improves operations" only if
P3 has lower total cost than P1 at the default scenario **and** keeps that
ordering in at least 3 of the 4 stockout-cost sensitivities.

---

## 9. Statistical conventions

### 9.1 Reproducibility

- Global seed: `20261002`. Seeds are passed explicitly to every estimator,
  sampler, and splitter.
- Each run logs the git SHA, spec version, dataset checksums, feature-set
  version, and split manifest to MLflow.

### 9.2 Uncertainty and comparisons

- **Forecast comparisons:** per-series scaled errors are paired across models.
  A bootstrap with 1,000 resamples is clustered by series, and errors are
  pooled across folds. We report 95% percentile CIs. A Wilcoxon signed-rank
  test on per-series RMSSE serves as a cross-check.
- An "improvement" requires the CI of the paired difference to exclude 0.
  §6.2 adds a fold-consistency rule for the multimodal experiment.
- NLP comparisons use a bootstrap over test documents for macro-F1, and
  McNemar's test for paired accuracy.

### 9.3 Anti-cherry-picking rules

- The metrics, horizons, quantiles, splits, populations, and scenarios in this
  document are fixed.
- Every model trained for a comparison table is reported, including models that
  lose.
- The test window is evaluated once per model family. Any re-run is logged and
  disclosed.

---

## 10. Reproducible evaluation

`make evaluate` regenerates every reported number from raw data:

```
evaluation/
├── split_manifest.json
├── baseline_forecasts.csv
├── statistical_models.csv
├── lightgbm_results.csv
├── lstm_results.csv
├── tft_results.csv
├── model_comparison.csv
├── forecast_ablation.csv
├── probabilistic_calibration.csv
├── nlp_baselines.csv
├── nlp_finetuning.csv
├── nlp_inference_benchmark.csv
├── multimodal_ablation.csv
├── inventory_simulation.csv
└── serving_benchmark.json
```

No performance number is typed by hand into the README, dashboards, or the web
app. They all read these artifacts or MLflow.

---

## 11. Compute budget and production contract

### 11.1 Development hardware (as of v0.1)

The development machine has an RTX 3050 Laptop GPU (**4 GB VRAM**) and 19 GB of
RAM. This hardware sets three constraints:

- **LightGBM on FULL:** use a training window that starts no earlier than `d_1000`,
  downcast dtypes (float32/int16), and categoricals. If memory still binds,
  train per store or per category and record that in the config.
- **F4/F5:** PANEL first. Use mixed precision and keep batch sizes within
  4 GB.
- **N4:** QLoRA on a model with ≤ 1.5B parameters. Larger models need cloud
  GPUs. Their cost is recorded and they are reported separately.

### 11.2 Promotion gates (champion/challenger)

**Forecast model:** a challenger becomes champion only if **all** of these hold:

1. Fold-averaged WRMSSE < champion's, and the paired CI of the difference lies entirely
   below 0 (an improvement, not a tie).
2. Test WRMSSE ≤ champion test WRMSSE × 1.01.
3. No category or store gets more than 3% worse in WRMSSE.
4. |Bias| ≤ 5% overall. Coverage of P90 lies within the §5.5 band.
5. Data validation passed for the training snapshot, and the inference
   schema contract test passes.

**NLP model:** gold-test macro-F1 ≥ champion − 0.01, structured-output validity
≥ 99.5%, and p95 latency within the SLO.

### 11.3 Serving SLOs (warm container, 1 vCPU / 2 GB unless stated)

| Endpoint | Metric | Target |
|----------|--------|--------|
| `POST /v1/forecast` (one series, precomputed features) | p50 / p95 / p99 | ≤ 30 / 100 / 250 ms |
| `POST /v1/inventory/recommend` | p95 | ≤ 150 ms |
| `POST /v1/events/extract` (N3, CPU) | p95 | ≤ 300 ms |
| `POST /v1/events/extract` (N4, GPU if deployed) | p95 | ≤ 1.5 s |
| Batch forecast, FULL × 28 days | wall-clock | ≤ 15 min on 4 vCPU |
| All endpoints | 5xx rate | < 0.1% |

### 11.4 Monitoring alert thresholds (initial)

| Signal | Warn | Alert |
|--------|------|-------|
| Feature PSI (per feature) | > 0.1 | > 0.25 |
| WAPE vs backtest (per category, weekly) | +10% rel. | +25% rel. |
| Bias (per category, weekly) | \|bias\| > 5% | > 10% |
| P90 coverage (rolling 28d) | outside [0.85, 0.95] | outside [0.80, 0.97] |
| Missing price/sales rows vs expected | > 0.5% | > 2% |
| NLP predicted-class distribution (JS divergence vs train) | > 0.1 | > 0.2 |
| NLP schema failures | > 0.5% | > 2% |
| p95 latency | > SLO | > 2× SLO |

---

## 12. Change Log

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 0.1 | 2026-10-02 | Initial contract | — |
| 0.5 | 2026-11-10 | Adds one forecast challenger, `lgbm_bc`: the champion LightGBM configuration with a per-category mean correction Σy / Σŷ, learned on each fold's out-of-sample calibration window (the 28 days before the origin, never the forecast window), clipped to 1 ± 0.10 and applied to the point forecast only; quantiles keep their empirical map. It is run once on FULL (5 folds + test) and judged by the unchanged §11.2 gates. `bias_clip` and the calibration window are fixed here and are not tuned after the result. If any gate fails, the champion stays and the result is reported. **Disclosure:** this challenger exists because `lgbm` failed gate 4 on the test window (bias −5.03%). The test-bias check is therefore not blind for `lgbm_bc`, so its fold-level backtest bias is reported alongside the gate | Under-forecasting is a systematic mean shift (backtest bias −4.1%, test −5.0%), not a quantile effect: the point forecast is a Tweedie mean. A correction learned only from pre-origin data fixes it without loosening the gate |
| 0.4 | 2026-10-03 | Scope and mapping decisions made while modelling: (a) LSTM/TFT run on PANEL only, on CPU, at 10 epochs × 40k windows; (b) every model trains only on the series of the population it is evaluated on; (c) multimodal arm D = LightGBM with sales, calendar, SNAP, price, hierarchy and past weather, and E/F add only the text family to D; (d) text signals apply to FOODS series in states the recall names or nationwide; (e) N4 (generative LoRA) needs a GPU and is reported as not run otherwise; (f) human gold labels are imported from the review tool, and items marked unsure are kept and reported | (a) Codespaces has 4 CPU cores and no GPU; (b) like-for-like comparisons; (c) hierarchy ids are part of the champion, so the text comparison isolates the text family; (d) the only defensible mapping for anonymized items (§2.4); (e) §11.1 compute; (f) transparency |
| 0.3 | 2026-10-03 | §5.5 quantile acceptance uses the discrete calibration bracket P(y<q) ≤ τ ≤ P(y≤q); WSPL is computed at the bottom level | The v0.1 band on P(y≤q) cannot be met by a correct forecast of count data with many zeros. Quantiles are produced only for bottom-level series |
| 0.2 | 2026-10-03 | §2.1 suspicious zero runs use the trailing share of zero days, P = p0^L < 0.001, not Poisson. Only 14–55-day runs are suspected stockouts; 56+-day runs are reported as dormant periods | On M5 the v0.1 rule flagged 14.6% of active series-days, 65% of them in runs of 90+ days. Those are regime changes, not replenishment stockouts. Both tests' shares are reported in `evaluation/eda/eda_overview.json` |
