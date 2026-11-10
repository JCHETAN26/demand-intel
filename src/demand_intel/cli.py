"""Command-line entry point: `uv run demand-intel --help`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pandas as pd
import typer

from demand_intel import SPEC_VERSION, __version__
from demand_intel.config import REPO_ROOT, settings
from demand_intel.forecasting.backtesting.splits import split_manifest

if TYPE_CHECKING:
    import numpy as np

app = typer.Typer(no_args_is_help=True, help="Demand Intelligence pipelines and evaluation.")
ingest_app = typer.Typer(no_args_is_help=True, help="Download, validate and store source data.")
app.add_typer(ingest_app, name="ingest")


@app.command()
def version() -> None:
    """Print package and spec versions."""
    typer.echo(f"demand-intel {__version__} (spec v{SPEC_VERSION})")


@app.command()
def splits() -> None:
    """Write the backtest/test split manifest (PROJECT_SPEC.md §5.1)."""
    path = settings.evaluation_dir / "split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_manifest(), indent=2) + "\n")
    typer.echo(f"wrote {path}")


@app.command()
def eda(panel_size: int = typer.Option(3000, help="PANEL sample size (spec §5.2).")) -> None:
    """Data-quality checks, exploratory analysis, hypotheses and the PANEL sample."""
    from demand_intel import SEED
    from demand_intel.analysis.eda import render_report, run_eda, write_eda
    from demand_intel.analysis.profile import sample_panel
    from demand_intel.data import load_m5

    m5 = load_m5(settings.validated_dir / "m5")
    result = run_eda(m5)
    eda_dir = settings.evaluation_dir / "eda"
    write_eda(result, eda_dir, settings.validated_dir / "m5")
    report = REPO_ROOT / "docs" / "eda_report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(render_report(eda_dir))

    panel = sample_panel(result.tables["_profile"], panel_size, SEED)
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "demand_class"]
    panel_path = settings.evaluation_dir / "panel_series.csv"
    panel[[*cols, "volume_tercile"]].to_csv(panel_path, index=False)
    typer.echo(f"eda: {eda_dir}, report {report}, panel {len(panel)} series")


def _population_rows(population: str, series: pd.DataFrame) -> np.ndarray:
    import numpy as np
    import pandas as pd

    if population == "full":
        return np.arange(len(series))
    panel_ids = pd.read_csv(settings.evaluation_dir / "panel_series.csv")["id"]
    pos = pd.Series(np.arange(len(series)), index=series["id"])
    return np.sort(pos.loc[panel_ids].to_numpy())


@app.command()
def backtest(
    models: Annotated[list[str], typer.Argument(help="Model names (see models/registry.py).")],
    population: str = typer.Option("full", help="full | panel (spec §5.2)."),
    folds: str = typer.Option("all", help="backtest | test | all."),
) -> None:
    """Rolling-origin backtest of one or more models; writes evaluation/backtest/."""
    import pandas as pd

    from demand_intel.data import load_m5
    from demand_intel.forecasting.backtesting.harness import (
        SLICES,
        TruthCache,
        folds_for,
        run_backtest,
    )
    from demand_intel.models.registry import make

    m5 = load_m5(settings.validated_dir / "m5")
    profile = pd.read_parquet(settings.validated_dir / "m5" / "series_profile.parquet")
    runs = pd.read_parquet(settings.validated_dir / "m5" / "suspicious_zero_runs.parquet")
    rows = _population_rows(population, m5.series)
    attrs = profile.iloc[rows][list(SLICES)]
    weather_path = settings.validated_dir / "weather" / "weather_state_daily.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else None
    extras = {}
    signals_path = settings.data_dir / "nlp" / "text_signals.npz"
    if signals_path.exists():
        from demand_intel.pipelines.features.text_signals import TextSignals

        extras["text"] = TextSignals.load(signals_path)
    cache = TruthCache(m5)
    for name in models:
        res = run_backtest(
            make(name),
            m5,
            rows,
            population,  # type: ignore[arg-type]
            folds_for(folds),  # type: ignore[arg-type]
            attrs,
            settings.evaluation_dir / "backtest",
            settings.data_dir / "forecasts",
            stockout_runs=runs,
            weather=weather,
            truth_cache=cache,
            extras=extras,
        )
        head = res[(res["slice"] == "all") & (res["window"] == 28)]
        for r in head.itertuples():
            typer.echo(
                f"{name:22} {population:5} {r.fold:7} WRMSSE {r.wrmsse:.4f}  WAPE {r.wape:.4f}  "
                f"bias {r.bias:+.4f}  {r.seconds:.1f}s"
            )


@app.command("forecast-batch")
def forecast_batch(
    model: str = typer.Option(
        "", help="LightGBM variant to publish instead of the champion (recorded as a fallback)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail instead of falling back when the champion is not servable."
    ),
) -> None:
    """Retrain the champion on all history and write the serving bundle (data/serving).

    Only LightGBM models can be published. If the registry champion is not one, the
    bundle falls back to `lgbm` and says so: a warning here, and `registry_champion`,
    `fallback` and `fallback_reason` in summary.json and in the API's health and summary.
    """
    import logging

    from demand_intel.data import load_m5
    from demand_intel.mlops.registry.registry import load_registry
    from demand_intel.models.lgbm.model import LightGBMForecaster
    from demand_intel.models.registry import make
    from demand_intel.pipelines.inference.serving import (
        ServingSelectionError,
        build_bundle,
        choose_serving_model,
    )

    champion = load_registry(settings.evaluation_dir / "registry.json")["tasks"]["forecast"][
        "champion"
    ]
    try:
        choice = choose_serving_model(
            champion,
            servable=lambda n: isinstance(make(n), LightGBMForecaster),
            requested=model,
            allow_fallback=not strict,
        )
    except ServingSelectionError as e:
        typer.echo(f"forecast-batch: {e}", err=True)
        raise typer.Exit(2) from e
    if choice.fallback:
        msg = f"SERVING FALLBACK: {choice.reason}"
        logging.getLogger("demand_intel.serving").warning(msg)
        typer.secho(msg, err=True, fg=typer.colors.YELLOW)
    forecaster = make(choice.served)

    m5 = load_m5(settings.validated_dir / "m5")
    profile = pd.read_parquet(settings.validated_dir / "m5" / "series_profile.parquet")
    info = build_bundle(
        m5,
        settings.data_dir / "serving",
        settings.data_dir / "models" / "lgbm",
        settings.evaluation_dir,
        profile,
        forecaster=forecaster,  # type: ignore[arg-type]
        choice=choice,
    )
    served = info.model if not choice.fallback else f"{info.model}, FALLBACK from {champion}"
    typer.echo(f"bundle: {info.path} (origin d_{info.origin}, {info.series} series, {served})")
    _write_bundle_extras(info.path)


def _write_bundle_extras(bundle: Path) -> None:
    from demand_intel.pipelines.inference.serving import write_extras

    summary = json.loads((bundle / "summary.json").read_text())
    recalls_path = settings.validated_dir / "text" / "recalls.parquet"
    recalls = pd.read_parquet(recalls_path) if recalls_path.exists() else None
    preds = None
    champion = settings.data_dir / "nlp" / "champion_predictions.parquet"
    if champion.exists():
        preds = pd.read_parquet(champion)
    write_extras(bundle, settings.evaluation_dir, recalls, preds, summary["origin_date"])


@app.command("bundle-risk")
def bundle_risk() -> None:
    """Recompute the risk table of an existing bundle (no retraining)."""
    import numpy as np

    from demand_intel.data import load_m5
    from demand_intel.forecasting.evaluation.metrics import QCOLS
    from demand_intel.pipelines.inference.serving import (
        EXCESS_RATIO,
        RISK_DEFINITION,
        RISK_LEAD_TIME,
        calibrated_rho,
        compute_risk,
    )

    bundle = settings.data_dir / "serving"
    m5 = load_m5(settings.validated_dir / "m5", with_prices=False)
    fc = pd.read_parquet(bundle / "forecast.parquet").sort_values(["series", "d"])
    n = int(fc["series"].max()) + 1
    qarr = np.stack([fc[c].to_numpy() for c in QCOLS], axis=-1).reshape(n, -1, len(QCOLS))
    rho = calibrated_rho(settings.evaluation_dir)
    origin = m5.n_days
    risk = compute_risk(qarr, m5.sales[:, origin - 28 : origin], rho)
    risk.to_parquet(bundle / "risk.parquet", index=False)
    summary = json.loads((bundle / "summary.json").read_text())
    summary["risk"] = {
        "definition": RISK_DEFINITION.format(k=RISK_LEAD_TIME + 1, r=EXCESS_RATIO, rho=rho),
        "rho": rho,
    }
    (bundle / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(
        f"stockout risk {int(risk['stockout_risk'].sum())}, excess {int(risk['excess_risk'].sum())}"
    )


@app.command("bundle-extras")
def bundle_extras() -> None:
    """Refresh the Model Lab payload and recent events in the serving bundle."""
    _write_bundle_extras(settings.data_dir / "serving")
    typer.echo("bundle extras written")


@app.command()
def promote(
    challenger: str = typer.Option(..., help="Forecast model proposed as champion."),
    population: str = typer.Option("full"),
) -> None:
    """Evaluate the §11.2 gates against the current champion; register if all pass."""
    from demand_intel.forecasting.evaluation.compare import load_results, summary_table
    from demand_intel.mlops.registry.gates import evaluate_gates
    from demand_intel.mlops.registry.registry import (
        load_registry,
        log_model_runs,
        record_decision,
        register_forecaster,
        tracking_uri,
    )

    reg_path = settings.evaluation_dir / "registry.json"
    champion = load_registry(reg_path)["tasks"]["forecast"]["champion"]
    results = load_results(settings.evaluation_dir / "backtest")
    report = evaluate_gates(
        results,
        settings.data_dir / "forecasts",
        challenger,
        champion,
        population,
        settings.validated_dir / "m5" / "_ingestion.json",
    )
    for g in report.gates:
        typer.echo(f"{'PASS' if g.passed else 'FAIL'} {g.name}: {g.detail}")
    uri = tracking_uri(settings.mlflow_tracking_uri, settings.data_dir / "mlflow")
    summary = summary_table(results)
    log_model_runs(uri, summary, settings.evaluation_dir / "backtest")
    version = None
    model_dir = settings.data_dir / "models" / "lgbm" / challenger / population / "test"
    if report.promote and model_dir.exists():
        row = summary[(summary["model"] == challenger) & (summary["population"] == population)]
        metrics = {k: float(row[k].iloc[0]) for k in ("test_wrmsse", "backtest_wrmsse") if k in row}
        version = register_forecaster(uri, "demand-forecaster", model_dir, metrics, "champion")
    record_decision(reg_path, "forecast", report, version)
    verdict = "PROMOTED" if report.promote else "NOT promoted"
    typer.echo(f"{challenger}: {verdict} (champion was {champion}); tracking {uri}")


@app.command()
def monitor(
    model: str = typer.Option("", help="Forecast model to monitor (default: registry champion)."),
) -> None:
    """Forecast, data-drift and NLP monitoring on the latest scored window (spec §11.4)."""
    import numpy as np

    from demand_intel.data import load_m5
    from demand_intel.forecasting.backtesting.harness import truncate
    from demand_intel.forecasting.backtesting.splits import BACKTEST_FOLDS, HORIZON, TEST_FOLD
    from demand_intel.mlops.monitoring.monitor import (
        MonitorReport,
        feature_drift,
        forecast_checks,
        nlp_checks,
    )
    from demand_intel.mlops.registry.registry import load_registry
    from demand_intel.models.nlp.labeling import weak_label
    from demand_intel.pipelines.features.build import DEFAULT_FAMILIES, build_features

    out = settings.evaluation_dir
    name = model or load_registry(out / "registry.json")["tasks"]["forecast"]["champion"]
    m5 = load_m5(settings.validated_dir / "m5")
    z = np.load(settings.data_dir / "forecasts" / name / "full" / "test.npz")
    rows = z["rows"]
    o = TEST_FOLD.origin
    y = m5.sales[rows, o : o + HORIZON].astype(float)
    q90 = z["q90"] if "q90" in z.files else None
    results = pd.read_csv(out / "backtest" / f"{name}__full.csv")
    folds = [f.name for f in BACKTEST_FOLDS]
    baseline = (
        results[(results["window"] == 28) & results["fold"].isin(folds)]
        .groupby(["slice", "value"])["wape"]
        .mean()
        .reset_index()
    )
    fc_rep, slices = forecast_checks(y, z["point"], q90, m5.series.iloc[rows], baseline)

    panel = _population_rows("panel", m5.series)
    data = truncate(m5, o)
    ref = build_features(data, panel, np.arange(o - 364 + 1, o + 1), 7, DEFAULT_FAMILIES).X
    cur = build_features(data, panel, np.arange(o + 1, o + 8), 7, DEFAULT_FAMILIES).X
    drift = feature_drift(ref, cur)
    data_rep = MonitorReport()
    from demand_intel.mlops.monitoring.monitor import Check

    for r in drift.itertuples():
        data_rep.checks.append(Check(f"psi:{r.feature}", float(r.psi), r.status))  # type: ignore[arg-type]

    nlp_summary = json.loads((out / "nlp_dataset.json").read_text())
    train_mix = nlp_summary["weak_label_distribution"]["train"]
    docs = pd.read_parquet(settings.data_dir / "nlp" / "dataset.parquet")
    batch = docs[docs["split"] == "test"]["reason_for_recall"].map(lambda t: weak_label(t).label)
    nlp_rep = nlp_checks({k: float(v) for k, v in train_mix.items()}, batch.tolist(), 0)

    report = {
        "model": name,
        "window": TEST_FOLD.name,
        "forecast": fc_rep.to_dict(),
        "data": data_rep.to_dict(),
        "nlp": nlp_rep.to_dict(),
    }
    (out / "monitoring_report.json").write_text(json.dumps(report, indent=2, default=float))
    slices.to_csv(out / "monitoring_forecast_slices.csv", index=False)
    drift.to_csv(out / "monitoring_feature_drift.csv", index=False)
    for part in ("forecast", "data", "nlp"):
        rep = report[part]
        flagged = [c["name"] for c in rep["checks"] if c["status"] != "ok"]  # type: ignore[index]
        typer.echo(f"{part}: {rep['status']} ({len(flagged)} flagged) {flagged[:6]}")  # type: ignore[index]


@app.command()
def report() -> None:
    """Assemble the forecasting artifacts and planned paired tests (spec §10)."""
    from demand_intel.forecasting.evaluation.report import build_forecast_artifacts

    written = build_forecast_artifacts(settings.evaluation_dir, settings.data_dir / "forecasts")
    for name in written:
        typer.echo(f"wrote evaluation/{name}")


@app.command()
def inventory(
    model: str = typer.Option(..., help="Forecast model whose backtest forecasts drive P2/P3."),
    population: str = typer.Option("panel", help="Series to simulate (spec §8.1: panel)."),
    forecast_population: str = typer.Option("full", help="Backtest whose forecasts to use."),
) -> None:
    """Historical inventory replay of policies P0-P3 and the oracle (spec §8)."""
    from demand_intel.data import load_m5
    from demand_intel.inventory.simulation.replay import run_replay

    m5 = load_m5(settings.validated_dir / "m5")
    rows = _population_rows(population, m5.series)
    metrics, diffs, calibration = run_replay(
        m5, rows, model, settings.data_dir / "forecasts", source=forecast_population
    )
    out = settings.evaluation_dir
    metrics.insert(0, "model", model)
    metrics.to_csv(out / "inventory_simulation.csv", index=False)
    diffs.insert(0, "model", model)
    diffs.to_csv(out / "inventory_policy_differences.csv", index=False)
    (out / "inventory_calibration.json").write_text(json.dumps(calibration, indent=2, default=str))
    totals = metrics.groupby(["scenario", "policy"])[["total_cost", "fill_rate"]].agg(
        {"total_cost": "sum", "fill_rate": "mean"}
    )
    typer.echo(totals.round(3).to_string())


@app.command()
def explain(
    model: str = typer.Option(
        "lgbm", help="LightGBM variant whose saved test-fold models to explain."
    ),
    sample: int = typer.Option(40_000, help="Rows sampled per bucket."),
) -> None:
    """SHAP importance (global, per category, per family) for the saved test-fold models."""
    import lightgbm as lgb
    import numpy as np

    from demand_intel import SEED
    from demand_intel.data import load_m5
    from demand_intel.forecasting.backtesting.harness import truncate
    from demand_intel.forecasting.backtesting.splits import TEST_FOLD
    from demand_intel.models.lgbm.explain import contributions, global_importance
    from demand_intel.pipelines.features.build import BUCKETS, build_features

    m5 = load_m5(settings.validated_dir / "m5")
    data = truncate(m5, TEST_FOLD.origin)
    model_dir = settings.data_dir / "models" / "lgbm" / model / "full" / TEST_FOLD.name
    meta = json.loads((model_dir / "meta.json").read_text())
    out = settings.evaluation_dir / "shap"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = np.arange(len(m5.series))
    frames_g, frames_c, frames_f = [], [], []
    for lo, hi in BUCKETS:
        booster = lgb.Booster(model_file=str(model_dir / f"bucket_{hi}.txt"))
        days = np.arange(TEST_FOLD.origin + lo, TEST_FOLD.origin + hi + 1)
        ff = build_features(data, rows, days, hi, meta["families"])
        pick = rng.choice(
            np.flatnonzero(ff.eligible), size=min(sample, int(ff.eligible.sum())), replace=False
        )
        c = contributions(booster, ff.X.iloc[pick].reset_index(drop=True))
        cats = m5.series["cat_id"].to_numpy()[ff.series_idx[pick]]
        g = global_importance(c).assign(bucket=hi)
        per_cat = global_importance(c, pd.Series(cats)).assign(bucket=hi)
        fam = c.by_family(units=True).abs().groupby(cats).mean().assign(bucket=hi)
        frames_g.append(g)
        frames_c.append(per_cat)
        frames_f.append(fam.reset_index(names="cat_id"))
    pd.concat(frames_g).to_csv(out / "shap_global.csv", index=False)
    pd.concat(frames_c).to_csv(out / "shap_by_category.csv", index=False)
    pd.concat(frames_f).to_csv(out / "shap_family_units_by_category.csv", index=False)
    typer.echo(
        pd.concat(frames_g).groupby("family")["mean_abs_shap"].sum().sort_values().to_string()
    )


@app.command("nlp-prepare")
def nlp_prepare(
    gold_size: int = typer.Option(600, help="Docs for human validation (spec §7.1)."),
) -> None:
    """Weak labels, leakage-safe splits and the stratified human-validation sample."""
    from demand_intel import SEED
    from demand_intel.models.nlp.dataset import gold_sample, make_splits
    from demand_intel.models.nlp.labeling import weak_label_frame

    docs = pd.read_parquet(settings.validated_dir / "text" / "recalls.parquet")
    docs = docs.merge(weak_label_frame(docs), on="doc_id")
    splits = make_splits(docs)
    out = settings.data_dir / "nlp"
    out.mkdir(parents=True, exist_ok=True)
    frame = splits.frame
    frame.to_parquet(out / "dataset.parquet", index=False)
    test = frame[frame["split"] == "test"]
    gold = gold_sample(test, "weak_label", gold_size, SEED)
    gold[
        [
            "doc_id",
            "report_date",
            "reason_for_recall",
            "product_description",
            "weak_label",
            "severity",
        ]
    ].to_parquet(out / "gold_candidates.parquet", index=False)
    summary = {
        "splits": splits.stats,
        "weak_label_distribution": {
            s: frame.loc[frame["split"] == s, "weak_label"].value_counts().to_dict()
            for s in ("train", "validation", "test")
        },
        "lf_conflict_rate": float(frame["lf_conflict"].mean()),
        "lf_coverage": float((frame["weak_label"] != "OTHER").mean()),
        "gold_candidates": len(gold),
    }
    (settings.evaluation_dir / "nlp_dataset.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary["splits"]))
    typer.echo(
        f"coverage {summary['lf_coverage']:.3f}, conflicts {summary['lf_conflict_rate']:.3f}"
    )


@app.command("nlp-train")
def nlp_train(
    models: str = typer.Option(
        "majority,rules,tfidf,embeddings",
        help="Comma-separated: majority, rules, tfidf, embeddings, encoder.",
    ),
    encoder_epochs: int = typer.Option(3),
) -> None:
    """Train NLP models on the training split; score validation/test (spec §7)."""
    import time

    from demand_intel.models.nlp.evaluate import score, summary_row
    from demand_intel.models.nlp.labeling import EVENT_TYPES
    from demand_intel.models.nlp.models import (
        SEVERITIES,
        EmbeddingModel,
        EncoderModel,
        MajorityModel,
        RulesModel,
        TfidfModel,
    )

    nlp_dir = settings.data_dir / "nlp"
    docs = pd.read_parquet(nlp_dir / "dataset.parquet")
    gold_path = nlp_dir / "gold_labels.parquet"
    gold = pd.read_parquet(gold_path).set_index("doc_id")["label"] if gold_path.exists() else None
    tr, va, te = (docs[docs["split"] == s] for s in ("train", "validation", "test"))
    factories: dict[str, Callable[[], Any]] = {
        "majority": MajorityModel,
        "rules": RulesModel,
        "tfidf": TfidfModel,
        "embeddings": EmbeddingModel,
        "encoder": lambda: EncoderModel(
            epochs=encoder_epochs, save_dir=settings.data_dir / "models" / "nlp" / "n3_encoder"
        ),
    }
    out_rows = []
    pred_dir = nlp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key in [m.strip() for m in models.split(",") if m.strip()]:
        model = factories[key]()
        t0 = time.perf_counter()
        args = (tr["text"].tolist(), tr["weak_label"].tolist(), tr["severity"].tolist())
        if key == "encoder":
            model.fit(
                *args, val=(va["text"].tolist(), va["weak_label"].tolist(), va["severity"].tolist())
            )
        else:
            model.fit(*args)
        fit_s = time.perf_counter() - t0
        for split, frame in (("validation", va), ("test", te)):
            pe, ps = model.predict_proba(frame["text"].tolist())
            pd.DataFrame(
                {
                    "doc_id": frame["doc_id"].to_numpy(),
                    "event_pred": [EVENT_TYPES[i] for i in pe.argmax(1)],
                    "event_conf": pe.max(1),
                    "severity_pred": [SEVERITIES[i] for i in ps.argmax(1)],
                    "severity_conf": ps.max(1),
                }
            ).to_parquet(pred_dir / f"{model.name}__{split}.parquet", index=False)
            sev = score(ps, frame["severity"].tolist(), SEVERITIES)
            out_rows.append(
                summary_row(model.name, "severity", "fda_class", split, sev, fit_seconds=fit_s)
            )
            weak = score(pe, frame["weak_label"].tolist(), EVENT_TYPES)
            out_rows.append(
                summary_row(model.name, "event_type", "weak", split, weak, fit_seconds=fit_s)
            )
            if split == "test" and gold is not None:
                g = frame["doc_id"].map(gold)
                has = g.notna().to_numpy()
                if has.sum():
                    gs = score(pe[has], g[has].tolist(), EVENT_TYPES)
                    out_rows.append(
                        summary_row(
                            model.name,
                            "event_type",
                            "gold",
                            split,
                            gs,
                            fit_seconds=fit_s,
                            n=int(has.sum()),
                        )
                    )
                    gs.per_class.assign(model=model.name).to_csv(
                        settings.evaluation_dir / f"nlp_per_class__{model.name}.csv", index=False
                    )
                    gs.confusion.to_csv(
                        settings.evaluation_dir / f"nlp_confusion__{model.name}.csv"
                    )
        typer.echo(f"{model.name}: trained in {fit_s:.0f}s")
    new = pd.DataFrame(out_rows)
    path = settings.evaluation_dir / "nlp_results.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(new["model"].unique())], new], ignore_index=True)
    new.to_csv(path, index=False)
    show = new[new["split"] == "test"][["model", "task", "truth", "macro_f1", "accuracy", "ece"]]
    typer.echo(show.round(3).to_string(index=False))


@app.command("nlp-signals")
def nlp_signals(
    event_source: str = typer.Option("rules", help="Event types from: rules | tfidf | encoder."),
) -> None:
    """Build text signals (spec §4.4) for the multimodal experiment."""
    from demand_intel.data import load_m5
    from demand_intel.models.nlp.labeling import EVENT_TYPES, weak_label
    from demand_intel.models.nlp.models import EmbeddingModel, EncoderModel, TfidfModel
    from demand_intel.pipelines.features.text_signals import build_signals

    m5 = load_m5(settings.validated_dir / "m5", with_prices=False)
    docs = pd.read_parquet(settings.validated_dir / "text" / "recalls.parquet")
    docs = docs[docs["report_date"] <= pd.Timestamp("2016-06-19")].reset_index(drop=True)
    texts = docs["text"].tolist()
    if event_source == "rules":
        docs["event_type"] = docs["reason_for_recall"].map(lambda t: weak_label(t).label)
    else:
        nlp = pd.read_parquet(settings.data_dir / "nlp" / "dataset.parquet")
        train = nlp[nlp["split"] == "train"]
        if event_source == "tfidf":
            model: Any = TfidfModel()
            model.fit(
                train["text"].tolist(), train["weak_label"].tolist(), train["severity"].tolist()
            )
        else:
            model = EncoderModel.load(settings.data_dir / "models" / "nlp" / "n3_encoder")
        probs, _ = model.predict_proba(texts)
        docs["event_type"] = [EVENT_TYPES[i] for i in probs.argmax(1)]
    emb = EmbeddingModel()._embed(texts)
    sig = build_signals(docs, m5.calendar, emb, source=event_source)
    out = settings.data_dir / "nlp" / "text_signals.npz"
    sig.save(out)
    typer.echo(f"text signals ({event_source}) for {len(docs)} reports → {out}")


@app.command("nlp-benchmark")
def nlp_benchmark(
    threads: int = typer.Option(1, help="CPU threads per model (1 = one vCPU)."),
) -> None:
    """Accuracy × latency × throughput × size × cost for the NLP models (spec §7.4)."""
    import pickle

    from sklearn.metrics import f1_score

    from demand_intel.models.nlp.benchmark import Candidate, encoder_variants, measure
    from demand_intel.models.nlp.labeling import EVENT_TYPES
    from demand_intel.models.nlp.models import SEVERITIES, EmbeddingModel, TfidfModel

    nlp_dir = settings.data_dir / "nlp"
    docs = pd.read_parquet(nlp_dir / "dataset.parquet")
    tr, te = docs[docs["split"] == "train"], docs[docs["split"] == "test"]
    gold_path = nlp_dir / "gold_labels.parquet"
    gold = pd.read_parquet(gold_path).set_index("doc_id")["label"] if gold_path.exists() else None
    texts = te["text"].tolist()
    cands = []
    tf = TfidfModel()
    tf.fit(tr["text"].tolist(), tr["weak_label"].tolist(), tr["severity"].tolist())
    cands.append(Candidate("n1_tfidf_logreg", tf.predict_proba, len(pickle.dumps(tf)) / 1e6))
    emb = EmbeddingModel()
    emb.fit(tr["text"].tolist(), tr["weak_label"].tolist(), tr["severity"].tolist())
    cands.append(Candidate("n2_embeddings_logreg", emb.predict_proba, 90.9))  # MiniLM-L6 weights
    notes: list[str] = []
    enc_dir = settings.data_dir / "models" / "nlp" / "n3_encoder"
    if (enc_dir / "heads.pt").exists():
        variants, notes = encoder_variants(enc_dir, threads)
        cands += variants
    rows = []
    for c in cands:
        stats = measure(c, texts)
        pe, ps = c.predict(texts)
        sev = [SEVERITIES.index(s) for s in te["severity"]]
        stats["severity_macro_f1"] = float(f1_score(sev, ps.argmax(1), average="macro"))
        truth, src = (
            (te["doc_id"].map(gold), "gold") if gold is not None else (te["weak_label"], "weak")
        )
        has = truth.notna().to_numpy()
        y = [EVENT_TYPES.index(t) for t in truth[has]]
        stats["event_macro_f1"] = float(f1_score(y, pe[has].argmax(1), average="macro"))
        stats["event_truth"] = src
        stats["threads"] = threads
        rows.append(stats)
        typer.echo(
            f"{c.name}: p50 {stats['p50_ms']:.1f} ms, {stats['throughput_docs_per_s']:.0f} docs/s"
        )
    out = pd.DataFrame(rows)
    out.to_csv(settings.evaluation_dir / "nlp_inference_benchmark.csv", index=False)
    for n in notes:
        typer.echo(n)


@app.command("nlp-import-gold")
def nlp_import_gold(
    export_dir: Annotated[
        Path, typer.Option(help="Folder of exported label documents (one JSON per recall).")
    ],
) -> None:
    """Import human-validated labels from the review tool into data/nlp/gold_labels.parquet."""
    from demand_intel.models.nlp.labeling import EVENT_TYPES

    rows = []
    for f in sorted(export_dir.rglob("*.json")):
        doc = json.loads(f.read_text())
        body = doc.get("data", doc)
        if body.get("label") not in EVENT_TYPES:
            continue
        rows.append(
            {
                "doc_id": f.stem,
                "label": body["label"],
                "weak_label": body.get("weak_label"),
                "unsure": bool(body.get("unsure", False)),
                "note": body.get("note", ""),
                "labeled_at": body.get("labeled_at"),
            }
        )
    gold = pd.DataFrame(rows)
    out = settings.data_dir / "nlp" / "gold_labels.parquet"
    gold.to_parquet(out, index=False)
    agree = float((gold["label"] == gold["weak_label"]).mean()) if len(gold) else float("nan")
    typer.echo(
        f"{len(gold)} gold labels ({int(gold['unsure'].sum()) if len(gold) else 0} unsure); "
        f"rule labels agree with humans on {agree:.1%}"
    )


@app.command()
def acceptance() -> None:
    """Answer the ten acceptance questions from the evaluation artifacts (plan Phase 37)."""
    from demand_intel.acceptance import answer_all, render

    answers = answer_all(settings.evaluation_dir)
    (settings.evaluation_dir / "acceptance.json").write_text(json.dumps(answers, indent=2))
    doc = REPO_ROOT / "docs" / "acceptance.md"
    doc.write_text(render(answers))
    for a in answers:
        typer.echo(f"[{a['verdict']:7}] {a['question']}")


@ingest_app.command("m5")
def ingest_m5(
    download: bool = typer.Option(True, help="Fetch from Kaggle if raw files are missing."),
    ledger: bool = typer.Option(True, help="Record the run in Postgres meta.ingestion_runs."),
    force: bool = typer.Option(False, help="Re-download and re-validate even if unchanged."),
) -> None:
    """Download (if needed), validate and store M5 as Parquet."""
    from demand_intel.pipelines.ingestion import m5
    from demand_intel.pipelines.ingestion.download import download_m5

    raw = settings.raw_dir / "m5"
    if download:
        raw = download_m5(settings.raw_dir, force=force)
    try:
        report = m5.ingest_m5(
            raw, settings.validated_dir / "m5", settings.quarantine_dir / "m5", force=force
        )
    except m5.IngestionError as err:
        for c in err.failed:
            typer.secho(f"FAIL {c.name}: {c.detail}", fg="red", err=True)
        raise typer.Exit(1) from err

    typer.echo(f"m5: {report.status}")
    for f in report.files:
        typer.echo(
            f"  {f.source_file}: {f.loaded_count}/{f.row_count} loaded, {f.rejected_count} rejected"
        )
    for c in report.checks:
        if not c.passed:
            typer.secho(f"  WARN {c.name}: {c.detail}", fg="yellow")
    if ledger:
        from demand_intel.pipelines.ingestion.ledger import record

        typer.echo(f"  ledger: {record(settings.database_url, report)} new row(s)")


@ingest_app.command("text")
def ingest_text(force: bool = typer.Option(False, help="Refetch even if unchanged.")) -> None:
    """Download and validate the openFDA food recall corpus (spec §2.4)."""
    from demand_intel.pipelines.ingestion.text import build_recalls, download_recalls

    raw = download_recalls(settings.raw_dir, force=force)
    out = settings.validated_dir / "text" / "recalls.parquet"
    report = build_recalls(raw, out, settings.quarantine_dir / "text" / "recalls.parquet")
    typer.echo(
        f"text: {report.loaded}/{report.total} loaded, {report.rejected} rejected "
        f"(source updated {report.last_updated})"
    )
    for reason, n in report.reasons.items():
        if n:
            typer.echo(f"  {reason}: {n}")


@ingest_app.command("weather")
def ingest_weather(force: bool = typer.Option(False, help="Refetch even if unchanged.")) -> None:
    """Download Open-Meteo history and build the state × day weather table."""
    from demand_intel.pipelines.ingestion.weather import build_state_weather, download_weather

    locations = REPO_ROOT / "configs" / "weather_locations.yaml"
    raw = download_weather(settings.raw_dir, locations, force=force)
    out = settings.validated_dir / "weather" / "weather_state_daily.parquet"
    report = build_state_weather(raw, locations, out)
    typer.echo(f"weather: {report.rows} state-days -> {out}")
    typer.echo(
        f"  missing cells {report.missing_cells}, longest gap {report.max_gap_days} d, "
        f"out-of-range {report.out_of_range_cells}, checks {report.checks}"
    )


if __name__ == "__main__":
    app()
