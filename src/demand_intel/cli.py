"""Command-line entry point: `uv run demand-intel --help`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

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
        )
        head = res[(res["slice"] == "all") & (res["window"] == 28)]
        for r in head.itertuples():
            typer.echo(
                f"{name:22} {population:5} {r.fold:7} WRMSSE {r.wrmsse:.4f}  WAPE {r.wape:.4f}  "
                f"bias {r.bias:+.4f}  {r.seconds:.1f}s"
            )


@app.command()
def inventory(
    model: str = typer.Option(..., help="Forecast model whose backtest forecasts drive P2/P3."),
    population: str = typer.Option("panel", help="Series to simulate (spec §8.1: panel)."),
) -> None:
    """Historical inventory replay of policies P0-P3 and the oracle (spec §8)."""
    from demand_intel.data import load_m5
    from demand_intel.inventory.simulation.replay import run_replay

    m5 = load_m5(settings.validated_dir / "m5")
    rows = _population_rows(population, m5.series)
    metrics, diffs, calibration = run_replay(m5, rows, model, settings.data_dir / "forecasts")
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
