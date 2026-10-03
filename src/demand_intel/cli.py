"""Command-line entry point: `uv run demand-intel --help`."""

import json

import typer

from demand_intel import SPEC_VERSION, __version__
from demand_intel.config import REPO_ROOT, settings
from demand_intel.forecasting.backtesting.splits import split_manifest

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
