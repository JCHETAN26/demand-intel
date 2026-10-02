"""Command-line entry point: `uv run demand-intel --help`."""

import json

import typer

from demand_intel import SPEC_VERSION, __version__
from demand_intel.config import settings
from demand_intel.forecasting.backtesting.splits import split_manifest

app = typer.Typer(no_args_is_help=True, help="Demand Intelligence pipelines and evaluation.")


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


if __name__ == "__main__":
    app()
