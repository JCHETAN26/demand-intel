from typer.testing import CliRunner

from demand_intel import SPEC_VERSION
from demand_intel.cli import app


def test_cli_imports_and_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"spec v{SPEC_VERSION}" in result.output
    help_text = runner.invoke(app, ["--help"]).output
    for command in ("backtest", "eda", "ingest", "splits"):
        assert command in help_text
