"""Forecasting artifacts for `make evaluate` (PROJECT_SPEC.md §10).

Everything here is derived from `evaluation/backtest/*.csv` and the saved forecasts; no
number is typed by hand. The planned comparisons are fixed in `PLANNED_TESTS` (spec §5.6,
§6.1, §6.2), so a comparison cannot be added or dropped depending on its result.
"""

from pathlib import Path

import pandas as pd

from demand_intel.forecasting.evaluation.compare import (
    available_models,
    load_results,
    paired_test,
    summary_table,
)
from demand_intel.forecasting.evaluation.metrics import QUANTILES

BASELINES = ("naive", "seasonal_naive", "moving_average_28")
STATISTICAL = ("ses_seasonal", "croston", "tsb", "stat_class_selected", "ets", "arima")
ABLATION = (
    "lgbm_abl1_sales",
    "lgbm_abl2_calendar",
    "lgbm_abl3_snap",
    "lgbm_abl4_price",
    "lgbm_abl5_hierarchy",
    "lgbm_abl6_weather",
)
# (model A, model B, population): A is the more complex model; a negative difference
# with a CI below 0 means A improves on B.
PLANNED_TESTS: tuple[tuple[str, str, str], ...] = (
    ("seasonal_naive", "naive", "full"),
    ("lgbm", "seasonal_naive", "full"),
    ("lgbm", "stat_class_selected", "full"),
    ("lgbm", "seasonal_naive", "panel"),
    ("lgbm", "ets", "panel"),
    ("lgbm", "arima", "panel"),
    ("ets", "seasonal_naive", "panel"),
    ("lstm", "lgbm", "panel"),
    ("tft", "lgbm", "panel"),
    ("tft", "lstm", "panel"),
    *((ABLATION[i + 1], ABLATION[i], "panel") for i in range(len(ABLATION) - 1)),
    ("lgbm_weather_oracle", "lgbm_abl6_weather", "panel"),
)


def calibration_table(results: pd.DataFrame) -> pd.DataFrame:
    """Discrete-calibration bracket per model × fold × slice (spec §5.5, v0.3)."""
    sel = results[(results["window"] == 28) & results["slice"].isin(["all", "cat_id"])]
    sel = sel[sel["cov_q50"].notna()] if "cov_q50" in sel.columns else sel.iloc[0:0]
    rows = []
    tol = 0.03
    for r in sel.itertuples():
        row = {
            "model": r.model,
            "population": r.population,
            "fold": r.fold,
            "slice": r.slice,
            "value": r.value,
            "wspl": getattr(r, "wspl", None),
        }
        ok_all = True
        for tau in QUANTILES:
            k = f"{round(tau * 100):02d}"
            lt, le = getattr(r, f"cov_lt_q{k}"), getattr(r, f"cov_q{k}")
            ok = (lt - tol) <= tau <= (le + tol)
            row[f"lt_q{k}"], row[f"le_q{k}"], row[f"ok_q{k}"] = lt, le, ok
            if tau in (0.1, 0.9):
                ok_all &= ok
        row["calibrated_p10_p90"] = ok_all
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    abl = summary[summary["model"].isin(ABLATION) & (summary["population"] == "panel")].copy()
    abl["step"] = abl["model"].map({m: i + 1 for i, m in enumerate(ABLATION)})
    abl = abl.sort_values("step")
    for m in ("backtest_wrmsse", "backtest_wape", "backtest_bias", "test_wrmsse"):
        abl[f"delta_{m}"] = abl[m].diff()
    return abl


def build_forecast_artifacts(eval_dir: Path, forecast_dir: Path) -> dict[str, Path]:
    results = load_results(eval_dir / "backtest")
    if results.empty:
        return {}
    summary = summary_table(results)
    written: dict[str, Path] = {}

    def write(name: str, df: pd.DataFrame) -> None:
        path = eval_dir / name
        df.to_csv(path, index=False)
        written[name] = path

    write("model_comparison.csv", summary)
    write("baseline_forecasts.csv", summary[summary["model"].isin(BASELINES)])
    write("statistical_models.csv", summary[summary["model"].isin(STATISTICAL)])
    for name, models in (
        ("lightgbm_results.csv", ("lgbm",)),
        ("lstm_results.csv", ("lstm",)),
        ("tft_results.csv", ("tft",)),
    ):
        detail = results[results["model"].isin(models)]
        if len(detail):
            write(name, detail)
    write("forecast_ablation.csv", ablation_table(summary))
    write("probabilistic_calibration.csv", calibration_table(results))

    tests = []
    for a, b, pop in PLANNED_TESTS:
        have = set(available_models(forecast_dir, pop)) if (forecast_dir).exists() else set()
        if {a, b} <= have:
            tests.append(paired_test(forecast_dir, a, b, pop).as_row())
        else:
            tests.append({"model_a": a, "model_b": b, "population": pop, "status": "not run"})
    write("paired_tests.csv", pd.DataFrame(tests))
    return written
