from pathlib import Path

import numpy as np
import pandas as pd

from demand_intel.forecasting.evaluation.compare import paired_test, summary_table
from demand_intel.forecasting.evaluation.report import ablation_table, calibration_table


def _write(d: Path, model: str, fold: str, rmsse: np.ndarray) -> None:
    p = d / model / "panel"
    p.mkdir(parents=True, exist_ok=True)
    n = len(rmsse)
    np.savez(p / f"{fold}.npz", rows=np.arange(n), rmsse=rmsse, weight=np.full(n, 1 / n))


def test_paired_test_detects_consistent_improvement(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    folds = ("fold_1", "fold_2", "fold_3")
    for f in folds:
        base = rng.uniform(0.5, 1.5, 400)
        _write(tmp_path, "good", f, base * 0.9)
        _write(tmp_path, "bad", f, base)
        _write(tmp_path, "same", f, base + rng.normal(0, 0.01, 400))
    better = paired_test(tmp_path, "good", "bad", "panel", folds, n_boot=300)
    assert better.improvement and better.fold_wins == 3 and better.wilcoxon_p < 0.01
    tie = paired_test(tmp_path, "same", "bad", "panel", folds, n_boot=300)
    assert not tie.improvement or abs(tie.diff) < 0.01


def test_summary_and_ablation_tables() -> None:
    rows = []
    for model, level in (("lgbm_abl1_sales", 1.0), ("lgbm_abl2_calendar", 0.9)):
        for fold in ("fold_1", "fold_2", "test"):
            rows.append(
                {
                    "model": model,
                    "population": "panel",
                    "fold": fold,
                    "slice": "all",
                    "value": "all",
                    "window": 28,
                    "wrmsse": level,
                    "wape": level,
                    "bias": 0.0,
                    "seconds": 1.0,
                }
            )
    summ = summary_table(pd.DataFrame(rows))
    abl = ablation_table(summ)
    assert abl["delta_backtest_wrmsse"].iloc[1] == -0.1 or np.isclose(
        abl["delta_backtest_wrmsse"].iloc[1], -0.1
    )


def test_calibration_bracket() -> None:
    r = {
        "model": "m",
        "population": "panel",
        "fold": "test",
        "slice": "all",
        "value": "all",
        "window": 28,
        "wspl": 0.3,
    }
    for k, (lt, le) in {
        "05": (0, 0.5),
        "10": (0, 0.5),
        "25": (0.1, 0.6),
        "50": (0.2, 0.7),
        "75": (0.5, 0.8),
        "90": (0.8, 0.95),
        "95": (0.9, 0.97),
    }.items():
        r[f"cov_lt_q{k}"], r[f"cov_q{k}"] = lt, le
    cal = calibration_table(pd.DataFrame([r]))
    assert bool(cal["calibrated_p10_p90"].iloc[0])
    r["cov_q90"] = 0.7  # P(y <= q90) far below 0.9: under-covers
    assert not bool(calibration_table(pd.DataFrame([r]))["calibrated_p10_p90"].iloc[0])
