"""Promotion gates for the forecasting champion (PROJECT_SPEC.md §11.2).

A challenger replaces the champion only if every gate passes:

1. improvement:   fold-averaged level-12 WRMSSE difference has a paired-bootstrap CI that
                  lies entirely below 0 (an improvement, not a tie)
2. test:          test WRMSSE <= champion test WRMSSE × 1.01
3. no_regression: no category or store gets more than 3% worse (backtest level-12 WRMSSE)
4. bias_and_calibration: |test bias| <= 5% and, for probabilistic models, P10 and P90
                  calibrated by the discrete bracket (spec §5.5) on the backtest folds
5. data:          the training data passed validation (M5 ingestion manifest has no failed
                  error-level check)

Gate 4's calibration half is skipped (reported as such) for point-only challengers;
promotion of a point-only model therefore needs the separate quantile service to stay.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from demand_intel.forecasting.evaluation.compare import BACKTEST, paired_test
from demand_intel.forecasting.evaluation.report import calibration_table

TEST_TOLERANCE = 1.01
SLICE_TOLERANCE = 1.03
MAX_ABS_BIAS = 0.05


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    challenger: str
    champion: str
    population: str
    gates: list[Gate] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger": self.challenger,
            "champion": self.champion,
            "population": self.population,
            "promote": self.promote,
            "gates": [g.__dict__ for g in self.gates],
        }


def _metric(
    results: pd.DataFrame,
    model: str,
    pop: str,
    fold_set: tuple[str, ...],
    col: str,
    slice_: str = "all",
) -> pd.Series:
    sel = results[
        (results["model"] == model)
        & (results["population"] == pop)
        & (results["window"] == 28)
        & (results["slice"] == slice_)
        & results["fold"].isin(fold_set)
    ]
    return sel.groupby("value")[col].mean()


def evaluate_gates(
    results: pd.DataFrame,
    forecast_dir: Path,
    challenger: str,
    champion: str,
    population: str,
    ingestion_manifest: Path,
) -> GateReport:
    rep = GateReport(challenger, champion, population)

    pt = paired_test(forecast_dir, challenger, champion, population, BACKTEST)
    rep.gates.append(
        Gate(
            "improvement",
            pt.improvement,
            f"Δ level-12 WRMSSE {pt.diff:+.4f} (95% CI {pt.ci_low:+.4f} to {pt.ci_high:+.4f}), "
            f"wins {pt.fold_wins}/{len(pt.folds)} folds, Wilcoxon p={pt.wilcoxon_p:.2g}",
        )
    )

    t_ch = _metric(results, challenger, population, ("test",), "wrmsse").get("all")
    t_cp = _metric(results, champion, population, ("test",), "wrmsse").get("all")
    ok = t_ch is not None and t_cp is not None and t_ch <= t_cp * TEST_TOLERANCE
    rep.gates.append(
        Gate("test", bool(ok), f"test WRMSSE {t_ch} vs champion {t_cp} × {TEST_TOLERANCE}")
    )

    worst, worst_key = 0.0, ""
    for slice_ in ("cat_id", "store_id"):
        a = _metric(results, challenger, population, BACKTEST, "wrmsse_l12", slice_)
        b = _metric(results, champion, population, BACKTEST, "wrmsse_l12", slice_)
        ratio = (a / b).dropna()
        if len(ratio) and ratio.max() > worst:
            worst, worst_key = float(ratio.max()), f"{slice_}={ratio.idxmax()}"
    rep.gates.append(
        Gate(
            "no_regression",
            worst <= SLICE_TOLERANCE,
            f"worst slice ratio {worst:.3f} ({worst_key}); limit {SLICE_TOLERANCE}",
        )
    )

    bias = _metric(results, challenger, population, ("test",), "bias").get("all")
    bias_ok = bias is not None and abs(bias) <= MAX_ABS_BIAS
    cal = calibration_table(
        results[(results["model"] == challenger) & (results["population"] == population)]
    )
    if len(cal):
        cal = cal[(cal["slice"] == "all") & cal["fold"].isin(BACKTEST)]
    if len(cal):
        cal_ok = bool(cal["calibrated_p10_p90"].all())
        cal_txt = f"P10/P90 calibrated on {int(cal['calibrated_p10_p90'].sum())}/{len(cal)} folds"
    else:
        cal_ok, cal_txt = True, "point-only model: calibration not applicable"
    rep.gates.append(
        Gate(
            "bias_and_calibration",
            bool(bias_ok and cal_ok),
            f"test bias {bias:+.4f} (limit ±{MAX_ABS_BIAS}); {cal_txt}"
            if bias is not None
            else "no bias",
        )
    )

    manifest = json.loads(ingestion_manifest.read_text()) if ingestion_manifest.exists() else {}
    failed = [
        c["name"]
        for c in manifest.get("checks", [])
        if c["severity"] == "error" and not c["passed"]
    ]
    rep.gates.append(
        Gate(
            "data",
            bool(manifest) and not failed,
            "ingestion manifest: all error-level checks passed"
            if manifest and not failed
            else f"failed checks: {failed or 'manifest missing'}",
        )
    )
    return rep
