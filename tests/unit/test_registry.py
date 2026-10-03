import json
from pathlib import Path

import numpy as np
import pandas as pd

from demand_intel.mlops.registry.gates import Gate, GateReport, evaluate_gates
from demand_intel.mlops.registry.registry import load_registry, record_decision, tracking_uri


def _results(model: str, level: float, bias: float) -> list[dict[str, object]]:
    rows = []
    for fold in ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "test"):
        rows.append(
            {
                "model": model,
                "population": "panel",
                "fold": fold,
                "window": 28,
                "slice": "all",
                "value": "all",
                "wrmsse": level,
                "wrmsse_l12": level,
                "bias": bias,
            }
        )
        for cat in ("FOODS", "HOBBIES"):
            rows.append(
                {
                    "model": model,
                    "population": "panel",
                    "fold": fold,
                    "window": 28,
                    "slice": "cat_id",
                    "value": cat,
                    "wrmsse_l12": level,
                }
            )
    return rows


def _npz(d: Path, model: str, scale: float) -> None:
    rng = np.random.default_rng(1)
    for fold in ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5"):
        p = d / model / "panel"
        p.mkdir(parents=True, exist_ok=True)
        base = rng.uniform(0.5, 1.5, 300)
        np.savez(
            p / f"{fold}.npz", rows=np.arange(300), rmsse=base * scale, weight=np.full(300, 1 / 300)
        )


def test_gates_promote_only_a_clear_improvement(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"checks": [{"name": "x", "severity": "error", "passed": True}]})
    )
    res = pd.DataFrame(
        _results("new", 0.8, -0.02) + _results("old", 1.0, 0.0) + _results("biased", 0.8, -0.2)
    )
    _npz(tmp_path, "new", 0.8)
    _npz(tmp_path, "old", 1.0)
    _npz(tmp_path, "biased", 0.8)
    good = evaluate_gates(res, tmp_path, "new", "old", "panel", manifest)
    assert good.promote, good.to_dict()
    bad = evaluate_gates(res, tmp_path, "biased", "old", "panel", manifest)
    assert (
        not bad.promote
        and not next(g for g in bad.gates if g.name == "bias_and_calibration").passed
    )
    worse = evaluate_gates(res, tmp_path, "old", "new", "panel", manifest)
    assert not worse.promote


def test_registry_records_decisions(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    assert load_registry(path)["tasks"]["forecast"]["champion"] == "seasonal_naive"
    rep = GateReport("lgbm", "seasonal_naive", "full", [Gate("improvement", True, "ok")])
    reg = record_decision(path, "forecast", rep)
    assert reg["tasks"]["forecast"]["champion"] == "lgbm"
    rep2 = GateReport("lstm", "lgbm", "panel", [Gate("improvement", False, "tie")])
    reg = record_decision(path, "forecast", rep2)
    assert reg["tasks"]["forecast"]["champion"] == "lgbm" and reg["tasks"]["forecast"][
        "challengers"
    ] == ["lstm"]
    assert len(reg["decisions"]) == 2


def test_tracking_falls_back_to_sqlite(tmp_path: Path) -> None:
    uri = tracking_uri("http://localhost:1", tmp_path)
    assert uri.startswith("sqlite:///")
