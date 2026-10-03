import json
from pathlib import Path

import pandas as pd

from demand_intel.acceptance import answer_all, render


def test_everything_pending_without_artifacts(tmp_path: Path) -> None:
    answers = answer_all(tmp_path)
    assert len(answers) == 10
    assert {a["verdict"] for a in answers} == {"pending"}
    assert "| Question | Verdict | Evidence |" in render(answers)


def test_verdicts_follow_the_evidence(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "model_a": "lgbm",
                "model_b": "seasonal_naive",
                "population": "full",
                "diff": -0.1,
                "ci_low": -0.12,
                "ci_high": -0.08,
                "fold_wins": 5,
                "improvement": True,
            },
            {
                "model_a": "lstm",
                "model_b": "lgbm",
                "population": "panel",
                "diff": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
                "fold_wins": 0,
                "improvement": False,
            },
        ]
    ).to_csv(tmp_path / "paired_tests.csv", index=False)
    (tmp_path / "serving_benchmark.json").write_text(
        json.dumps(
            {
                "hardware": {"cores": 4},
                "configs": {"b": {"/v1/forecast": {"p95_ms": 12.0, "rps": 900}}},
            }
        )
    )
    by_q = {a["question"].split(":")[0]: a["verdict"] for a in answer_all(tmp_path)}
    assert by_q["Forecasting"] == "yes"
    assert by_q["Deep learning"] == "no"
    assert by_q["Production"] == "yes"
