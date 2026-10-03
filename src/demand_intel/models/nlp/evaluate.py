"""NLP metrics (PROJECT_SPEC.md §7.4): macro-F1 (primary), per-class precision/recall,
confusion matrix, calibration (ECE of the top-class confidence), accuracy.

Event-type metrics name their truth source: `gold` (human-validated, the only valid test
truth) or `weak` (agreement with the labelling rules, reported until gold labels exist and
never used for model selection or promotion).
"""

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from demand_intel import SEED


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(total)


@dataclass
class TaskScores:
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    accuracy: float
    ece: float
    per_class: pd.DataFrame
    confusion: pd.DataFrame


def score(
    probs: np.ndarray, truth: list[str], classes: tuple[str, ...], n_boot: int = 1000
) -> TaskScores:
    y = np.array([classes.index(t) for t in truth])
    pred = probs.argmax(axis=1)
    labels = sorted(set(y.tolist()))
    macro = float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), zero_division=0
    )
    per_class = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=list(range(len(classes)))), index=classes, columns=classes
    )
    return TaskScores(
        macro_f1=macro,
        macro_f1_ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        accuracy=float(accuracy_score(y, pred)),
        ece=ece(probs, y),
        per_class=per_class,
        confusion=cm,
    )


def summary_row(
    model: str, task: str, truth_source: str, split: str, s: TaskScores, **extra: Any
) -> dict[str, Any]:
    return {
        "model": model,
        "task": task,
        "truth": truth_source,
        "split": split,
        "macro_f1": s.macro_f1,
        "macro_f1_ci_low": s.macro_f1_ci[0],
        "macro_f1_ci_high": s.macro_f1_ci[1],
        "accuracy": s.accuracy,
        "ece": s.ece,
        **extra,
    }
