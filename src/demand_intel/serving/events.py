"""Structured event extraction for `/v1/events/extract` (plan Phases 21-23, 30).

The NLP champion (see models/nlp) implements `EventModel`; until one is registered the
rule-based labeller serves, and every response names the model that produced it.
Geography is parsed deterministically from the text; direction is always "negative"
because every document in this corpus is a recall.
"""

from dataclasses import dataclass
from typing import Protocol

from demand_intel.models.nlp.labeling import weak_label
from demand_intel.pipelines.ingestion.text import parse_distribution

SEVERITY_BY_CLASS = {"Class I": "high", "Class II": "medium", "Class III": "low"}


@dataclass(frozen=True)
class Prediction:
    event_type: str
    confidence: float
    severity: str | None
    severity_confidence: float | None


class EventModel(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class RuleModel:
    """Fallback: the weak-supervision rules. Confidence 0.9 when exactly one rule fires,
    0.6 on conflicts, 0.3 when nothing fires."""

    name = "rules-v1"

    def predict(self, texts: list[str]) -> list[Prediction]:
        out = []
        for t in texts:
            w = weak_label(t)
            conf = 0.3 if not w.fired else (0.6 if w.conflict else 0.9)
            out.append(Prediction(w.label, conf, None, None))
        return out


@dataclass(frozen=True)
class ExtractedEvent:
    event_type: str
    confidence: float
    severity: str | None
    severity_confidence: float | None
    direction: str
    category: str
    states: list[str]
    nationwide: bool
    model: str


def extract(
    model: EventModel, texts: list[str], distribution: list[str] | None = None
) -> list[ExtractedEvent]:
    preds = model.predict(texts)
    dists = distribution or texts
    events = []
    for p, d in zip(preds, dists, strict=True):
        states, nationwide = parse_distribution(d)
        events.append(
            ExtractedEvent(
                event_type=p.event_type,
                confidence=round(p.confidence, 4),
                severity=p.severity,
                severity_confidence=None
                if p.severity_confidence is None
                else round(p.severity_confidence, 4),
                direction="negative",
                category="FOODS",
                states=states,
                nationwide=nationwide,
                model=model.name,
            )
        )
    return events
