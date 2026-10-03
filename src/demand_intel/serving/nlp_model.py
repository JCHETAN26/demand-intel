"""Loader for the registered NLP champion; `None` keeps the rule-based fallback."""

from demand_intel.serving.events import EventModel


def load_champion() -> EventModel | None:
    return None
