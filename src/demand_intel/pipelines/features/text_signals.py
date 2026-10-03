"""Text-derived temporal signals for the multimodal experiment (PROJECT_SPEC.md §4.4, §6.2).

Mapping (the only defensible one, spec §2.4): a recall applies to a state when its
distribution names that state or is nationwide, and only to FOODS series (the corpus is
FDA food recalls). Signals are indexed by `report_date` (publication), and the feature
builder reads them at the anchor day, so they are past-only.

- events: trailing 7/28-day counts per hazard group, high-severity count, a severity-
  weighted sum and nationwide count  (arm F: event types from the event model)
- embed:  mean sentence embedding of the trailing 28 days of reports, PCA to 8 dims fit on
  reports published before 2015-07-01 only  (arm E: generic text embeddings)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("CA", "TX", "WI")
GROUPS = {
    "pathogen": ("PATHOGEN_LISTERIA", "PATHOGEN_SALMONELLA", "PATHOGEN_OTHER"),
    "allergen": ("UNDECLARED_ALLERGEN",),
    "contaminant": ("FOREIGN_MATERIAL", "CHEMICAL_CONTAMINANT", "UNAPPROVED_INGREDIENT"),
    "process_quality": ("PROCESSING_DEFECT", "SANITATION_PESTS", "SPOILAGE_QUALITY"),
    "label_other": ("LABELING_OTHER", "OTHER"),
}
SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.2}
PCA_FIT_END = pd.Timestamp("2015-07-01")
EMBED_DIM = 8


@dataclass
class TextSignals:
    events: np.ndarray  # (3 states, n_cal_days, k) trailing-window aggregates by report day
    event_names: list[str]
    embed: np.ndarray  # (3, n_cal_days, EMBED_DIM)
    source: str  # which model produced event types

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            events=self.events,
            embed=self.embed,
            names=np.array(self.event_names),
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: Path) -> "TextSignals":
        z = np.load(path)
        return cls(z["events"], z["names"].tolist(), z["embed"], str(z["source"]))


def _day_index(dates: pd.Series, calendar: pd.DataFrame) -> np.ndarray:
    d_of = pd.Series(calendar["d"].to_numpy(), index=pd.to_datetime(calendar["date"]))
    out: np.ndarray = d_of.reindex(pd.to_datetime(dates)).to_numpy()
    return out


def _trailing(daily: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([np.zeros((1, daily.shape[1])), np.cumsum(daily, axis=0)])
    idx = np.arange(1, len(daily) + 1)
    out: np.ndarray = c[idx] - c[np.maximum(idx - w, 0)]
    return out


def build_signals(
    docs: pd.DataFrame, calendar: pd.DataFrame, embeddings: np.ndarray | None, source: str
) -> TextSignals:
    """docs: doc_id, report_date, states, nationwide, event_type, severity (aligned with
    `embeddings` rows when given)."""
    n_cal = len(calendar)
    day = _day_index(docs["report_date"], calendar)
    inside = ~np.isnan(day)
    docs, day = docs[inside].reset_index(drop=True), day[inside].astype(int)
    emb = embeddings[inside] if embeddings is not None else None

    group_of = {t: g for g, ts in GROUPS.items() for t in ts}
    names = [f"text_{g}_{w}d" for w in (7, 28) for g in GROUPS] + [
        "text_high_severity_7d",
        "text_high_severity_28d",
        "text_severity_sum_7d",
        "text_severity_sum_28d",
        "text_nationwide_7d",
        "text_all_28d",
    ]
    events = np.zeros((len(STATES), n_cal, len(names)), dtype=np.float32)
    embed = np.zeros((len(STATES), n_cal, EMBED_DIM), dtype=np.float32)

    pca = None
    if emb is not None:
        from sklearn.decomposition import PCA

        fit = (pd.to_datetime(docs["report_date"]) < PCA_FIT_END).to_numpy()
        k = min(EMBED_DIM, int(fit.sum()), emb.shape[1])
        pca = PCA(n_components=k, random_state=0).fit(emb[fit])
        reduced = np.zeros((len(emb), EMBED_DIM))
        reduced[:, :k] = pca.transform(emb)

    for si, state in enumerate(STATES):
        applies = (
            docs["nationwide"] | docs["states"].map(lambda s, st=state: st in list(s))
        ).to_numpy()
        sel = docs[applies]
        d = day[applies] - 1
        daily = np.zeros((n_cal, len(GROUPS) + 3))
        for gi, g in enumerate(GROUPS):
            mask = sel["event_type"].map(group_of).eq(g).to_numpy()
            np.add.at(daily[:, gi], d[mask], 1)
        np.add.at(daily[:, len(GROUPS)], d[(sel["severity"] == "high").to_numpy()], 1)
        np.add.at(
            daily[:, len(GROUPS) + 1], d, sel["severity"].map(SEVERITY_WEIGHT).fillna(0).to_numpy()
        )
        np.add.at(daily[:, len(GROUPS) + 2], d[sel["nationwide"].to_numpy()], 1)
        t7, t28 = _trailing(daily, 7), _trailing(daily, 28)
        ng = len(GROUPS)
        events[si] = np.concatenate(
            [
                t7[:, :ng],
                t28[:, :ng],
                t7[:, ng : ng + 1],
                t28[:, ng : ng + 1],
                t7[:, ng + 1 : ng + 2],
                t28[:, ng + 1 : ng + 2],
                t7[:, ng + 2 : ng + 3],
                t28[:, :ng].sum(axis=1, keepdims=True),
            ],
            axis=1,
        )
        if pca is not None:
            sums = np.zeros((n_cal, EMBED_DIM))
            counts = np.zeros(n_cal)
            np.add.at(sums, d, reduced[applies])
            np.add.at(counts, d, 1)
            s28 = _trailing(sums, 28)
            c28 = _trailing(counts[:, None], 28)[:, 0]
            embed[si] = np.where(c28[:, None] > 0, s28 / np.maximum(c28[:, None], 1), 0.0)
    return TextSignals(events=events, event_names=names, embed=embed, source=source)


def signal_features(
    sig: TextSignals, series: pd.DataFrame, days: np.ndarray, hb: int, kind: str
) -> dict[str, np.ndarray]:
    """Features for series × target days, read at the anchor day (past-only); zero for
    non-FOODS series."""
    anchor = days - hb  # 1-based day
    state_pos = series["state_id"].map({s: i for i, s in enumerate(STATES)}).to_numpy()
    foods = (series["cat_id"] == "FOODS").to_numpy()[:, None]
    if kind == "events":
        arr, names = sig.events, sig.event_names
    else:
        arr, names = sig.embed, [f"text_embed_{j}" for j in range(sig.embed.shape[2])]
    out = {}
    for k, name in enumerate(names):
        vals = arr[state_pos][:, anchor - 1, k]
        out[name] = np.where(foods, vals, 0.0)
    return out
