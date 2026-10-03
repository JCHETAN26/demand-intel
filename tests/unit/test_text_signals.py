import numpy as np
import pandas as pd

from demand_intel.pipelines.features.text_signals import build_signals, signal_features
from tests.conftest import synthetic_m5


def test_signals_are_past_only_state_mapped_and_foods_only() -> None:
    m5 = synthetic_m5()
    docs = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "report_date": pd.to_datetime(["2012-03-01", "2012-03-03", "2012-03-03"]),
            "states": [["CA"], ["TX"], []],
            "nationwide": [False, False, True],
            "event_type": ["PATHOGEN_LISTERIA", "UNDECLARED_ALLERGEN", "FOREIGN_MATERIAL"],
            "severity": ["high", "medium", "low"],
        }
    )
    rng = np.random.default_rng(0)
    sig = build_signals(docs, m5.calendar, rng.normal(size=(3, 16)), source="test")
    d_of = dict(zip(pd.to_datetime(m5.calendar["date"]), m5.calendar["d"], strict=True))
    da, db = d_of[pd.Timestamp("2012-03-01")], d_of[pd.Timestamp("2012-03-03")]
    names = sig.event_names
    ca = sig.events[0]
    assert ca[da - 1, names.index("text_pathogen_7d")] == 1
    assert ca[da - 2, names.index("text_pathogen_7d")] == 0  # nothing before publication
    assert ca[db - 1, names.index("text_contaminant_7d")] == 1  # nationwide reaches CA
    assert ca[db - 1, names.index("text_allergen_7d")] == 0  # TX-only recall does not
    assert sig.events[1][db - 1, names.index("text_allergen_7d")] == 1

    feats = signal_features(sig, m5.series, np.array([db + 7]), 7, "events")
    foods = (m5.series["cat_id"] == "FOODS").to_numpy()
    assert (feats["text_all_28d"][~foods] == 0).all()
    assert (feats["text_all_28d"][foods] > 0).all()
    emb = signal_features(sig, m5.series, np.array([db + 7]), 7, "embed")
    assert len(emb) == 8
