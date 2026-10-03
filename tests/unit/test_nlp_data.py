import numpy as np
import pandas as pd
import pytest

from demand_intel.models.nlp.labeling import EVENT_TYPES, weak_label


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("Product may be contaminated with Listeria monocytogenes.", "PATHOGEN_LISTERIA"),
        ("Milk powder ingredient tested positive for Salmonella.", "PATHOGEN_SALMONELLA"),
        ("Undeclared milk and soy.", "UNDECLARED_ALLERGEN"),
        ("The product contains undeclared sildenafil.", "UNAPPROVED_INGREDIENT"),
        ("Pieces of metal found in product.", "FOREIGN_MATERIAL"),
        ("Product was determined to be under processed.", "PROCESSING_DEFECT"),
        ("Elevated levels of lead.", "CHEMICAL_CONTAMINANT"),
        ("Recalled due to rancidity.", "SPOILAGE_QUALITY"),
        ("Insanitary conditions at the facility.", "SANITATION_PESTS"),
        ("Incorrect net weight on label.", "LABELING_OTHER"),
        ("Firm initiated recall.", "OTHER"),
        ("Possible Cyclospora contamination of lettuce.", "PATHOGEN_OTHER"),
    ],
)
def test_weak_labels(reason: str, expected: str) -> None:
    assert weak_label(reason).label == expected
    assert expected in EVENT_TYPES


def test_conflicts_are_flagged() -> None:
    w = weak_label("Listeria found; label also missing undeclared wheat.")
    assert w.label == "PATHOGEN_LISTERIA" and w.conflict


def test_splits_respect_time_events_and_near_duplicates() -> None:
    pytest.importorskip("datasketch")
    from demand_intel.models.nlp.dataset import gold_sample, make_splits

    base = "Firm is recalling granola bars because they may contain undeclared peanuts and almonds"
    docs = pd.DataFrame(
        {
            "doc_id": [f"D{i}" for i in range(6)],
            "event_id": ["e1", "e2", "e2", "e3", "e4", "e5"],
            "report_date": pd.to_datetime(
                ["2014-01-01", "2015-06-01", "2015-08-01", "2015-09-01", "2016-02-01", "2016-03-01"]
            ),
            "text": [
                base,
                "Salmonella in pistachios sold in bulk bins nationwide",
                "Salmonella in pistachios second lot",
                "Listeria in deli salads made at plant A",
                base + " today",
                "Metal fragments found in frozen vegetables bag",
            ],
        }
    )
    s = make_splits(docs).frame.set_index("doc_id")["split"]
    assert s["D0"] == "train"
    assert s["D2"] == "train"  # event e2 started in train → whole event in train
    assert s["D3"] == "validation"
    assert s["D4"] == "dropped_near_duplicate_test"  # near-duplicate of a training doc
    assert s["D5"] == "test"

    test = pd.DataFrame({"weak_label": ["A"] * 50 + ["B"] * 5 + ["C"] * 2})
    g = gold_sample(test, "weak_label", 20, seed=0)
    assert len(g) <= 20 and set(g["weak_label"]) == {"A", "B", "C"}
    assert np.unique(g.index).size == len(g)
