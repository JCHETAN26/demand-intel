"""Event taxonomy and weak-supervision labelling functions (PROJECT_SPEC.md §7, plan 18-19).

The taxonomy was derived from the openFDA corpus (see docs/nlp_taxonomy.md): recall
reasons fall into these hazard types, covering about 92% of reports with simple patterns.
Weak labels are *candidates*: they seed training data and pre-fill the human validation
tool. They are never ground truth; the test set uses human-validated labels only.

Rules are applied in priority order, and the first rule that fires sets the weak label.
Specific causes beat generic ones: a pathogen in a dairy product is a pathogen recall, not
an allergen one, and an undeclared allergen is not a generic labelling issue.
"""

import re
from dataclasses import dataclass

import pandas as pd

EVENT_TYPES = (
    "UNAPPROVED_INGREDIENT",
    "CHEMICAL_CONTAMINANT",
    "PATHOGEN_LISTERIA",
    "PATHOGEN_SALMONELLA",
    "PATHOGEN_OTHER",
    "PROCESSING_DEFECT",
    "FOREIGN_MATERIAL",
    "UNDECLARED_ALLERGEN",
    "SANITATION_PESTS",
    "SPOILAGE_QUALITY",
    "LABELING_OTHER",
    "OTHER",
)

ALLERGENS = (
    r"milk|dairy|whey|casein|lactose|peanuts?|soy(bean)?|wheat|gluten|eggs?|tree ?nuts?|almonds?|"
    r"walnuts?|cashews?|pecans?|hazelnuts?|pistachios?|macadamia|brazil nuts?|coconut|"
    r"shellfish|shrimp|crab|lobster|crustacean|fish|anchov|sesame|mustard|sulfites?|fd&c|yellow ?(no\.? ?)?5"
)

RULES: tuple[tuple[str, str], ...] = (
    (
        "UNAPPROVED_INGREDIENT",
        r"unapproved|sibutramine|sildenafil|tadalafil|dmaa|ephedr|kratom|oleander|phenolphthalein|"
        r"undeclared (drug|active pharmaceutical)|steroid|not approved",
    ),
    (
        "CHEMICAL_CONTAMINANT",
        r"\blead\b|arsenic|pesticide|mercury|cadmium|bromate|chloramphenicol|histamine|scombro|"
        r"aflatoxin|mycotoxin|patulin|chemical|cleaning (agent|solution)|sanitizer|toxin|"
        r"melamine|benzene|ethylene oxide|excess(ive)? (levels? of )?(vitamin|iron|selenium|iodine)",
    ),
    ("PATHOGEN_LISTERIA", r"listeria"),
    ("PATHOGEN_SALMONELLA", r"salmonella"),
    (
        "PATHOGEN_OTHER",
        r"e\.? ?coli|\bstec\b|o157|cyclospora|norovirus|hepatitis|clostridium|bacillus cereus|"
        r"staphylococcus|cronobacter|vibrio|campylobacter|shigella|pathogen",
    ),
    (
        "PROCESSING_DEFECT",
        r"botul|under[- ]?process|improper(ly)? (processed|acidified|heat|seal)|acidif|low[- ]acid|"
        r"swollen|seal|leak|unpasteuri[sz]ed|pasteuri[sz]ation|haccp|temperature|refrigerat|thaw|"
        r"not (properly )?(cooked|processed|retorted)|scheduled process|water activity",
    ),
    (
        "FOREIGN_MATERIAL",
        r"foreign|metal|glass|plastic|rubber|wood|stone|rock|bone (fragment|piece)|fragment|"
        r"hard objects?|wire|insulation",
    ),
    (
        "UNDECLARED_ALLERGEN",
        rf"undeclared|allergen|(contains?|may contain|presence of) ({ALLERGENS})|({ALLERGENS}) (not declared|not listed)",
    ),
    (
        "SANITATION_PESTS",
        r"insanitary|unsanitary|sanitation|rodent|insect|beetle|pest|filth|\bgmps?\b|cgmp|"
        r"good manufacturing|conditions",
    ),
    (
        "SPOILAGE_QUALITY",
        r"mold|mould|spoil|yeast|fermentat|decompos|rancid|off[- ](odor|taste|flavor)",
    ),
    ("LABELING_OTHER", r"label|misbrand|mislabel|net weight|nutrition facts|incorrect|wrong"),
)
_COMPILED = tuple((name, re.compile(pattern, re.I)) for name, pattern in RULES)


@dataclass(frozen=True)
class WeakLabel:
    label: str
    fired: tuple[str, ...]

    @property
    def conflict(self) -> bool:
        return len(self.fired) > 1


def weak_label(reason: str) -> WeakLabel:
    fired = tuple(name for name, rx in _COMPILED if rx.search(reason or ""))
    return WeakLabel(fired[0] if fired else "OTHER", fired)


def weak_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["reason_for_recall"].map(weak_label)
    return pd.DataFrame(
        {
            "doc_id": df["doc_id"].to_numpy(),
            "weak_label": labels.map(lambda w: w.label).to_numpy(),
            "lf_fired": labels.map(lambda w: list(w.fired)).to_numpy(),
            "lf_conflict": labels.map(lambda w: w.conflict).to_numpy(),
        }
    )
