"""NLP splits with no leakage (PROJECT_SPEC.md §7.1).

- Time: train < 2015-07-01 <= validation < 2016-01-01 <= test <= 2016-06-19 (the end of
  the M5 calendar). The model that produces forecasting signals for the M5 test window is
  therefore trained only on reports published before it.
- Events: a recall event (openFDA `event_id`) can have many reports. Each event goes
  wholly to the split of its earliest report.
- Near-duplicates: validation/test reports whose text is a near-duplicate (MinHash
  Jaccard >= 0.8 on word 3-shingles) of any training report are removed. Firms reuse
  wording across recalls, and keeping such reports would inflate test scores.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2015-07-01")
VAL_END = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2016-06-19")
JACCARD = 0.8
NUM_PERM = 128


def _shingles(text: str, k: int = 3) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + k]) for i in range(max(len(words) - k + 1, 1))}


def near_duplicates_of(reference: pd.Series, candidates: pd.Series) -> np.ndarray:
    """Boolean mask over `candidates`: near-duplicate of some text in `reference`."""
    from datasketch import MinHash, MinHashLSH

    def sketch(text: str) -> "MinHash":
        m = MinHash(num_perm=NUM_PERM)
        for s in _shingles(text):
            m.update(s.encode())
        return m

    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    for i, text in enumerate(reference):
        lsh.insert(str(i), sketch(text))
    return np.array([bool(lsh.query(sketch(t))) for t in candidates])


@dataclass
class Splits:
    frame: pd.DataFrame  # documents with a `split` column (train / validation / test / dropped)
    stats: dict[str, int]


def make_splits(docs: pd.DataFrame, text_col: str = "text") -> Splits:
    df = docs[docs["report_date"] <= TEST_END].copy()
    first = df.groupby("event_id")["report_date"].transform("min")
    df["split"] = np.select(
        [first < TRAIN_END, first < VAL_END], ["train", "validation"], default="test"
    )
    train_text = df.loc[df["split"] == "train", text_col]
    for split in ("validation", "test"):
        idx = df.index[df["split"] == split]
        dup = near_duplicates_of(train_text, df.loc[idx, text_col])
        df.loc[idx[dup], "split"] = f"dropped_near_duplicate_{split}"
    stats = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    return Splits(frame=df, stats=stats)


def gold_sample(test: pd.DataFrame, label_col: str, size: int, seed: int) -> pd.DataFrame:
    """Stratified sample for human validation: every class gets min(its size, floor) docs,
    the rest proportionally, so rare classes are measurable in macro-F1."""
    rng = np.random.default_rng(seed)
    counts = test[label_col].value_counts()
    floor = max(size // (3 * len(counts)), 1)
    take = counts.clip(upper=floor)
    remaining = size - int(take.sum())
    if remaining > 0:
        extra = ((counts - take) / (counts - take).sum() * remaining).round().astype(int)
        take = (take + extra).clip(upper=counts)
    picks = [
        rng.choice(test.index[test[label_col] == lab], size=int(n), replace=False)
        for lab, n in take.items()
    ]
    return test.loc[np.sort(np.concatenate(picks))]
