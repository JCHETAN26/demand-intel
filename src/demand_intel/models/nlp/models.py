"""NLP model ladder N0-N3 (PROJECT_SPEC.md §7.3, plan Phases 20-21).

Two targets per document:
- `event_type`: 12 hazard classes. Training labels are weak (rules); test truth is the
  human-validated gold set (until it exists, agreement with weak labels is reported and
  labelled as such).
- `severity`: FDA classification (high/medium/low), real ground truth for every document.

Every model implements fit(texts, event_labels, severity_labels) and predict_proba(texts)
→ (event_probs, severity_probs) over the fixed label orders below.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from demand_intel import SEED
from demand_intel.models.nlp.labeling import EVENT_TYPES, weak_label

SEVERITIES = ("high", "medium", "low")


class TextModel(Protocol):
    name: str

    def fit(self, texts: list[str], events: list[str], severities: list[str]) -> None: ...

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]: ...


def _onehot(labels: list[str], classes: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((len(labels), len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for r, lab in enumerate(labels):
        out[r, idx[lab]] = 1.0
    return out


@dataclass
class MajorityModel:
    """N0: predicts the training majority class for both targets."""

    name: str = "n0_majority"
    event_prior: np.ndarray = field(default_factory=lambda: np.zeros(len(EVENT_TYPES)))
    sev_prior: np.ndarray = field(default_factory=lambda: np.zeros(len(SEVERITIES)))

    def fit(self, texts: list[str], events: list[str], severities: list[str]) -> None:
        self.event_prior = _onehot(events, EVENT_TYPES).mean(axis=0)
        self.sev_prior = _onehot(severities, SEVERITIES).mean(axis=0)

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        return np.tile(self.event_prior, (n, 1)), np.tile(self.sev_prior, (n, 1))


@dataclass
class RulesModel:
    """N0b: the labelling rules themselves (event only; severity from the majority prior)."""

    name: str = "n0_rules"
    sev_prior: np.ndarray = field(default_factory=lambda: np.zeros(len(SEVERITIES)))

    def fit(self, texts: list[str], events: list[str], severities: list[str]) -> None:
        self.sev_prior = _onehot(severities, SEVERITIES).mean(axis=0)

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        events = [weak_label(t.split("\nProduct:")[0]).label for t in texts]
        return _onehot(events, EVENT_TYPES), np.tile(self.sev_prior, (len(texts), 1))


def _logreg() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)


def _proba(clf: Any, X: Any, classes: tuple[str, ...]) -> np.ndarray:  # noqa: N803
    p = clf.predict_proba(X)
    out = np.zeros((p.shape[0], len(classes)))
    for j, c in enumerate(clf.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


@dataclass
class TfidfModel:
    """N1: word + character TF-IDF, class-balanced logistic regression per target."""

    name: str = "n1_tfidf_logreg"
    vec: Any = None
    clf_event: Any = None
    clf_sev: Any = None

    def fit(self, texts: list[str], events: list[str], severities: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.vec = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                    ),
                ),
            ]
        )
        X = self.vec.fit_transform(texts)  # noqa: N806
        self.clf_event = _logreg().fit(X, events)
        self.clf_sev = _logreg().fit(X, severities)

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = self.vec.transform(texts)  # noqa: N806
        return _proba(self.clf_event, X, EVENT_TYPES), _proba(self.clf_sev, X, SEVERITIES)


@dataclass
class EmbeddingModel:
    """N2: frozen generic sentence embeddings + logistic regression per target."""

    name: str = "n2_embeddings_logreg"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder: Any = None
    clf_event: Any = None
    clf_sev: Any = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.encoder_name, device="cpu")
        emb: np.ndarray = self.encoder.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return emb

    def fit(self, texts: list[str], events: list[str], severities: list[str]) -> None:
        X = self._embed(texts)  # noqa: N806
        self.clf_event = _logreg().fit(X, events)
        self.clf_sev = _logreg().fit(X, severities)

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = self._embed(texts)  # noqa: N806
        return _proba(self.clf_event, X, EVENT_TYPES), _proba(self.clf_sev, X, SEVERITIES)


@dataclass
class EncoderModel:
    """N3: fine-tuned transformer encoder with two classification heads (multi-task).

    Loss = cross-entropy(event) + cross-entropy(severity); class-weighted event loss
    counters the skewed weak-label distribution. Early stopping on validation loss.
    """

    name: str = "n3_encoder"
    base: str = "distilroberta-base"
    max_length: int = 160
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-5
    threads: int = 4
    save_dir: Path | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    model: Any = None
    tokenizer: Any = None

    def _build(self) -> None:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base)
        encoder = AutoModel.from_pretrained(self.base)

        class TwoHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                hidden = encoder.config.hidden_size
                self.drop = nn.Dropout(0.1)
                self.event = nn.Linear(hidden, len(EVENT_TYPES))
                self.severity = nn.Linear(hidden, len(SEVERITIES))

            def forward(self, **batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                h = self.encoder(**batch).last_hidden_state[:, 0]
                h = self.drop(h)
                return self.event(h), self.severity(h)

        self.model = TwoHead()

    def _batches(
        self, texts: list[str], *labels: np.ndarray, shuffle: bool, rng: np.random.Generator
    ) -> Any:
        order = rng.permutation(len(texts)) if shuffle else np.arange(len(texts))
        for lo in range(0, len(texts), self.batch_size):
            idx = order[lo : lo + self.batch_size]
            enc = self.tokenizer(
                [texts[i] for i in idx],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            yield enc, [lab[idx] for lab in labels]

    def fit(
        self,
        texts: list[str],
        events: list[str],
        severities: list[str],
        val: tuple[list[str], list[str], list[str]] | None = None,
    ) -> None:
        import torch

        torch.manual_seed(SEED)
        torch.set_num_threads(self.threads)
        self._build()
        rng = np.random.default_rng(SEED)
        ev = np.array([EVENT_TYPES.index(e) for e in events])
        sv = np.array([SEVERITIES.index(s) for s in severities])
        counts = np.bincount(ev, minlength=len(EVENT_TYPES)).astype(float)
        weights = torch.tensor(
            np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0),
            dtype=torch.float32,
        ).clamp(max=10.0)
        ce_event = torch.nn.CrossEntropyLoss(weight=weights)
        ce_sev = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * (len(texts) // self.batch_size + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=steps, pct_start=0.1
        )
        best, best_state = float("inf"), None
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for enc, (e, s) in self._batches(texts, ev, sv, shuffle=True, rng=rng):
                le, ls = self.model(**enc)
                loss = ce_event(le, torch.as_tensor(e)) + ce_sev(ls, torch.as_tensor(s))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(e)
            val_loss = float("nan")
            if val is not None:
                pe, ps = self.predict_proba(val[0])
                ve = np.array([EVENT_TYPES.index(x) for x in val[1]])
                vs = np.array([SEVERITIES.index(x) for x in val[2]])
                val_loss = float(
                    -np.log(pe[np.arange(len(ve)), ve] + 1e-9).mean()
                    - np.log(ps[np.arange(len(vs)), vs] + 1e-9).mean()
                )
                if val_loss < best:
                    best = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.log.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(texts),
                    "val_loss": val_loss,
                    "seconds": time.perf_counter() - t0,
                }
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.save_dir is not None:
            self.save(self.save_dir)

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.model.eval()
        pe, ps = [], []
        with torch.no_grad():
            for enc, _ in self._batches(texts, shuffle=False, rng=np.random.default_rng(0)):
                le, ls = self.model(**enc)
                pe.append(torch.softmax(le, -1).numpy())
                ps.append(torch.softmax(ls, -1).numpy())
        return np.concatenate(pe), np.concatenate(ps)

    def save(self, path: Path) -> None:
        import torch

        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(path)
        (path / "config.txt").write_text(f"{self.base}\n{self.max_length}\n")

    @classmethod
    def load(cls, path: Path) -> "EncoderModel":
        import torch

        base, max_len = (path / "config.txt").read_text().split()
        m = cls(base=base, max_length=int(max_len))
        m._build()
        m.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        m.model.eval()
        return m
