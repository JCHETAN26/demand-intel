"""NLP inference benchmark (PROJECT_SPEC.md §7.4, plan Phases 22, 32).

accuracy × latency × throughput × memory × size × cost, on CPU, for:
- N1 TF-IDF + logistic regression
- N2 MiniLM embeddings + logistic regression
- N3 fine-tuned encoder: fp32 PyTorch, int8 dynamic quantization, ONNX Runtime (fp32)

Cost per 1,000 documents is an ESTIMATE: single-document wall time × AWS Fargate
on-demand prices (us-west-2: $0.04048 per vCPU-hour, $0.004445 per GB-hour) for a
1 vCPU / 2 GB task. Structured-output validity is 100% by construction for classifiers
(they always emit a label from the fixed set); it is measured for generative models.
"""

import gc
import os
import resource
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VCPU_HOUR = 0.04048
GB_HOUR = 0.004445


@dataclass
class Candidate:
    name: str
    predict: Callable[[list[str]], tuple[np.ndarray, np.ndarray]]
    size_mb: float


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _dir_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def measure(c: Candidate, texts: list[str], n_single: int = 200, batch: int = 32) -> dict[str, Any]:
    c.predict(texts[:8])  # warm-up
    single = []
    for t in texts[:n_single]:
        t0 = time.perf_counter()
        c.predict([t])
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for lo in range(0, len(texts), batch):
        c.predict(texts[lo : lo + batch])
    batch_s = time.perf_counter() - t0
    p50 = float(np.percentile(single, 50))
    hours_per_1k = p50 * 1000 / 3600
    return {
        "model": c.name,
        "p50_ms": p50 * 1000,
        "p95_ms": float(np.percentile(single, 95)) * 1000,
        "throughput_docs_per_s": len(texts) / batch_s,
        "size_mb": c.size_mb,
        "peak_rss_mb": _peak_rss_mb(),
        "cost_per_1k_docs_usd_estimate": hours_per_1k * (VCPU_HOUR + 2 * GB_HOUR),
        "schema_validity": 1.0,
    }


def encoder_variants(model_dir: Path, threads: int = 1) -> tuple[list[Candidate], list[str]]:
    """fp32, int8 dynamic-quantized and ONNX versions of the saved N3 encoder."""
    import torch

    from demand_intel.models.nlp.models import EncoderModel

    torch.set_num_threads(threads)
    notes: list[str] = []
    fp32 = EncoderModel.load(model_dir)
    out = [Candidate("n3_encoder_fp32", fp32.predict_proba, _dir_mb(model_dir))]

    q = EncoderModel.load(model_dir)
    q.model = torch.ao.quantization.quantize_dynamic(q.model, {torch.nn.Linear}, dtype=torch.qint8)
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(q.model.state_dict(), Path(tmp) / "q.pt")
        q_size = (Path(tmp) / "q.pt").stat().st_size / 1e6
    out.append(Candidate("n3_encoder_int8_dynamic", q.predict_proba, q_size))

    try:
        import onnxruntime as ort

        onnx_path = Path(tempfile.mkdtemp()) / "encoder.onnx"
        enc = fp32.tokenizer(["warm up text"], return_tensors="pt", padding=True)

        class Wrapper(torch.nn.Module):
            def __init__(self, m: Any) -> None:
                super().__init__()
                self.m = m

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
                return self.m(input_ids=input_ids, attention_mask=attention_mask)

        torch.onnx.export(
            Wrapper(fp32.model).eval(),
            (enc["input_ids"], enc["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["event", "severity"],
            dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}},
            opset_version=17,
            dynamo=False,
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        def onnx_predict(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
            e = fp32.tokenizer(
                texts,
                truncation=True,
                max_length=fp32.max_length,
                padding=True,
                return_tensors="np",
            )
            le, ls = sess.run(
                None, {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
            )

            def softmax(x: np.ndarray) -> np.ndarray:
                z = np.exp(x - x.max(axis=1, keepdims=True))
                return z / z.sum(axis=1, keepdims=True)

            return softmax(le), softmax(ls)

        out.append(Candidate("n3_encoder_onnx_fp32", onnx_predict, onnx_path.stat().st_size / 1e6))
    except Exception as err:  # export support differs across torch versions; report, don't fail
        notes.append(f"ONNX export failed: {type(err).__name__}: {str(err)[:200]}")
    gc.collect()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return out, notes
