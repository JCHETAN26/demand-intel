"""Training loop and `Forecaster` wrapper shared by the PyTorch models (plan Phase 14).

Per fold: the network trains on windows whose targets end at least 28 days before the
origin; the window whose targets are the last 28 observed days is the validation set for
early stopping and checkpoint selection (the same out-of-sample split LightGBM uses). The
best checkpoint forecasts from the origin.

Engineering: Dataset/DataLoader, AdamW, gradient clipping, early stopping, best-checkpoint
saving, and mixed precision (bf16 autocast) when a CUDA device is available.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from demand_intel import SEED
from demand_intel.forecasting.backtesting.harness import FoldContext, Forecast
from demand_intel.forecasting.backtesting.splits import HORIZON
from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.models.deep_learning.data import ENC, SequenceArrays, WindowDataset, window

QUANTILE_T = torch.tensor(QUANTILES)


def pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """q: (B, H, Q), y: (B, H)."""
    diff = y.unsqueeze(-1) - q
    tau = QUANTILE_T.to(q.device)
    return torch.maximum(tau * diff, (tau - 1) * diff).mean()


def forecast_loss(mean: torch.Tensor, q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared error for the mean (matches WRMSSE) + pinball for the quantiles."""
    return torch.mean((mean - y) ** 2) + pinball_loss(q, y)


@dataclass
class TrainConfig:
    train_days: int = 728
    origin_stride: int = 3
    samples_per_epoch: int = 60_000
    batch_size: int = 512
    epochs: int = 15
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    clip: float = 1.0
    threads: int = 4


def _batches(ds: WindowDataset, batch: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g, num_workers=0)


def _to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class DeepForecaster:
    name: str
    model_factory: Callable[[SequenceArrays], nn.Module]
    config: TrainConfig = field(default_factory=TrainConfig)
    checkpoint_dir: Path | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    def _fit(self, arr: SequenceArrays, origin: int, fold: str) -> nn.Module:
        cfg = self.config
        torch.manual_seed(SEED)
        torch.set_num_threads(cfg.threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        rng = np.random.default_rng(SEED)

        val_origin = origin - HORIZON
        first = max(ENC, val_origin - cfg.train_days)
        train_origins = np.arange(first, val_origin - HORIZON + 1, cfg.origin_stride)
        val_ds = WindowDataset(arr, np.array([val_origin]))
        model = self.model_factory(arr).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        best, best_state, bad, t0 = (
            float("inf"),
            copy.deepcopy(model.state_dict()),
            0,
            time.perf_counter(),
        )
        for epoch in range(cfg.epochs):
            model.train()
            ds = WindowDataset(arr, train_origins, rng, cfg.samples_per_epoch)
            train_loss = 0.0
            for batch in _batches(ds, cfg.batch_size, True, SEED + epoch):
                b = _to(batch, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    mean, q = model(b)
                    loss = forecast_loss(mean.float(), q.float(), b["y"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                opt.step()
                train_loss += float(loss.detach()) * len(b["y"])
            sched.step()
            model.eval()
            val_loss, n = 0.0, 0
            with torch.no_grad():
                for batch in _batches(val_ds, 2048, False, SEED):
                    b = _to(batch, device)
                    mean, q = model(b)
                    val_loss += float(forecast_loss(mean.float(), q.float(), b["y"])) * len(b["y"])
                    n += len(b["y"])
            val_loss /= max(n, 1)
            self.log.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss / len(ds),
                    "val_loss": val_loss,
                    "seconds": time.perf_counter() - t0,
                }
            )
            if val_loss < best - 1e-5:
                best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        if self.checkpoint_dir is not None:
            d = self.checkpoint_dir / self.name
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "val_loss": best}, d / f"{fold}.pt")
        return model

    def forecast(self, ctx: FoldContext) -> Forecast:
        arr = SequenceArrays.build(ctx.data, ctx.rows)
        model = self._fit(arr, ctx.origin, ctx.fold.name)
        device = next(model.parameters()).device
        model.eval()
        means, quants = [], []
        with torch.no_grad():
            for lo in range(0, len(ctx.rows), 2048):
                items = [
                    window(arr, i, ctx.origin, False)
                    for i in range(lo, min(lo + 2048, len(ctx.rows)))
                ]
                b = _to({k: torch.stack([s[k] for s in items]) for k in items[0]}, device)
                mean, q = model(b)
                scale = b["scale"][:, None]
                means.append((mean.float() * scale).cpu().numpy())
                quants.append((q.float() * scale[..., None]).cpu().numpy())
        point = np.concatenate(means)
        qarr = np.concatenate(quants)
        on_sale = arr.known[:, ctx.origin : ctx.origin + HORIZON, 3] > 0
        point = np.where(on_sale, np.clip(point, 0, None), 0.0).astype(np.float32)
        quantiles = {
            q: np.where(on_sale, np.clip(qarr[:, :, j], 0, None), 0.0).astype(np.float32)
            for j, q in enumerate(QUANTILES)
        }
        return Forecast(point=point, quantiles=quantiles)
