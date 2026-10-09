"""From-scratch rung (PRD Amendment A2, M3 R5): a small CNN on log-mel spectrograms,
randomly initialized, trained only on our synthetic clips. Licence-clean by construction
(no third-party weights); it answers "how much does pretraining buy?" next to the
MERT/CLAP rungs.

- Features: 22.05 kHz mono, 1024-pt FFT, hop 512, 128 mel bins, log(1+100·mel), 10 s
  -> (128, 431). Cached once per (dataset, split) as float16.
- Model: 4 conv blocks (2× [3×3 conv, BN, ReLU], 2×2 avg-pool), 32-64-128-256 channels,
  freq-mean then time (mean + max) pooling, dropout, linear head over every taxonomy
  node. ~1.2 M params.
- Training: masked BCE with per-node pos_weight (same as the probe), AdamW + cosine LR,
  SpecAugment (2 freq + 2 time masks), random 8 s crops at train time and full 10 s at
  eval, early stopping on validation macro AP.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SR, N_FFT, HOP, N_MELS, SECONDS = 22050, 1024, 512, 128, 10.0
FRAMES = int(SR * SECONDS / HOP) + 1


def logmel(path: Path) -> np.ndarray:
    import librosa
    from .embed import load_clip
    y = load_clip(path, SR, SECONDS)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                       fmin=30, fmax=SR // 2)
    return np.log1p(100.0 * m).astype(np.float16)[:, :FRAMES]


def build_mel_cache(items: Sequence[tuple[str, Path]], out: Path, workers: int = 6) -> Path:
    """Streams log-mels into a disk-backed `.npy` memmap (+ `.ids.txt`), so 100k clips
    (~11 GB) never have to fit in RAM. Older `.npz` caches are still readable."""
    from concurrent.futures import ProcessPoolExecutor
    out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = [i for i, _ in items]
    X = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(ids), N_MELS, FRAMES))
    with ProcessPoolExecutor(workers) as ex:
        for k, m in enumerate(ex.map(logmel, [p for _, p in items], chunksize=32)):
            X[k, :, : m.shape[1]] = m
            if m.shape[1] < FRAMES:
                X[k, :, m.shape[1]:] = 0
    X.flush()
    out.with_suffix(".ids.txt").write_text("\n".join(ids))
    return out


def load_mel_cache(path: Path) -> tuple[list[str], np.ndarray]:
    """-> (ids, X) where X is a read-only memmap for `.npy` caches (batches are read from
    disk on demand), or an in-memory array for legacy `.npz` caches."""
    path = Path(path)
    if path.with_suffix(".npy").exists():
        ids = path.with_suffix(".ids.txt").read_text().split("\n")
        return ids, np.load(path.with_suffix(".npy"), mmap_mode="r")
    z = np.load(path, allow_pickle=False)
    return z["ids"].tolist(), z["X"]


@dataclass
class ScratchConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    patience: int = 8
    crop_frames: int = int(SR * 8.0 / HOP)
    max_pos_weight: float = 20.0
    channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.3
    seed: int = 0


def make_model(n_out: int, cfg: ScratchConfig):
    import torch.nn as nn

    layers, c_in = [], 1
    for c in cfg.channels:
        layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                   nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                   nn.AvgPool2d(2)]
        c_in = c

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn0 = nn.BatchNorm2d(1)
            self.body = nn.Sequential(*layers)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(c_in * 2, n_out)

        def forward(self, x):                       # x: (B, mels, frames)
            import torch
            h = self.body(self.bn0(x[:, None]))     # (B, C, F', T')
            h = h.mean(dim=2)                       # freq pool -> (B, C, T')
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.head(self.drop(h))

    return Net()


def _spec_augment(x, rng, n_masks=2, f_w=16, t_w=40):
    x = x.clone()
    B, F, T = x.shape
    for b in range(B):
        for _ in range(n_masks):
            f = int(rng.integers(0, f_w)); f0 = int(rng.integers(0, max(1, F - f)))
            x[b, f0:f0 + f, :] = 0
            t = int(rng.integers(0, t_w)); t0 = int(rng.integers(0, max(1, T - t)))
            x[b, :, t0:t0 + t] = 0
    return x


def _val_map(y, s, m) -> float:
    from ..eval import metrics as M
    ap, reasons = M.per_class_ap(y, s, m)
    mac = M.macro(ap, reasons)
    return float(mac.value) if mac.value is not None else 0.0


def train(X: np.ndarray, Y: np.ndarray, Mk: np.ndarray, Xv: np.ndarray, Yv: np.ndarray,
          Mv: np.ndarray, cfg: ScratchConfig, device: str | None = None, log=print):
    import torch

    from .embed import _device
    dev = device or _device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = make_model(Y.shape[1], cfg).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)); Mt = torch.from_numpy(Mk.astype(np.float32))
    pw = (((1 - Yt) * Mt).sum(0) / (Yt * Mt).sum(0).clamp_min(1)).clamp(1, cfg.max_pos_weight).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(X) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(X))
        for k in range(0, len(X), cfg.batch_size):
            i = order[k:k + cfg.batch_size]
            st = rng.integers(0, X.shape[2] - cfg.crop_frames + 1, size=len(i))
            xb = np.stack([X[j, :, s:s + cfg.crop_frames] for j, s in zip(i, st)]).astype(np.float32)
            xb = _spec_augment(torch.from_numpy(xb), rng).to(dev)
            z = model(xb)
            yb, mb = Yt[i].to(dev), Mt[i].to(dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, yb, weight=mb, pos_weight=pw, reduction="sum") / mb.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(i)
        sv = predict(model, Xv, dev)
        v = _val_map(Yv, sv, Mv)
        hist.append({"epoch": ep, "train_loss": tot / len(X), "val_map": v, "sec": round(time.time() - t0, 1)})
        log(f"epoch {ep} loss {tot / len(X):.4f} val_mAP {v:.4f} ({time.time() - t0:.0f}s)")
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def predict(model, X: np.ndarray, dev: str, batch: int = 64) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[k:k + batch].astype(np.float32)).to(dev))).cpu().numpy())
    return np.concatenate(out)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def save(model, path: Path, cfg: ScratchConfig, hist: list, extra: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": asdict(cfg)}, path)
    path.with_suffix(".json").write_text(json.dumps({"cfg": asdict(cfg), "history": hist,
                                                     "n_params": n_params(model), **extra}, indent=1))
