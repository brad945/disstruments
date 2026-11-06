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
    from .embed import load_clip
    return logmel_array(load_clip(path, SR, SECONDS))


def logmel_array(y: np.ndarray) -> np.ndarray:
    """Mono audio at `SR` (one window) -> (N_MELS, <=FRAMES) float16. Shared by the mel cache
    and the app's serving path (M5), so training and serving features cannot drift."""
    import librosa
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
    # M5 follow-up: random input gain, uniform in [-gain_db, +gain_db] dB per clip (0 = off).
    # Renders are loudness-normalised, so without it the model learns a level prior
    # (served model: MoisesDB AP fell steadily from +3 dB to -6 dB).
    gain_db: float = 0.0
    # Anchor plan Phase 1 (docs/debriefs/anchor-plan.md): fine-tune recipe controls.
    schedule: str = "onecycle"            # "onecycle" | "flat-decay" (no re-warm: flat, then linear to 0)
    flat_fraction: float = 0.7            # flat-decay: fraction of steps at the full LR
    head_lr: float = 0.0                  # 0 = same as lr; else the head gets its own LR
    freeze_bn_stats: bool = False         # BN running mean/var frozen (affine still trains)
    frozen_rows: tuple[int, ...] = ()     # head rows held at their initial values
    freeze_blocks: tuple[int, ...] = ()   # conv blocks (1-based) excluded from training


def flat_decay_factor(step: int, total: int, flat_fraction: float) -> float:
    """LR multiplier: 1 for the first `flat_fraction` of steps, then linear to 0."""
    flat = int(total * flat_fraction)
    if step < flat:
        return 1.0
    return max(0.0, 1.0 - (step - flat) / max(1, total - flat))


def block_of(key: str) -> str:
    """State-dict key -> parameter group name: body.<i>.* has 7 modules per conv block."""
    if key.startswith("body."):
        return f"block{int(key.split('.')[1]) // 7 + 1}"
    return key.split(".")[0]


def relative_weight_change(s_from: dict, s_to: dict) -> dict[str, float]:
    """||theta_to - theta_from|| / ||theta_from|| per block (float params only, no running stats)."""
    import torch
    acc: dict[str, list[float]] = {}
    for k, t0 in s_from.items():
        if not torch.is_floating_point(t0) or "running" in k:
            continue
        d = acc.setdefault(block_of(k), [0.0, 0.0])
        d[0] += float(((s_to[k].float() - t0.float()) ** 2).sum())
        d[1] += float((t0.float() ** 2).sum())
    return {g: (v[0] ** 0.5) / (v[1] ** 0.5 + 1e-12) for g, v in acc.items()}


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


def _gain(x, rng, max_db: float):
    """Exact level change on cached features: they are log1p(100 * mel power), so a gain
    of g dB scales the power by 10**(g/10) inside the log1p."""
    import torch
    g = torch.from_numpy(10 ** (rng.uniform(-max_db, max_db, size=(x.shape[0], 1, 1)) / 10)).to(x.dtype)
    return torch.log1p(g * torch.expm1(x))


def _val_map(y, s, m) -> float:
    from ..eval import metrics as M
    ap, reasons = M.per_class_ap(y, s, m)
    mac = M.macro(ap, reasons)
    return float(mac.value) if mac.value is not None else 0.0


def train(X: np.ndarray, Y: np.ndarray, Mk: np.ndarray, Xv: np.ndarray, Yv: np.ndarray,
          Mv: np.ndarray, cfg: ScratchConfig, device: str | None = None, log=print,
          init_state: dict | None = None, diag: dict | None = None):
    """`init_state` (a saved model's state dict) starts from those weights instead of a
    random init: the triangle's fine-tune corner. Same architecture required.
    `diag`, if given, is filled with per-run diagnostics (anchor plan: epoch-1 grad-norm /
    weight-norm ratios per block, relative weight change from the start, best epoch)."""
    import torch
    import torch.nn as nn

    from .embed import _device
    dev = device or _device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = make_model(Y.shape[1], cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(dev)
    start_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    # Frozen conv blocks: no gradient, and their BN statistics never update.
    frozen_bn: list[nn.Module] = []
    for b in cfg.freeze_blocks:
        for mod in list(model.body)[7 * (b - 1): 7 * b]:
            for p in mod.parameters():
                p.requires_grad_(False)
            if isinstance(mod, nn.BatchNorm2d):
                frozen_bn.append(mod)
    all_bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_eval = all_bn if cfg.freeze_bn_stats else frozen_bn

    Yt = torch.from_numpy(Y.astype(np.float32)); Mt = torch.from_numpy(Mk.astype(np.float32))
    pw = (((1 - Yt) * Mt).sum(0) / (Yt * Mt).sum(0).clamp_min(1)).clamp(1, cfg.max_pos_weight).to(dev)
    head_lr = cfg.head_lr or cfg.lr
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad],
               "lr": cfg.lr},
              {"params": list(model.head.parameters()), "lr": head_lr}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(X) / cfg.batch_size))
    if cfg.schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr, head_lr], total_steps=steps, pct_start=0.1)
    elif cfg.schedule == "flat-decay":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: flat_decay_factor(s, steps, cfg.flat_fraction))
    else:
        raise ValueError(f"unknown schedule {cfg.schedule!r}")
    frozen_rows = list(cfg.frozen_rows)
    with torch.no_grad():
        w_frozen = model.head.weight[frozen_rows].clone()
        b_frozen = model.head.bias[frozen_rows].clone()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rgn: dict[str, list[float]] = {}
    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    for ep in range(cfg.epochs):
        model.train()
        for m in bn_eval:
            m.eval()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(X))
        for k in range(0, len(X), cfg.batch_size):
            i = order[k:k + cfg.batch_size]
            st = rng.integers(0, X.shape[2] - cfg.crop_frames + 1, size=len(i))
            xb = np.stack([X[j, :, s:s + cfg.crop_frames] for j, s in zip(i, st)]).astype(np.float32)
            xb = torch.from_numpy(xb)
            if cfg.gain_db > 0:
                xb = _gain(xb, rng, cfg.gain_db)
            xb = _spec_augment(xb, rng).to(dev)
            z = model(xb)
            yb, mb = Yt[i].to(dev), Mt[i].to(dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, yb, weight=mb, pos_weight=pw, reduction="sum") / mb.sum().clamp_min(1)
            opt.zero_grad(); loss.backward()
            if ep == 0:                                 # Auto-RGN style: which tensors want to move
                with torch.no_grad():
                    for n, p in named:
                        if p.grad is not None:
                            rgn.setdefault(n, []).append(float(p.grad.norm() / (p.norm() + 1e-12)))
            opt.step(); sched.step()
            if frozen_rows:
                with torch.no_grad():
                    model.head.weight[frozen_rows] = w_frozen
                    model.head.bias[frozen_rows] = b_frozen
            tot += float(loss.detach()) * len(i)
        sv = predict(model, Xv, dev)
        v = _val_map(Yv, sv, Mv)
        hist.append({"epoch": ep, "train_loss": tot / len(X), "val_map": v, "sec": round(time.time() - t0, 1)})
        log(f"epoch {ep} loss {tot / len(X):.4f} val_mAP {v:.4f} ({time.time() - t0:.0f}s)")
        if v > best + 1e-4:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if diag is not None:
        per_block: dict[str, list[float]] = {}
        for n, vals in rgn.items():
            per_block.setdefault(block_of(n), []).append(float(np.mean(vals)))
        diag.update({"rgn_epoch1": {g: float(np.mean(v)) for g, v in per_block.items()},
                     "relative_weight_change": relative_weight_change(start_state, best_state),
                     "lr_groups": [cfg.lr, head_lr], "schedule": cfg.schedule,
                     "best_epoch": best_ep, "epochs_run": len(hist),
                     "train_loss_at_best": hist[best_ep]["train_loss"] if best_ep >= 0 else None,
                     "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                     "frozen_rows": len(frozen_rows), "freeze_blocks": list(cfg.freeze_blocks),
                     "freeze_bn_stats": cfg.freeze_bn_stats})
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


def load_model(path: Path, n_out: int, device: str):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = dict(ck["cfg"])
    for key in ("channels", "frozen_rows", "freeze_blocks"):
        if key in c:
            c[key] = tuple(c[key])
    cfg = ScratchConfig(**c)
    model = make_model(n_out, cfg)
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def save(model, path: Path, cfg: ScratchConfig, hist: list, extra: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": asdict(cfg)}, path)
    path.with_suffix(".json").write_text(json.dumps({"cfg": asdict(cfg), "history": hist,
                                                     "n_params": n_params(model), **extra}, indent=1))
