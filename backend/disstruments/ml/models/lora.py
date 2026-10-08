"""Rung R3 (ML_ENGINEERING §4.2): LoRA adapters on the top transformer blocks + the same
layer-weighted linear head as the probe, trained end to end on synthetic audio.

Targets:
- **MERT-95M:** q/k/v/out projections of encoder layers 8–11 (top third). Hidden states
  are captured by hooks, as in `embed.py`.
- **CLAP:** query/key/value of HTSAT stage 3 (and the last 4 blocks of stage 2).
  Hidden states come from `output_hidden_states`.

Only the LoRA matrices and the head train; the backbone stays frozen. Training crops 5 s
windows (MERT's pretraining length; half the cost), and eval uses the full 10 s.
Licence note: a MERT LoRA inherits MERT's CC-BY-NC weights (research only); a CLAP LoRA
stays Apache-2.0.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class LoraConfigM3:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    lr_lora: float = 3e-4
    lr_head: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 3
    batch_size: int = 8
    train_seconds: float = 5.0
    max_pos_weight: float = 20.0
    seed: int = 0


TARGETS = {
    "mert95m": r"encoder\.layers\.(8|9|10|11)\.attention\.(q_proj|k_proj|v_proj|out_proj)",
    "clap": r"audio_model\.audio_encoder\.layers\.(3\.blocks\.\d+|2\.blocks\.(8|9|10|11))\.attention\.self\.(query|key|value)",
}


class Backbone:
    """Trainable feature extractor: list of mono clips at `sr` -> (B, L, D) pooled layers."""

    def __init__(self, name: str, dev: str):
        import torch
        self.name, self.dev = name, dev
        if name == "mert95m":
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            hf = "m-a-p/MERT-v1-95M"
            self.model = AutoModel.from_pretrained(hf, trust_remote_code=True, local_files_only=True)
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(hf, trust_remote_code=True, local_files_only=True)
            self.sr = 24000
            # HuBERT LayerDrop randomly skips layers in train mode (a pretraining
            # regularizer), which breaks the fixed 13-layer feature stack: disable it.
            self.model.config.layerdrop = 0.0
            if hasattr(self.model.encoder, "layerdrop"):
                self.model.encoder.layerdrop = 0.0
            self._cap: list = []
            enc = self.model.encoder
            enc.layers[0].register_forward_pre_hook(lambda m, a: self._cap.append(a[0]))
            for layer in enc.layers:
                layer.register_forward_hook(lambda m, a, o: self._cap.append(o[0] if isinstance(o, tuple) else o))
        elif name == "clap":
            from transformers import ClapModel, ClapProcessor
            hf = "laion/larger_clap_music"
            self.model = ClapModel.from_pretrained(hf, use_safetensors=False, local_files_only=True)
            self.proc = ClapProcessor.from_pretrained(hf, local_files_only=True)
            self.sr = 48000
        else:
            raise ValueError(name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.torch = torch

    def add_lora(self, cfg: LoraConfigM3) -> int:
        from peft import LoraConfig, inject_adapter_in_model
        lc = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                        target_modules=TARGETS[self.name])
        self.model = inject_adapter_in_model(lc, self.model)
        n = 0
        for nm, p in self.model.named_parameters():
            if "lora_" in nm:
                p.requires_grad_(True)
                n += p.numel()
        self.model.to(self.dev)
        if n == 0:
            raise RuntimeError(f"LoRA matched no modules for {self.name}")
        return n

    def lora_params(self):
        return [p for n, p in self.model.named_parameters() if "lora_" in n]

    def features(self, batch: Sequence[np.ndarray]):
        torch = self.torch
        if self.name == "mert95m":
            self._cap.clear()
            x = self.fe(list(batch), sampling_rate=self.sr, return_tensors="pt", padding=True)
            self.model(**{k: v.to(self.dev) for k, v in x.items()})
            return torch.stack(self._cap, dim=1).mean(dim=2)                  # (B, 13, 768)
        x = self.proc(audio=list(batch), sampling_rate=self.sr, return_tensors="pt")
        x = {k: v.to(self.dev) for k, v in x.items()}
        a = self.model.audio_model(**x, output_hidden_states=True)
        layers = [h.flatten(2).mean(dim=-1) if h.dim() == 4 else h.mean(dim=1) for h in a.hidden_states]
        proj = self.model.audio_projection(a.pooler_output)
        d = max(max(l.shape[-1] for l in layers), proj.shape[-1])
        pad = lambda t: torch.nn.functional.pad(t, (0, d - t.shape[-1]))  # noqa: E731
        return torch.stack([pad(l) for l in layers] + [pad(proj)], dim=1)   # (B, 6, 1024)


class Head:
    def __init__(self, n_layers: int, dim: int, n_out: int, dev: str):
        import torch
        self.layer_logits = torch.zeros(n_layers, device=dev, requires_grad=True)
        self.norm = torch.nn.LayerNorm(dim).to(dev)
        self.lin = torch.nn.Linear(dim, n_out).to(dev)

    def params(self):
        return [self.layer_logits, *self.norm.parameters(), *self.lin.parameters()]

    def __call__(self, h):
        import torch
        w = torch.softmax(self.layer_logits, 0)
        return self.lin(self.norm((h * w[None, :, None]).sum(dim=1)))


def _load(path: Path, sr: int, seconds: float, rng: np.random.Generator | None):
    from .embed import load_clip
    y = load_clip(path, sr, 10.0)
    if rng is None or seconds >= 10.0:
        return y
    n = int(sr * seconds)
    s = int(rng.integers(0, len(y) - n + 1))
    return y[s:s + n]


def _val_map(y, s, m) -> float:
    from ..eval import metrics as M
    ap, r = M.per_class_ap(y, s, m)
    mac = M.macro(ap, r)
    return float(mac.value) if mac.value is not None else 0.0


def predict(bb: Backbone, head: Head, paths: Sequence[Path], batch: int = 8) -> np.ndarray:
    import torch
    from concurrent.futures import ThreadPoolExecutor
    bb.model.eval()
    out = []
    with torch.no_grad(), ThreadPoolExecutor(4) as ex:
        for k in range(0, len(paths), batch):
            clips = list(ex.map(lambda p: _load(p, bb.sr, 10.0, None), paths[k:k + batch]))
            out.append(torch.sigmoid(head(bb.features(clips))).float().cpu().numpy())
    return np.concatenate(out)


def train(bb: Backbone, paths: Sequence[Path], Y: np.ndarray, Mk: np.ndarray,
          vpaths: Sequence[Path], Yv: np.ndarray, Mv: np.ndarray, cfg: LoraConfigM3,
          log=print, ckpt: Path | None = None):
    import torch
    from concurrent.futures import ThreadPoolExecutor

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_lora = bb.add_lora(cfg)
    with torch.no_grad():
        h0 = bb.features([np.zeros(int(bb.sr * 1.0), np.float32)])
    head = Head(h0.shape[1], h0.shape[2], Y.shape[1], bb.dev)
    Yt = torch.from_numpy(Y.astype(np.float32)); Mt = torch.from_numpy(Mk.astype(np.float32))
    pw = (((1 - Yt) * Mt).sum(0) / (Yt * Mt).sum(0).clamp_min(1)).clamp(1, cfg.max_pos_weight).to(bb.dev)
    opt = torch.optim.AdamW([{"params": bb.lora_params(), "lr": cfg.lr_lora},
                             {"params": head.params(), "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(paths) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_lora, cfg.lr_head],
                                                total_steps=steps, pct_start=0.1)
    hist, best, best_state = [], -1.0, None
    for ep in range(cfg.epochs):
        bb.model.train()
        t0, tot, order = time.time(), 0.0, rng.permutation(len(paths))
        with ThreadPoolExecutor(4) as ex:
            for k in range(0, len(order), cfg.batch_size):
                i = order[k:k + cfg.batch_size]
                seeds = rng.integers(0, 2 ** 31, size=len(i))
                clips = list(ex.map(lambda t: _load(paths[t[0]], bb.sr, cfg.train_seconds,
                                                    np.random.default_rng(t[1])), zip(i, seeds)))
                z = head(bb.features(clips))
                yb, mb = Yt[i].to(bb.dev), Mt[i].to(bb.dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    z, yb, weight=mb, pos_weight=pw, reduction="sum") / mb.sum().clamp_min(1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss.detach()) * len(i)
                if (k // cfg.batch_size) % 100 == 0:
                    log(f"  ep {ep} step {k // cfg.batch_size}/{len(order) // cfg.batch_size} "
                        f"loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)")
        sv = predict(bb, head, vpaths)
        v = _val_map(Yv, sv, Mv)
        hist.append({"epoch": ep, "train_loss": tot / len(paths), "val_map": v, "sec": round(time.time() - t0)})
        log(f"epoch {ep} loss {tot / len(paths):.4f} val_mAP {v:.4f} ({time.time() - t0:.0f}s)")
        if v > best:
            best = v
            best_state = ({n: p.detach().cpu().clone() for n, p in bb.model.named_parameters() if "lora_" in n},
                          [p.detach().cpu().clone() for p in head.params()])
            if ckpt:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"lora": best_state[0], "head": best_state[1], "cfg": asdict(cfg)}, ckpt)
    with torch.no_grad():                               # restore best epoch
        named = dict(bb.model.named_parameters())
        for n, t in best_state[0].items():
            named[n].copy_(t.to(bb.dev))
        for p, t in zip(head.params(), best_state[1]):
            p.copy_(t.to(bb.dev))
    if ckpt:
        ckpt.with_suffix(".json").write_text(json.dumps({"cfg": asdict(cfg), "history": hist,
                                                         "n_lora_params": n_lora}, indent=1))
    return head, hist, n_lora
