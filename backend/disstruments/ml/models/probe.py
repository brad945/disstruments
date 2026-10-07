"""Linear probe on frozen embeddings (M2 S6; ML_ENGINEERING §4.2 rung 1).

Model: softmax-weighted sum over the cached layers (L learnable scalars) -> per-layer
feature standardization (train stats) -> one linear layer to every taxonomy node
(sigmoid, multi-label). Loss: BCE masked by `observed` (partial labels never count as
negatives) with per-node positive weighting for class balance. Early stopping on the
validation macro AP (same masked AP the harness reports). Hierarchy consistency is applied
at evaluation by the harness (max-propagation, §4.3), not baked into training.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..eval import metrics as M


@dataclass
class ProbeConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    patience: int = 15
    max_pos_weight: float = 20.0
    seed: int = 0


def _val_map(y: np.ndarray, s: np.ndarray, m: np.ndarray) -> float:
    ap, reasons = M.per_class_ap(y, s, m)
    mac = M.macro(ap, reasons)
    return float(mac.value) if mac.value is not None else 0.0


class LinearProbe:
    def __init__(self, n_layers: int, dim: int, n_out: int, cfg: ProbeConfig):
        import torch

        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.layer_logits = torch.zeros(n_layers, requires_grad=True)
        self.head = torch.nn.Linear(dim, n_out)
        self.mu = torch.zeros(n_layers, dim)
        self.sd = torch.ones(n_layers, dim)
        self.history: list[dict] = []

    def _feats(self, x):
        import torch
        x = (x - self.mu) / self.sd
        w = torch.softmax(self.layer_logits, dim=0)
        return (x * w[None, :, None]).sum(dim=1)

    def logits(self, x):
        return self.head(self._feats(x))

    def fit(self, X: np.ndarray, Y: np.ndarray, Mk: np.ndarray,
            Xv: np.ndarray, Yv: np.ndarray, Mv: np.ndarray) -> "LinearProbe":
        import torch

        cfg = self.cfg
        Xt = torch.from_numpy(X.astype(np.float32))
        self.mu = Xt.mean(dim=0)
        self.sd = Xt.std(dim=0).clamp_min(1e-4)
        Yt, Mt = torch.from_numpy(Y.astype(np.float32)), torch.from_numpy(Mk.astype(np.float32))
        pos = (Yt * Mt).sum(0)
        neg = ((1 - Yt) * Mt).sum(0)
        pw = (neg / pos.clamp_min(1)).clamp(1.0, cfg.max_pos_weight)
        params = [self.layer_logits, *self.head.parameters()]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        g = torch.Generator().manual_seed(cfg.seed)
        Xvt = torch.from_numpy(Xv.astype(np.float32))
        best, best_state, bad = -1.0, None, 0
        for ep in range(cfg.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for k in range(0, len(Xt), cfg.batch_size):
                i = perm[k:k + cfg.batch_size]
                z = self.logits(Xt[i])
                l = torch.nn.functional.binary_cross_entropy_with_logits(
                    z, Yt[i], weight=Mt[i], pos_weight=pw, reduction="sum") / Mt[i].sum().clamp_min(1)
                opt.zero_grad()
                l.backward()
                opt.step()
                tot += float(l) * len(i)
            with torch.no_grad():
                sv = torch.sigmoid(self.logits(Xvt)).numpy()
            v = _val_map(Yv, sv, Mv)
            self.history.append({"epoch": ep, "train_loss": tot / len(Xt), "val_map": v})
            if v > best + 1e-4:
                best, bad = v, 0
                best_state = (self.layer_logits.detach().clone(),
                              {k: t.detach().clone() for k, t in self.head.state_dict().items()})
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            with torch.no_grad():
                self.layer_logits.copy_(best_state[0])
            self.head.load_state_dict(best_state[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.logits(torch.from_numpy(X.astype(np.float32)))).numpy()

    def layer_weights(self) -> list[float]:
        import torch
        return torch.softmax(self.layer_logits.detach(), 0).tolist()

    def save(self, path: Path, extra: dict | None = None) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"layer_logits": self.layer_logits.detach(), "head": self.head.state_dict(),
                    "mu": self.mu, "sd": self.sd, "cfg": asdict(self.cfg)}, path)
        path.with_suffix(".json").write_text(json.dumps(
            {"cfg": asdict(self.cfg), "layer_weights": self.layer_weights(),
             "history": self.history, **(extra or {})}, indent=1))
