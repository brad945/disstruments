"""Per-taxonomy-level temperature scaling (ML_ENGINEERING §4.4, PRD F30, M3 C1).

For each taxonomy level l, one scalar T_l rescales logits: p' = sigmoid(logit(p) / T_l).
T_l is fit by minimizing masked BCE on a held-out calibration set. Temperature is monotone,
so rankings (AP) are unchanged; only the probabilities move. That is what makes F15's
fixed UI thresholds (0.30 / 0.60) mean something.

The experiment is *which data you calibrate on*: synthetic val (in-domain for training,
out-of-domain for real use) vs a held-out real split (OpenMIC train, never trained on).
Levels with no observed labels in the calibration set keep T = 1 and are reported.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _bce(z: np.ndarray, y: np.ndarray, m: np.ndarray, T: float) -> float:
    s = z / T
    # numerically stable BCE with logits, masked mean
    loss = np.maximum(s, 0) - s * y + np.log1p(np.exp(-np.abs(s)))
    return float((loss * m).sum() / max(1.0, m.sum()))


def fit_temperatures(scores: np.ndarray, y: np.ndarray, m: np.ndarray,
                     node_levels: Sequence[int], *, bounds=(0.05, 20.0)) -> dict[int, dict]:
    """-> {level: {"T": float, "n_obs": int, "nll_before": float, "nll_after": float}}."""
    from scipy.optimize import minimize_scalar

    z = _logit(np.asarray(scores, float))
    y = np.asarray(y, float)
    m = np.asarray(m, float)
    levels = np.asarray(node_levels)
    out: dict[int, dict] = {}
    for lv in sorted(set(levels.tolist())):
        cols = levels == lv
        mm = m[:, cols]
        n = int(mm.sum())
        if n == 0 or y[:, cols][mm > 0].min() == y[:, cols][mm > 0].max():
            out[lv] = {"T": 1.0, "n_obs": n, "fitted": False}
            continue
        f = lambda logT: _bce(z[:, cols], y[:, cols], mm, float(np.exp(logT)))  # noqa: E731
        r = minimize_scalar(f, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
        out[lv] = {"T": float(np.exp(r.x)), "n_obs": n, "fitted": True,
                   "nll_before": f(0.0), "nll_after": float(r.fun)}
    return out


def apply_temperatures(scores: np.ndarray, node_levels: Sequence[int],
                       temps: dict[int, dict]) -> np.ndarray:
    z = _logit(np.asarray(scores, float))
    T = np.array([temps.get(lv, {"T": 1.0})["T"] for lv in node_levels])
    return 1.0 / (1.0 + np.exp(-z / T[None, :]))
