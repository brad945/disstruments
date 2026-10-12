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


def fit_platt(scores: np.ndarray, y: np.ndarray, m: np.ndarray, *,
              a_bounds=(1e-3, 50.0), b_bounds=(-50.0, 50.0)) -> list[dict]:
    """Per-column Platt scaling p' = sigmoid(a * logit(p) + b), fit by masked BCE.

    For the incumbent PANNs tagger (M5): its scores rank well but sit far below F15's fixed
    thresholds, and one temperature can't fix that (scaling a logit of -2 never crosses 0),
    so each class gets a bias too. `a > 0` keeps the map monotone: AP is unchanged.
    Columns without both classes, or with constant scores, keep the identity map."""
    from scipy.optimize import minimize

    z = _logit(np.asarray(scores, float))
    y = np.asarray(y, float)
    m = np.asarray(m, float)
    out = []
    for j in range(z.shape[1]):
        mm = m[:, j] > 0
        zj, yj = z[mm, j], y[mm, j]
        n, n_pos = int(mm.sum()), int(yj.sum())
        if n_pos == 0 or n_pos == n or np.ptp(zj) == 0:
            out.append({"a": 1.0, "b": 0.0, "n_obs": n, "n_pos": n_pos, "fitted": False})
            continue
        ones = np.ones_like(yj)
        f = lambda ab: _bce(ab[0] * zj + ab[1], yj, ones, 1.0)  # noqa: E731
        r = minimize(f, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[a_bounds, b_bounds])
        out.append({"a": float(r.x[0]), "b": float(r.x[1]), "n_obs": n, "n_pos": n_pos,
                    "fitted": True, "nll_before": f([1.0, 0.0]), "nll_after": float(r.fun)})
    return out


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(np.asarray(p, float)) + b)))
