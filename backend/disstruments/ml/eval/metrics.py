"""Multi-label, hierarchical, masked evaluation metrics (numpy only).

Conventions for every function:
- `y`: (N, K) bool ground truth; `s`: (N, K) float scores; `m`: (N, K) bool mask —
  True where the label is KNOWN. Masked-out entries never count (OpenMIC is partial;
  coarse sources leave fine descendants unknown).
- Per-class metrics are undefined for a class with no observed entries, no observed
  positives, or no observed negatives (AP of an all-positive column is 1.0 regardless
  of the scores — a node only ever observed when present would inflate macro means).
  Such classes are SKIPPED: value NaN, reason recorded, and the skip counts are returned
  next to every macro average. Nothing is silently averaged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SKIP_UNOBSERVED = "unobserved"
SKIP_NO_POSITIVES = "no_positives"
SKIP_NO_NEGATIVES = "no_negatives"


def _check(y: np.ndarray, s: np.ndarray, m: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    m = np.ones_like(y, dtype=bool) if m is None else np.asarray(m, dtype=bool)
    if y.ndim != 2 or y.shape != s.shape or y.shape != m.shape:
        raise ValueError(f"shape mismatch y{y.shape} s{s.shape} m{m.shape}")
    if np.isnan(s[m]).any():
        raise ValueError("NaN scores in observed entries")
    return y, s, m


def skip_reason(y_col: np.ndarray, m_col: np.ndarray) -> str | None:
    """Why a class column cannot be scored (None = scoreable)."""
    if not m_col.any():
        return SKIP_UNOBSERVED
    pos = int(y_col[m_col].sum())
    if pos == 0:
        return SKIP_NO_POSITIVES
    if pos == int(m_col.sum()):
        return SKIP_NO_NEGATIVES
    return None


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AP = sum_n (R_n - R_{n-1}) P_n over distinct score thresholds (tie-aware).

    Same definition as sklearn's average_precision_score. Requires >= 1 positive.
    """
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        raise ValueError("average_precision needs at least one positive")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # last index of each tie group; `!=` (not diff) so tied +-inf scores stay one group
    last = np.r_[np.nonzero(s[1:] != s[:-1])[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@dataclass
class Macro:
    """A macro average with explicit skip accounting."""
    value: float | None
    n_classes: int
    n_skipped: int
    skipped: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"value": self.value, "n_classes": self.n_classes,
                "n_skipped": self.n_skipped, "skipped": dict(self.skipped)}


def macro(values: np.ndarray, reasons: list[str | None], cols: np.ndarray | None = None) -> Macro:
    idx = np.arange(len(values)) if cols is None else np.asarray(cols, dtype=int)
    rs = [reasons[i] for i in idx]
    ok = [i for i, r in zip(idx, rs) if r is None]
    skipped: dict[str, int] = {}
    for r in rs:
        if r is not None:
            skipped[r] = skipped.get(r, 0) + 1
    val = float(np.mean(values[ok])) if ok else None
    return Macro(val, len(ok), len(idx) - len(ok), skipped)


def per_class_ap(y, s, m=None) -> tuple[np.ndarray, list[str | None]]:
    """(AP per class with NaN for skipped classes, skip reason per class)."""
    y, s, m = _check(y, s, m)
    ap = np.full(y.shape[1], np.nan)
    reasons: list[str | None] = []
    for k in range(y.shape[1]):
        r = skip_reason(y[:, k], m[:, k])
        reasons.append(r)
        if r is None:
            ap[k] = average_precision(y[m[:, k], k], s[m[:, k], k])
    return ap, reasons


def mean_average_precision(y, s, m=None, cols=None) -> Macro:
    """Macro mAP over (a subset of) classes; skipped classes are counted, not averaged."""
    ap, reasons = per_class_ap(y, s, m)
    return macro(ap, reasons, cols)


def micro_average_precision(y, s, m=None, cols=None) -> float | None:
    """Micro-AP: one AP over all observed (item, class) pairs of `cols`, pooled.
    None if the pool has no positives. Weights classes by their observed counts."""
    y, s, m = _check(y, s, m)
    idx = np.arange(y.shape[1]) if cols is None else np.asarray(cols, dtype=int)
    if idx.size == 0:
        return None
    yy, ss, mm = y[:, idx], s[:, idx], m[:, idx]
    if not (yy & mm).any():
        return None
    return average_precision(yy[mm], ss[mm])


@dataclass
class PRF:
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    precision: np.ndarray       # NaN for skipped classes
    recall: np.ndarray
    f1: np.ndarray
    reasons: list[str | None]


def prf_at_threshold(y, s, m=None, threshold: float | np.ndarray = 0.5) -> PRF:
    """Per-class P/R/F1 with prediction = score >= threshold (scalar or per-class).

    Precision with no predicted positives is 0; F1 is 0 when P + R = 0.
    """
    y, s, m = _check(y, s, m)
    pred = s >= np.asarray(threshold, dtype=float)
    tp = np.sum(pred & y & m, axis=0)
    fp = np.sum(pred & ~y & m, axis=0)
    fn = np.sum(~pred & y & m, axis=0)
    reasons = [skip_reason(y[:, k], m[:, k]) for k in range(y.shape[1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r = tp / np.maximum(tp + fn, 1)
        f = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    bad = np.array([x is not None for x in reasons], dtype=bool)
    p, r, f = (np.where(bad, np.nan, v).astype(float) for v in (p, r, f))
    return PRF(tp, fp, fn, p, r, f, reasons)


def macro_f1(prf: PRF, cols=None) -> Macro:
    return macro(prf.f1, prf.reasons, cols)


def micro_prf(prf: PRF, cols=None) -> dict[str, float | None]:
    """Micro P/R/F1 pooling TP/FP/FN over (a subset of) classes, skipped ones included —
    their false alarms are real errors even when per-class F1 is undefined."""
    idx = slice(None) if cols is None else np.asarray(cols, dtype=int)
    tp, fp, fn = int(prf.tp[idx].sum()), int(prf.fp[idx].sum()), int(prf.fn[idx].sum())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    # F1 = 2TP / (2TP + FP + FN): defined (0.0) when nothing is predicted but positives
    # exist; None only when there is nothing to score at all.
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def hierarchical_prf(y, pred, m, ancestors: np.ndarray) -> dict[str, float | int | None]:
    """Hierarchical P/R/F1 (Kiritchenko et al., 2005), micro-averaged over samples.

    For each sample i: P^_i = ancestor-closure of the predicted set, T^_i = ancestor-
    closure of the true set, both restricted to observed nodes (m). Then
        hP = sum_i |P^_i & T^_i| / sum_i |P^_i|,  hR = sum_i |P^_i & T^_i| / sum_i |T^_i|,
        hF = 2 hP hR / (hP + hR).
    Predicting the right parent but wrong leaf earns partial credit through the shared
    ancestors. `ancestors[i, j]` = node j is an ancestor-or-self of node i (over the same
    K columns). Returns None for undefined ratios.
    """
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    m = np.asarray(m, dtype=bool)
    a = np.asarray(ancestors, dtype=np.int64)
    close = lambda x: (x.astype(np.int64) @ a) > 0      # noqa: E731
    ph, th = close(pred) & m, close(y) & m
    inter, npred, ntrue = int((ph & th).sum()), int(ph.sum()), int(th.sum())
    hp = inter / npred if npred else None
    hr = inter / ntrue if ntrue else None
    hf = 2 * inter / (npred + ntrue) if npred + ntrue else None      # == 2PR/(P+R)
    return {"precision": hp, "recall": hr, "f1": hf,
            "intersection": inter, "n_pred": npred, "n_true": ntrue}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Binary ECE with equal-width bins over [0, 1].

    ECE = sum_b (n_b / N) * |acc_b - conf_b|, acc_b = fraction positive in bin b,
    conf_b = mean probability in bin b; bin b = min(floor(p * B), B - 1) so p = 1.0
    lands in the last bin. Probabilities must be in [0, 1] (not logits).
    """
    p = np.asarray(probs, dtype=float).ravel()
    t = np.asarray(labels, dtype=bool).ravel()
    if p.shape != t.shape:
        raise ValueError("probs/labels shape mismatch")
    if not np.isfinite(p).all():
        raise ValueError("ECE needs finite probabilities (got NaN/inf)")
    if p.size and (p.min() < 0 or p.max() > 1):
        raise ValueError("ECE needs probabilities in [0, 1]")
    if p.size == 0:
        return {"ece": None, "n": 0, "bins": []}
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece, bins = 0.0, []
    for k in range(n_bins):
        sel = b == k
        n = int(sel.sum())
        if n == 0:
            continue
        conf, acc = float(p[sel].mean()), float(t[sel].mean())
        ece += n / p.size * abs(acc - conf)
        bins.append({"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": n, "conf": conf, "acc": acc})
    return {"ece": float(ece), "n": int(p.size), "bins": bins}


def multilabel_ece(y, s, m=None, cols=None, n_bins: int = 15) -> dict:
    """ECE pooling every observed (item, class) pair over the chosen classes.

    Each per-label sigmoid output is treated as a binary probability. Callers pass the
    columns to pool (e.g. one taxonomy level, per ML_ENGINEERING §4.4) and should exclude
    columns whose observation depends on the label (see module docstring).
    """
    y, s, m = _check(y, s, m)
    idx = np.arange(y.shape[1]) if cols is None else np.asarray(cols, dtype=int)
    sel = m[:, idx]
    return expected_calibration_error(s[:, idx][sel], y[:, idx][sel], n_bins)


def classwise_ece(y, s, m=None, cols=None, n_bins: int = 15) -> dict:
    """Classwise ECE (Nixon et al., 2019): binary ECE per class over its observed entries,
    averaged over classes. Unlike pooled ECE it is not dominated by the many easy
    negatives of rare classes (a constant label-prior predictor scores ~0 pooled but
    not classwise when priors are off per class)."""
    y, s, m = _check(y, s, m)
    idx = np.arange(y.shape[1]) if cols is None else np.asarray(cols, dtype=int)
    per = []
    for k in idx:
        sel = m[:, k]
        if sel.any():
            per.append(expected_calibration_error(s[sel, k], y[sel, k], n_bins)["ece"])
    return {"ece": float(np.mean(per)) if per else None, "n_classes": len(per)}


def brier(y, s, m=None, cols=None) -> dict:
    """Mean squared error of probabilities over observed (item, class) pairs."""
    y, s, m = _check(y, s, m)
    idx = np.arange(y.shape[1]) if cols is None else np.asarray(cols, dtype=int)
    sel = m[:, idx]
    p, t = s[:, idx][sel], y[:, idx][sel]
    if p.size and not np.isfinite(p).all():
        raise ValueError("Brier score needs finite probabilities")
    return {"brier": float(np.mean((p - t) ** 2)) if p.size else None, "n": int(p.size)}
