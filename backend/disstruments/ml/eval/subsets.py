"""Node-subset scoring for small labelled test sets (golden set; docs/debriefs/golden.md).

On a 25-song set most nodes are unscoreable, so the primary metric is the mean AP over a
node list frozen in advance (trained-on nodes with enough positives AND negatives). This
module restricts harness arrays to that list and gives single-run artist-bootstrap CIs
drawn exactly like `harness.paired_bootstrap` (same sort, grouping and RNG stream), so a
model's CI, the chance CI and the paired delta CI use the same resamples.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..datasets.base import Record
from ..taxonomy import Taxonomy
from . import metrics as M
from .harness import EvalArrays, summarize


def eligible_nodes(records: Sequence[Record], taxonomy: Taxonomy, trained: Iterable[str], *,
                   min_pos: int = 3, min_neg: int = 3,
                   levels: Iterable[int] | None = None) -> list[str]:
    """Trained nodes with >= min_pos observed positives and >= min_neg observed negatives
    (taxonomy order), optionally only at `levels`."""
    trained, lv = set(trained), None if levels is None else set(levels)
    out = []
    for n in taxonomy.nodes:
        if n not in trained or (lv is not None and taxonomy.level(n) not in lv):
            continue
        obs = sum(n in r.observed for r in records)
        pos = sum(n in r.positive for r in records)
        if pos >= min_pos and obs - pos >= min_neg:
            out.append(n)
    return out


def restrict(a: EvalArrays, nodes: Sequence[str]) -> EvalArrays:
    """The same items and labels, scoring only `nodes` (all must have been scored)."""
    col = {n: j for j, n in enumerate(a.nodes)}
    missing = [n for n in nodes if n not in col]
    if missing:
        raise ValueError(f"nodes not scored in these arrays: {missing}")
    return EvalArrays(a.item_ids, list(nodes), a.y_full, a.m_full,
                      a.scores[:, [col[n] for n in nodes]], a.threshold, a.n_bins, a.groups)


def constant_arrays(a: EvalArrays) -> EvalArrays:
    """Chance reference: constant scores, so every node's AP equals its prevalence among the
    observed items, in the full set and in every bootstrap resample."""
    return EvalArrays(a.item_ids, a.nodes, a.y_full, a.m_full, np.zeros_like(a.scores),
                      a.threshold, a.n_bins, a.groups)


def expected_random_ap(n: int, k: int) -> float:
    """Exact E[AP] of a uniformly random ranking of n items with k positives:
    (k-1)/(n-1) + H_n (n-k) / (n(n-1)). Above prevalence k/n at small n."""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    if n == 1:
        return 1.0
    h = float(np.sum(1.0 / np.arange(1, n + 1)))
    return (k - 1) / (n - 1) + h * (n - k) / (n * (n - 1))


def _groups(a: EvalArrays) -> tuple[EvalArrays, list[np.ndarray]]:
    s = a.take(np.argsort(a.item_ids))          # paired_bootstrap sorts the same way
    rows: dict[str, list[int]] = {}
    for k, g in enumerate(s.group_keys()):
        rows.setdefault(g, []).append(k)
    return s, [np.asarray(v) for v in rows.values()]


def bootstrap_ci(a: EvalArrays, taxonomy: Taxonomy, metrics: Sequence[str], *,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 return_draws: bool = False) -> dict[str, dict[str, Any]]:
    """Single-run cluster (artist) bootstrap percentile CIs. Resamples where a metric is
    undefined are dropped and counted (`n_valid`), as in `paired_bootstrap`."""
    s, groups = _groups(a)
    point = summarize(s, taxonomy)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        r = summarize(s.take(np.concatenate([groups[g] for g in pick])), taxonomy)
        for k in metrics:
            if r.get(k) is not None:
                draws[k].append(r[k])
    out = {}
    for k in metrics:
        d = np.asarray(draws[k])
        out[k] = {"value": point.get(k),
                  "ci": ([float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]
                         if d.size else None),
                  "n_valid": int(d.size), "n_boot": n_boot, "seed": seed, "n_groups": len(groups),
                  "n_items": len(s.item_ids)}
        if return_draws:
            out[k]["draws"] = d
    return out


def per_node(a: EvalArrays, taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    """Per scored node: positives / observed, prevalence, exact random-ranking AP, AP."""
    tidx = taxonomy.index()
    out = {}
    for j, n in enumerate(a.nodes):
        y, m = a.y_full[:, tidx[n]], a.m_full[:, tidx[n]]
        k, obs = int((y & m).sum()), int(m.sum())
        reason = M.skip_reason(y, m)
        out[n] = {"level": taxonomy.level(n), "n_positive": k, "n_observed": obs,
                  "prevalence": k / obs if obs else None,
                  "random_ap": expected_random_ap(obs, k) if reason is None else None,
                  "ap": M.average_precision(y[m], a.scores[m, j]) if reason is None else None,
                  "skipped": reason}
    return out


def positive_ranks(a: EvalArrays, taxonomy: Taxonomy, node: str) -> list[dict[str, Any]]:
    """For a rare node: each positive item's score and its rank among the observed items
    (1 = highest; ties share the best rank)."""
    j = a.nodes.index(node)
    t = taxonomy.index()[node]
    m, y = a.m_full[:, t], a.y_full[:, t]
    obs_scores = a.scores[m, j]
    return [{"item_id": a.item_ids[i], "score": float(a.scores[i, j]),
             "rank": int(1 + np.sum(obs_scores > a.scores[i, j])), "n_observed": int(m.sum())}
            for i in np.nonzero(y & m)[0]]
