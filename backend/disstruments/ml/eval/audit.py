"""Label-set audits (no predictions needed): real per-leaf coverage and label-dependent
missingness. Surfaced by `python -m disstruments.ml.eval coverage`.

Why missingness matters: a node's metrics only use the items where it is OBSERVED. If
whether a node is observed depends on its own label (v1: kit tracks made `percussion`
unknown unless a hi-hat mic made it positive), the observed subset over-represents
positives and AP/ECE for that node measure the masking rule, not the model. We cannot see
the hidden labels, so the audit uses the only observable symptom: the positive rate among
observed items vs. the rate over all items (positives are always observed, so the two
differ only through masking). Masking that is independent of the label also opens a gap,
but a small one; a large gap on a node masked on many items is the red flag.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from ..datasets.base import SPLITS, Record
from ..taxonomy import Taxonomy

# Flag when the observed-positive rate exceeds the overall rate by more than this AND the
# node is unobserved on at least MIN_UNOBSERVED_FRAC of items. Calibrated on public
# MedleyDB: v1 `percussion` = 0.56 vs 0.39 (gap 0.17, masked on 30% of tracks) must flag;
# label-independent masking (e.g. `fx/processed sound` -> keys.synth) must not.
MAX_RATE_GAP = 0.10
MIN_UNOBSERVED_FRAC = 0.10
MIN_POSITIVES = 5


def missingness(records: Sequence[Record], taxonomy: Taxonomy, *,
                max_gap: float = MAX_RATE_GAP, min_unobserved_frac: float = MIN_UNOBSERVED_FRAC,
                min_positives: int = MIN_POSITIVES) -> dict[str, dict[str, Any]]:
    """Per node: n, n_observed, n_positive, overall/observed positive rate, gap, flagged."""
    n = len(records)
    out: dict[str, dict[str, Any]] = {}
    if not n:
        return out
    for node in taxonomy.nodes:
        n_obs = sum(node in r.observed for r in records)
        n_pos = sum(node in r.positive for r in records)
        overall = n_pos / n
        observed = n_pos / n_obs if n_obs else None
        gap = (observed - overall) if observed is not None else None
        unobs_frac = 1 - n_obs / n
        # nodes never observed negative are skipped by every metric anyway (no_negatives)
        flagged = bool(gap is not None and n_pos >= min_positives and gap > max_gap
                       and unobs_frac >= min_unobserved_frac and n_obs > n_pos)
        out[node] = {"n": n, "n_observed": n_obs, "n_positive": n_pos,
                     "rate_overall": overall, "rate_observed": observed, "gap": gap,
                     "unobserved_frac": unobs_frac, "flagged": flagged}
    return out


def coverage(records: Sequence[Record], taxonomy: Taxonomy, dataset: str) -> dict[str, Any]:
    """Real positives per leaf per split (item level) + `data:` tag consistency.

    `claims_without_examples`: leaves whose taxonomy `data:` names `dataset` but that have
    zero positives in the loaded records (the tag over-promises). `examples_without_claim`:
    leaves with positives whose tag omits `dataset`.
    """
    splits = [s for s in SPLITS if any(r.split == s for r in records)]
    splits += sorted({r.split for r in records} - set(splits))
    per_leaf: dict[str, dict[str, int]] = {}
    for leaf in taxonomy.leaves():
        c = Counter(r.split for r in records if leaf in r.positive)
        per_leaf[leaf] = {s: c.get(s, 0) for s in splits} | {"total": sum(c.values())}
    claims = [l for l in taxonomy.leaves() if dataset in taxonomy.info[l].data]
    test = per_leaf and "test" in splits
    summary = {
        "n_items": len(records), "splits": {s: sum(r.split == s for r in records) for s in splits},
        "n_leaves": len(per_leaf),
        "leaves_with_any_positive": sum(v["total"] > 0 for v in per_leaf.values()),
        "leaves_test_ge1": sum(v.get("test", 0) >= 1 for v in per_leaf.values()) if test else None,
        "leaves_test_ge3": sum(v.get("test", 0) >= 3 for v in per_leaf.values()) if test else None,
    }
    return {
        "dataset": dataset, "summary": summary, "per_leaf": per_leaf,
        "claims_without_examples": [l for l in claims if per_leaf[l]["total"] == 0],
        "examples_without_claim": [l for l, v in per_leaf.items()
                                   if v["total"] > 0 and dataset not in taxonomy.info[l].data],
    }
