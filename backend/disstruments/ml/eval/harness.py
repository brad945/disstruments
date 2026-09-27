"""Eval harness core: records + predictions -> metrics report; baselines; promotion gate.

Scoring universe (two different ones, on purpose):
- Per-class metrics (AP, F1, ECE, Brier; `map_*`, `f1_*`, `ece*`) use the taxonomy nodes
  the predictions cover. With `propagate=True` (default, mirrors serving per
  ML_ENGINEERING §4.3) missing ancestor columns are synthesized as max over their
  predicted descendants and every parent is max-propagated, so a leaf-only model is still
  scored at parent level. A model that predicts fewer nodes is not penalized in mAP for
  the nodes it skips; `predictions.n_nodes_evaluated` says how many were scored.
- Hierarchical P/R/F1 always uses the FULL taxonomy (unpredicted nodes = not predicted),
  so it does penalize a model that never predicts deep nodes. Once predictions are
  ancestor-closed it equals micro-F1 over all nodes (`f1_micro_all`); the informative
  contrast is hier F1 vs. flat leaf-level micro-F1 (`f1_micro_leaf`).

F1 is reported at the run threshold (default 0.5) and at 0.6 (`*_t60`), the PRD F15
"confident" threshold above which the UI states a detection without a "?".

Every summary number is computed by `summarize()` from per-item arrays that `run` stores
next to the report, so `compare` can re-compute metrics on bootstrap resamples of items
(paired bootstrap CIs on deltas) with exactly the same code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..datasets.base import Record
from ..datasets.openmic import class_to_node
from ..taxonomy import Taxonomy
from . import metrics as M
from .predictions import Predictions

CONFIDENT_THRESHOLD = 0.6          # PRD F15: >= 0.60 shown as fact, 0.30-0.60 with "?"
DEFAULT_SECONDARIES = ("map_macro", "map_leaf", "f1_macro_leaf", "hier_f1", "ece", "ece_classwise")
MIN_POSITIVES = 3                  # map_leaf_min3: leaves with >= this many observed positives
MIN_VALID_FRACTION = 0.95          # bootstrap: below this share of valid resamples, no significance
REL_TOL = 0.02                     # mAP/F1-type: may regress by <= 2% of the incumbent value
ABS_TOL_CALIBRATION = 0.005        # ECE/Brier-type: may regress by <= 0.005 absolute
_EPS = 1e-9


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def default_primary(dataset: str | None) -> str:
    """OpenMIC is only labeled on the 20 classes; everything else is leaf-labeled."""
    return "openmic20_map" if dataset == "openmic" else "map_leaf"


def lower_is_better(metric: str) -> bool:
    return metric.startswith("ece") or metric.endswith("_ece") or metric.startswith("brier")


def tolerance_for(metric: str, incumbent_value: float, tolerance: float | None = None) -> float:
    """Allowed regression for a secondary. `tolerance` (if given) = absolute for all."""
    if tolerance is not None:
        return tolerance
    if lower_is_better(metric):
        return ABS_TOL_CALIBRATION
    return REL_TOL * abs(incumbent_value)


# ---------------------------------------------------------------------- labels
def label_items(records: Sequence[Record], unit: str = "mix",
                exclude_bleed: bool = False) -> list[tuple[str, frozenset, frozenset]]:
    """(item_id, positive, observed) per evaluable unit: whole item or each stem.

    `exclude_bleed` (stem unit only) drops stems of tracks flagged `has_bleed` (MedleyDB):
    their stem labels describe the intended source, but the audio also carries others.
    """
    if unit == "mix":
        if exclude_bleed:
            raise ValueError("exclude_bleed applies to unit='stem' only")
        out = [(r.item_id, r.positive, r.observed) for r in records]
    elif unit == "stem":
        out = [(f"{r.item_id}/{s.stem_id}", s.positive, s.observed) for r in records for s in r.stems
               if not (exclude_bleed and s.bleed)]
        if not out:
            raise ValueError("unit=stem but these records have no (non-bleed) stems")
    else:
        raise ValueError("unit must be 'mix' or 'stem'")
    ids = [i for i, _, _ in out]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate evaluation items (would be double-weighted): {dup[:10]}")
    return out


def label_matrices(items, taxonomy: Taxonomy) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx = taxonomy.index()
    y = np.zeros((len(items), len(taxonomy)), dtype=bool)
    m = np.zeros_like(y)
    for i, (_, pos, obs) in enumerate(items):
        y[i, [idx[n] for n in pos]] = True
        m[i, [idx[n] for n in obs]] = True
    if (y & ~m).any():
        raise AssertionError("positive label outside observed mask")
    return [it[0] for it in items], y, m


def item_groups(records: Sequence[Record], unit: str = "mix") -> dict[str, str]:
    """item_id -> bootstrap group: the artist, else the track (stems of a track stay together)."""
    def g(r: Record) -> str:
        return f"artist:{r.artist}" if r.artist is not None else f"track:{r.item_id}"
    if unit == "stem":
        return {f"{r.item_id}/{s.stem_id}": g(r) for r in records for s in r.stems}
    return {r.item_id: g(r) for r in records}


def labels_hash(items) -> str:
    return _sha([(i, sorted(p), sorted(o)) for i, p, o in sorted(items, key=lambda x: x[0])])


# ---------------------------------------------------------------------- scores
def complete_scores(nodes: Sequence[str], scores: np.ndarray,
                    taxonomy: Taxonomy) -> tuple[list[str], np.ndarray, list[str]]:
    """Add missing ancestor columns (max over present descendants), taxonomy-ordered."""
    present = set(nodes)
    needed = set(taxonomy.close_upward(nodes))
    synthesized = [n for n in taxonomy.nodes if n in needed - present]
    order = [n for n in taxonomy.nodes if n in needed]
    pos = {n: i for i, n in enumerate(nodes)}
    out = np.full((scores.shape[0], len(order)), -np.inf)
    for j, n in enumerate(order):
        if n in pos:
            out[:, j] = scores[:, pos[n]]
    out = taxonomy.max_propagate(out, order)
    return order, out, synthesized


# ---------------------------------------------------------------------- core computation
@dataclass
class EvalArrays:
    """Everything the summary is a function of (stored as `arrays.npz` in a run dir)."""
    item_ids: list[str]
    nodes: list[str]                # evaluated (scored) nodes, taxonomy order if propagated
    y_full: np.ndarray              # (N, |taxonomy|) bool
    m_full: np.ndarray              # (N, |taxonomy|) bool
    scores: np.ndarray              # (N, len(nodes)) float, after propagation
    threshold: float
    n_bins: int
    # bootstrap resampling unit per item: "artist:<a>", or "track:<id>" when the artist is
    # unknown. None (old arrays / hand-built) = fall back to the track prefix of item_ids.
    groups: list[str] | None = None

    def take(self, rows: np.ndarray) -> "EvalArrays":
        return EvalArrays([self.item_ids[i] for i in rows], self.nodes, self.y_full[rows],
                          self.m_full[rows], self.scores[rows], self.threshold, self.n_bins,
                          None if self.groups is None else [self.groups[i] for i in rows])

    def group_keys(self) -> list[str]:
        if self.groups is not None:
            return list(self.groups)
        return ["track:" + i.split("/")[0] for i in self.item_ids]

    def save(self, path: Path) -> None:
        extra = {} if self.groups is None else {"groups": np.array(self.groups)}
        np.savez_compressed(path, item_ids=np.array(self.item_ids), nodes=np.array(self.nodes),
                            y_full=self.y_full, m_full=self.m_full, scores=self.scores,
                            threshold=np.array(self.threshold), n_bins=np.array(self.n_bins), **extra)

    @classmethod
    def load(cls, path: Path) -> "EvalArrays":
        with np.load(path, allow_pickle=False) as z:
            return cls([str(x) for x in z["item_ids"]], [str(x) for x in z["nodes"]],
                       z["y_full"].astype(bool), z["m_full"].astype(bool),
                       z["scores"].astype(float), float(z["threshold"]), int(z["n_bins"]),
                       [str(x) for x in z["groups"]] if "groups" in z.files else None)


def _compute(a: EvalArrays, taxonomy: Taxonomy) -> dict[str, Any]:
    nodes, scores = a.nodes, a.scores
    tidx = taxonomy.index()
    cols_full = [tidx[n] for n in nodes]
    y, m = a.y_full[:, cols_full], a.m_full[:, cols_full]

    ap, ap_reasons = M.per_class_ap(y, scores, m)
    prf = M.prf_at_threshold(y, scores, m, a.threshold)
    prf60 = M.prf_at_threshold(y, scores, m, CONFIDENT_THRESHOLD)
    levels = sorted({taxonomy.level(n) for n in nodes})
    leaf_cols = [j for j, n in enumerate(nodes) if taxonomy.is_leaf(n)]
    # calibration: drop columns whose observation depends on the label (only seen positive)
    ece_ok = [j for j, r in enumerate(ap_reasons) if r not in (M.SKIP_UNOBSERVED, M.SKIP_NO_NEGATIVES)]

    anc = taxonomy.ancestor_matrix()
    hier = {}
    for tag, thr in (("", a.threshold), ("_t60", CONFIDENT_THRESHOLD)):
        pred_full = np.zeros_like(a.y_full)
        pred_full[:, cols_full] = scores >= thr
        hier[tag] = M.hierarchical_prf(a.y_full, pred_full, a.m_full, anc)

    per_level = {}
    for lv in levels:
        cols = [j for j, n in enumerate(nodes) if taxonomy.level(n) == lv]
        e = M.multilabel_ece(y, scores, m, [c for c in cols if c in ece_ok], a.n_bins)
        per_level[str(lv)] = {"map": M.macro(ap, ap_reasons, cols).as_dict(),
                              "f1_macro": M.macro_f1(prf, cols).as_dict(),
                              "ece": e["ece"], "ece_n": e["n"]}
    ece_all = M.multilabel_ece(y, scores, m, ece_ok, a.n_bins)
    ece_cw = M.classwise_ece(y, scores, m, ece_ok, a.n_bins)
    brier = M.brier(y, scores, m, ece_ok)
    map_all = M.macro(ap, ap_reasons)
    map_leaf = M.macro(ap, ap_reasons, leaf_cols)
    # companions: map_leaf is dominated by leaves with 1-2 positives on small test splits
    n_pos = (y & m).sum(0)
    map_leaf_min3 = M.macro(ap, ap_reasons, [j for j in leaf_cols if n_pos[j] >= MIN_POSITIVES])
    map_micro_leaf = M.micro_average_precision(y, scores, m, leaf_cols)
    f1_leaf = M.macro_f1(prf, leaf_cols)

    c2n = class_to_node(taxonomy)
    col_of = {n: j for j, n in enumerate(nodes)}
    om = {c: (None if n not in col_of or np.isnan(ap[col_of[n]]) else float(ap[col_of[n]]))
          for c, n in c2n.items()}
    om_vals = [v for v in om.values() if v is not None]
    # benchmark-comparable calibration: only the 20 OpenMIC-mapped columns (pooled ECE on
    # OpenMIC is diluted by derived negatives, e.g. all voice.* nodes when voice=0)
    om_cols = [col_of[n] for n in c2n.values() if n in col_of and col_of[n] in ece_ok]
    om_ece = M.multilabel_ece(y, scores, m, om_cols, a.n_bins)

    summary = {
        "map_macro": map_all.value, "map_leaf": map_leaf.value,
        "map_leaf_min3": map_leaf_min3.value, "map_micro_leaf": map_micro_leaf,
        "f1_macro_leaf": f1_leaf.value, "f1_micro_leaf": M.micro_prf(prf, leaf_cols)["f1"],
        "hier_f1": hier[""]["f1"], "hier_precision": hier[""]["precision"],
        "hier_recall": hier[""]["recall"],
        "f1_macro_leaf_t60": M.macro_f1(prf60, leaf_cols).value,
        "f1_micro_leaf_t60": M.micro_prf(prf60, leaf_cols)["f1"],
        "hier_f1_t60": hier["_t60"]["f1"],
        "ece": ece_all["ece"], "ece_classwise": ece_cw["ece"], "brier": brier["brier"],
        **{f"map_level_{lv}": per_level[str(lv)]["map"]["value"] for lv in map(str, levels)},
        **{f"ece_level_{lv}": per_level[str(lv)]["ece"] for lv in map(str, levels)},
        "openmic20_map": float(np.mean(om_vals)) if om_vals else None,
        "openmic20_ece": om_ece["ece"],
    }
    return {"summary": summary, "ap": ap, "ap_reasons": ap_reasons, "prf": prf, "y": y, "m": m,
            "leaf_cols": leaf_cols, "per_level": per_level, "hier": hier[""],
            "map_all": map_all, "map_leaf": map_leaf, "map_leaf_min3": map_leaf_min3,
            "f1_leaf": f1_leaf, "ece_all": ece_all,
            "ece_cw": ece_cw, "brier": brier, "ece_ok": ece_ok, "om": om, "om_vals": om_vals,
            "om_ece": om_ece}


def summarize(a: EvalArrays, taxonomy: Taxonomy) -> dict[str, float | None]:
    """The report `summary` block from per-item arrays (used for bootstrap resamples)."""
    return _compute(a, taxonomy)["summary"]


def evaluate(records: Sequence[Record], preds: Predictions, taxonomy: Taxonomy, *,
             threshold: float = 0.5, n_bins: int = 15, propagate: bool = True,
             allow_missing: bool = False, unit: str = "mix", exclude_bleed: bool = False,
             return_arrays: bool = False):
    """Score predictions against records. Returns the metrics part of the report
    (and the `EvalArrays` if `return_arrays`)."""
    warnings = preds.validate(taxonomy)
    ids = [r.item_id for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate record ids (would be double-weighted): {dup[:10]}")
    items = label_items(records, unit, exclude_bleed)
    item_ids, y_full, m_full = label_matrices(items, taxonomy)

    row = {i: k for k, i in enumerate(preds.item_ids)}
    missing = [i for i in item_ids if i not in row]
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} items have no predictions (e.g. {missing[:3]}); "
                         f"use allow_missing to evaluate the rest")
    keep = [k for k, i in enumerate(item_ids) if i in row]
    extra = len(set(preds.item_ids) - set(item_ids))
    y_full, m_full = y_full[keep], m_full[keep]
    raw = preds.scores[[row[item_ids[k]] for k in keep]]

    bad, checked = taxonomy.violations(raw, preds.nodes)
    nodes, scores, synthesized = list(preds.nodes), raw, []
    if propagate:
        nodes, scores, synthesized = complete_scores(preds.nodes, raw, taxonomy)
    groups = item_groups(records, unit)
    arrays = EvalArrays([item_ids[k] for k in keep], list(nodes), y_full, m_full,
                        np.asarray(scores, dtype=float), float(threshold), int(n_bins),
                        [groups[item_ids[k]] for k in keep])
    c = _compute(arrays, taxonomy)
    ap, ap_reasons, prf, y, m = c["ap"], c["ap_reasons"], c["prf"], c["y"], c["m"]

    per_node = {}
    for j, n in enumerate(nodes):
        per_node[n] = {
            "level": taxonomy.level(n), "leaf": taxonomy.is_leaf(n),
            "n_observed": int(m[:, j].sum()), "n_positive": int((y[:, j] & m[:, j]).sum()),
            "ap": None if np.isnan(ap[j]) else float(ap[j]),
            "precision": None if np.isnan(prf.precision[j]) else float(prf.precision[j]),
            "recall": None if np.isnan(prf.recall[j]) else float(prf.recall[j]),
            "f1": None if np.isnan(prf.f1[j]) else float(prf.f1[j]),
            "tp": int(prf.tp[j]), "fp": int(prf.fp[j]), "fn": int(prf.fn[j]),
            "skipped": ap_reasons[j], "synthesized": n in synthesized,
        }
    result = {
        "summary": c["summary"],
        "macro_detail": {"map_macro": c["map_all"].as_dict(), "map_leaf": c["map_leaf"].as_dict(),
                         "map_leaf_min3": {**c["map_leaf_min3"].as_dict(),
                                           "min_positives": MIN_POSITIVES},
                         "f1_macro_leaf": c["f1_leaf"].as_dict(),
                         "f1_micro_leaf": M.micro_prf(prf, c["leaf_cols"]),
                         "f1_micro_all": M.micro_prf(prf),
                         "hier": c["hier"]},
        "per_level": c["per_level"],
        "per_node": per_node,
        "calibration": {"n_bins": n_bins, "pooled": c["ece_all"],
                        "classwise": c["ece_cw"], "brier": c["brier"],
                        "excluded_columns": len(nodes) - len(c["ece_ok"])},
        "openmic20": {"map": c["summary"]["openmic20_map"], "n_classes": len(c["om_vals"]),
                      "per_class_ap": c["om"], "ece": c["om_ece"]["ece"], "ece_n": c["om_ece"]["n"]},
        "items": {"n_items": len(item_ids), "n_evaluated": len(keep), "n_missing_predictions": len(missing),
                  "n_extra_predictions": extra, "unit": unit, "exclude_bleed": exclude_bleed,
                  "labels_hash": labels_hash([items[k] for k in keep]),   # kept items only
                  "mean_observed_fraction": float(m_full.mean()) if m_full.size else None},
        "predictions": {"n_nodes_given": len(preds.nodes), "n_nodes_evaluated": len(nodes),
                        "synthesized_nodes": synthesized,
                        "evaluated_nodes_hash": _sha(sorted(nodes)),
                        "consistency_violations_raw": {"n": bad, "of": checked},
                        "propagated": propagate, "threshold": threshold,
                        "confident_threshold": CONFIDENT_THRESHOLD,
                        "warnings": warnings, "meta": preds.meta},
    }
    return (result, arrays) if return_arrays else result


# ---------------------------------------------------------------------- baselines
def prior_scores(train_records: Iterable[Record], taxonomy: Taxonomy, alpha: float = 1.0,
                 unit: str = "mix") -> np.ndarray:
    """Per-node Laplace-smoothed positive rate over OBSERVED training labels (K,)."""
    items = label_items(list(train_records), unit)
    _, y, m = label_matrices(items, taxonomy)
    return ((y & m).sum(0) + alpha) / (m.sum(0) + 2 * alpha)


def baseline_predictions(kind: str, item_ids: Sequence[str], taxonomy: Taxonomy, *,
                         train_records: Sequence[Record] | None = None, seed: int = 0,
                         unit: str = "mix") -> np.ndarray:
    """(N, K) scores over all taxonomy nodes. kind: 'prior' (label prior) or 'random'."""
    if kind == "prior":
        if not train_records:
            raise ValueError("prior baseline needs training records")
        return np.tile(prior_scores(train_records, taxonomy, unit=unit), (len(item_ids), 1))
    if kind == "random":
        return np.random.default_rng(seed).random((len(item_ids), len(taxonomy)))
    raise ValueError("baseline kind must be 'prior' or 'random'")


# ---------------------------------------------------------------------- bootstrap
def paired_bootstrap(cand: EvalArrays, inc: EvalArrays, taxonomy: Taxonomy,
                     metrics: Sequence[str], *, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Paired cluster bootstrap of metric deltas (candidate - incumbent).

    Both runs must have scored the same items with identical labels and groups; each
    resample draws GROUPS with replacement and recomputes BOTH summaries on the same rows
    (so item difficulty cancels). The group is the ARTIST (`EvalArrays.groups`; the track
    when the artist is unknown): all items of an artist (tracks, and stems of those
    tracks) move together, since they are correlated — resampling them independently
    makes CIs too narrow. Arrays without groups fall back to the track prefix of the item
    id. Percentile (1 - alpha) CI. Resamples where a metric is undefined on either side
    are dropped and counted (`n_valid`); `promotion_gate` refuses significance when
    n_valid / n_boot < MIN_VALID_FRACTION.
    """
    order_c = np.argsort(cand.item_ids)
    order_i = np.argsort(inc.item_ids)
    c, i = cand.take(order_c), inc.take(order_i)
    if c.item_ids != i.item_ids:
        raise ValueError("bootstrap: runs scored different item sets")
    if not (np.array_equal(c.y_full, i.y_full) and np.array_equal(c.m_full, i.m_full)):
        raise ValueError("bootstrap: runs have different labels/masks for the same items")
    if not c.item_ids:
        raise ValueError("bootstrap: no items")
    keys = c.group_keys()
    if keys != i.group_keys():
        raise ValueError("bootstrap: runs have different bootstrap groups (artists) for the "
                         "same items; re-run both with the current harness")
    group_rows: dict[str, list[int]] = {}
    for k, g in enumerate(keys):
        group_rows.setdefault(g, []).append(k)
    groups = [np.asarray(v) for v in group_rows.values()]
    point_c, point_i = summarize(c, taxonomy), summarize(i, taxonomy)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[g] for g in pick])
        sc, si = summarize(c.take(rows), taxonomy), summarize(i.take(rows), taxonomy)
        for k in metrics:
            if sc.get(k) is not None and si.get(k) is not None:
                draws[k].append(sc[k] - si[k])
    out = {}
    for k in metrics:
        d = np.asarray(draws[k])
        pc, pi = point_c.get(k), point_i.get(k)
        out[k] = {"delta": None if pc is None or pi is None else pc - pi,
                  "ci": ([float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]
                         if d.size else None),
                  "n_valid": int(d.size), "n_boot": n_boot, "alpha": alpha, "seed": seed,
                  "n_groups": len(groups), "n_items": len(c.item_ids)}
    return out


# ---------------------------------------------------------------------- promotion gate
def comparability_key(r: dict) -> dict[str, Any]:
    """Everything that must match for two reports to be comparable: the full eval config
    (threshold, bins, propagate, unit, loader settings incl. mappings sha) except the
    loader seed (the split itself is compared via its hash), plus items/labels/nodes and
    taxonomy version + sha. Missing fields compare as None on both sides."""
    cfg = dict(r.get("config") or {})
    if isinstance(cfg.get("loader"), dict):
        cfg["loader"] = {k: v for k, v in cfg["loader"].items() if k != "seed"}
    prov = r.get("provenance") or {}
    items = r.get("items") or {}
    return {"config": cfg,
            "labels_hash": items.get("labels_hash"), "n_evaluated": items.get("n_evaluated"),
            "evaluated_nodes_hash": (r.get("predictions") or {}).get("evaluated_nodes_hash"),
            "taxonomy_version": prov.get("taxonomy_version"),
            "taxonomy_sha256": prov.get("taxonomy_sha256"),
            "mappings_sha256": prov.get("mappings_sha256"),
            "split_hash": (r.get("dataset") or {}).get("split_hash")}


def promotion_gate(candidate: dict, incumbent: dict, *, primary: str | None = None,
                   secondaries: Sequence[str] = DEFAULT_SECONDARIES,
                   tolerance: float | None = None,
                   bootstrap: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """PRD §8.3: promote iff the candidate beats the incumbent on `primary` and no
    secondary regresses by more than its tolerance.

    - primary default: `openmic20_map` on OpenMIC, `map_leaf` elsewhere.
    - tolerance: per metric by default (mAP/F1-type: 2% of the incumbent value; ECE/Brier
      type: 0.005 absolute); a float = absolute tolerance for every secondary.
    - direction-aware: every `ece*`/`*_ece`/`brier*` metric is lower-is-better.
    - bootstrap (from `paired_bootstrap`): if given, the primary delta's CI must exclude 0
      in the improving direction (CLI `compare` always supplies it). If fewer than
      MIN_VALID_FRACTION of the resamples were valid, significance is unavailable
      (not significant; never promotes).
    Refuses (comparable=False) unless `comparability_key` matches.
    """
    kc, ki = comparability_key(candidate), comparability_key(incumbent)
    diff = []
    for k in kc:
        if k == "config":
            for ck in sorted(set(kc[k]) | set(ki[k])):
                if kc[k].get(ck) != ki[k].get(ck):
                    diff.append(f"config.{ck} differs: {kc[k].get(ck)!r} vs {ki[k].get(ck)!r}")
        elif kc[k] != ki[k]:
            diff.append(f"{k} differs: {kc[k]!r} vs {ki[k]!r}")
    if diff:
        return {"promote": False, "comparable": False, "reasons": diff}

    primary = primary or default_primary((candidate.get("config") or {}).get("dataset"))

    def get(r: dict, k: str):
        return (r.get("summary") or {}).get(k)

    reasons, deltas, skipped, allowed = [], {}, [], {}
    c, i = get(candidate, primary), get(incumbent, primary)
    if c is None or i is None:
        return {"promote": False, "comparable": False, "primary": primary,
                "reasons": [f"primary {primary} undefined"]}
    lower = lower_is_better(primary)
    deltas[primary] = c - i
    if not ((c < i - _EPS) if lower else (c > i + _EPS)):
        reasons.append(f"primary {primary}: {c:.4f} does not beat {i:.4f}")
    significance = None
    if bootstrap is not None:
        b = bootstrap.get(primary) or {}
        ci, n_valid, n_boot = b.get("ci"), b.get("n_valid") or 0, b.get("n_boot") or 0
        frac = n_valid / n_boot if n_boot else 0.0
        available = frac >= MIN_VALID_FRACTION - _EPS
        ok = available and ci is not None and ((ci[1] < 0) if lower else (ci[0] > 0))
        significance = {"metric": primary, "ci": ci, "n_valid": b.get("n_valid"), "significant": ok,
                        "available": available, "valid_fraction": frac}
        if not available:
            why = (f"only {n_valid}/{n_boot} bootstrap resamples had {primary} defined on both "
                   f"sides (< {MIN_VALID_FRACTION:.0%}): significance unavailable")
            significance["reason"] = why
            reasons.append(f"primary {primary}: {why}")
        elif not ok:
            reasons.append(f"primary {primary}: bootstrap {int((1 - b.get('alpha', 0.05)) * 100)}% CI "
                           f"of delta {ci} does not exclude 0")
    for k in secondaries:
        if k == primary:
            continue
        c2, i2 = get(candidate, k), get(incumbent, k)
        if c2 is None and i2 is None:
            skipped.append(k)               # undefined for both (e.g. leaf mAP on OpenMIC)
            continue
        if c2 is None or i2 is None:
            reasons.append(f"secondary {k} undefined for {'candidate' if c2 is None else 'incumbent'}")
            continue
        deltas[k] = c2 - i2
        regress = (c2 - i2) if lower_is_better(k) else (i2 - c2)
        allowed[k] = tolerance_for(k, i2, tolerance)
        if regress > allowed[k] + _EPS:
            reasons.append(f"secondary {k} regressed by {regress:.4f} > allowed {allowed[k]:.4f}")
    return {"promote": not reasons, "comparable": True, "primary": primary,
            "tolerance": tolerance if tolerance is not None else
            {"relative": REL_TOL, "calibration_absolute": ABS_TOL_CALIBRATION},
            "allowed_regression": allowed, "deltas": deltas, "significance": significance,
            "skipped_secondaries": skipped, "reasons": reasons}
