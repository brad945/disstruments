"""Predictions file format (model output consumed by the eval harness).

JSON (`.json`):
    {"format": "disstruments.predictions/v1",
     "taxonomy_version": "1.0.0",
     "dataset": "medleydb",                 # optional; checked if present
     "nodes": ["guitar", "guitar.acoustic", ...],   # any subset of taxonomy node ids
     "items": {"<item_id>": [score per node], ...},
     "meta": {"model": "...", "latency_s": 1.2, ...}}   # optional, copied into reports
NPZ (`.npz`): arrays `item_ids` (str), `nodes` (str), `scores` (float, N x K),
`taxonomy_version` (0-d str), optional `dataset` and `meta_json` (0-d str).

Scores are per-node probabilities in [0, 1] (sigmoid outputs, not logits). Item ids are
the dataset's ids (MedleyDB track id, OpenMIC sample_key, Slakh `TrackXXXXX`); for
stem-level eval use `<item_id>/<stem_id>`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..taxonomy import Taxonomy

FORMAT = "disstruments.predictions/v1"


@dataclass
class Predictions:
    item_ids: list[str]
    nodes: list[str]
    scores: np.ndarray                  # (N, K) float
    taxonomy_version: str
    dataset: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""
    path: str = ""

    def validate(self, taxonomy: Taxonomy) -> list[str]:
        """Raise on hard errors; return warnings (e.g. minor taxonomy version drift)."""
        errs, warns = [], []
        if self.scores.shape != (len(self.item_ids), len(self.nodes)):
            errs.append(f"scores shape {self.scores.shape} != ({len(self.item_ids)}, {len(self.nodes)})")
        if not self.nodes:
            errs.append("no nodes: predictions must cover at least one taxonomy node")
        if not self.item_ids:
            errs.append("no items")
        if len(set(self.item_ids)) != len(self.item_ids):
            errs.append("duplicate item ids")
        if len(set(self.nodes)) != len(self.nodes):
            errs.append("duplicate nodes")
        unknown = [n for n in self.nodes if n not in taxonomy]
        if unknown:
            errs.append(f"nodes not in taxonomy {taxonomy.version}: {unknown[:10]}")
        if not np.isfinite(self.scores).all():
            errs.append("non-finite scores")
        elif self.scores.size and (self.scores.min() < 0 or self.scores.max() > 1):
            errs.append("scores must be probabilities in [0, 1]")
        major = self.taxonomy_version.split(".")[0]
        if major != str(taxonomy.major):
            errs.append(f"predictions use taxonomy {self.taxonomy_version}, harness has {taxonomy.version}")
        elif self.taxonomy_version != taxonomy.version:
            warns.append(f"taxonomy version drift: predictions {self.taxonomy_version} vs {taxonomy.version}")
        if errs:
            raise ValueError("invalid predictions: " + "; ".join(errs))
        return warns


class _Pairs(dict):
    """JSON object that remembers its raw key/value pairs (duplicate keys included)."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.pairs = list(pairs)


def load_predictions(path: Path | str) -> Predictions:
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if p.suffix == ".json":
        doc = json.loads(raw, object_pairs_hook=_Pairs)
        if doc.get("format") != FORMAT:
            raise ValueError(f"{p}: format must be {FORMAT!r}")
        # keep duplicate item keys (json would silently keep the last) so validate() rejects them
        pairs = doc["items"].pairs if isinstance(doc["items"], _Pairs) else list(doc["items"].items())
        ids = [k for k, _ in pairs]
        scores = np.asarray([v for _, v in pairs], dtype=float).reshape(len(ids), len(doc["nodes"]))
        return Predictions(ids, list(doc["nodes"]), scores, str(doc["taxonomy_version"]),
                           doc.get("dataset"), doc.get("meta") or {}, sha, str(p))
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
            return Predictions([str(x) for x in z["item_ids"]], [str(x) for x in z["nodes"]],
                               np.asarray(z["scores"], dtype=float), str(z["taxonomy_version"]),
                               str(z["dataset"]) if "dataset" in z else None, meta, sha, str(p))
    raise ValueError(f"{p}: predictions must be .json or .npz")


def save_predictions(path: Path | str, item_ids: Sequence[str], nodes: Sequence[str],
                     scores: np.ndarray, taxonomy_version: str, dataset: str | None = None,
                     meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=float)
    if p.suffix == ".json":
        doc = {"format": FORMAT, "taxonomy_version": taxonomy_version, "dataset": dataset,
               "nodes": list(nodes),
               "items": {i: [round(float(x), 6) for x in row] for i, row in zip(item_ids, scores)},
               "meta": meta or {}}
        p.write_text(json.dumps(doc, indent=1))
    elif p.suffix == ".npz":
        extra = {"dataset": np.array(dataset)} if dataset else {}
        np.savez_compressed(p, item_ids=np.array(list(item_ids)), nodes=np.array(list(nodes)),
                            scores=scores, taxonomy_version=np.array(taxonomy_version),
                            meta_json=np.array(json.dumps(meta or {})), **extra)
    else:
        raise ValueError(f"{p}: predictions must be .json or .npz")
    return p
