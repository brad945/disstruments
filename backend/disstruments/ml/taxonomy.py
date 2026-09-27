"""Hierarchical instrument taxonomy (PRD F27, ML_ENGINEERING §2).

Loads and validates `taxonomy.yaml`: dotted node ids (`guitar.electric.distorted`),
parent implied by the id prefix, max depth 3. Provides tree queries and the two
consistency helpers the rest of the ML stack relies on:

- `close_upward`: ancestor closure of a label set (a distorted guitar IS a guitar).
- `max_propagate`: score consistency (parent := max(parent, children)), so a child
  never outranks its parent — the inference-time rule of ML_ENGINEERING §4.3.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.yaml")
MAX_DEPTH = 3
KNOWN_SOURCES = frozenset({"medleydb", "openmic", "slakh"})
PLANNED_SOURCES = frozenset({"synthetic", "golden"})
_SEGMENT = re.compile(r"^[a-z][a-z0-9_]*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class TaxonomyError(ValueError):
    """Invalid taxonomy/mapping file. Message lists every problem found."""


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    parent: str | None
    level: int                              # 1 = family
    data: tuple[str, ...] = ()              # real sources reaching this leaf exactly
    planned: tuple[str, ...] = ()           # future sources (synthetic renderer)
    note: str = ""


@dataclass(frozen=True)
class Taxonomy:
    version: str
    nodes: tuple[str, ...]                  # canonical (pre-order) column order
    info: dict[str, Node]
    sha256: str
    _children: dict[str, tuple[str, ...]] = field(repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path | str | None = None) -> "Taxonomy":
        p = Path(path) if path else TAXONOMY_PATH
        raw = p.read_bytes()
        return cls.from_dict(yaml.safe_load(raw), sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_dict(cls, doc: dict, sha256: str = "") -> "Taxonomy":
        if not isinstance(doc, dict):
            raise TaxonomyError(f"invalid taxonomy: expected a mapping, got {type(doc).__name__}")
        if not isinstance(doc.get("nodes") or [], list):
            raise TaxonomyError("invalid taxonomy: `nodes` must be a list")
        errors: list[str] = []
        version = str(doc.get("version", ""))
        if not _SEMVER.match(version):
            errors.append(f"version must be semver X.Y.Z, got {version!r}")
        ids: list[str] = []
        info: dict[str, Node] = {}
        for i, entry in enumerate(doc.get("nodes") or []):
            if not isinstance(entry, dict):
                errors.append(f"node #{i}: expected a mapping with `id`, got {entry!r}")
                continue
            nid = str(entry.get("id", ""))
            segs = nid.split(".")
            if not all(_SEGMENT.match(s) for s in segs):
                errors.append(f"node #{i}: bad id {nid!r}")
                continue
            if nid in info:
                errors.append(f"duplicate node {nid}")
                continue
            if len(segs) > MAX_DEPTH:
                errors.append(f"{nid}: depth {len(segs)} > {MAX_DEPTH}")
            parent = ".".join(segs[:-1]) or None
            if parent is not None and parent not in info:
                errors.append(f"{nid}: parent {parent} missing or listed after child")
            data = tuple(entry.get("data") or ())
            planned = tuple(entry.get("planned") or ())
            if set(data) - KNOWN_SOURCES:
                errors.append(f"{nid}: unknown data sources {sorted(set(data) - KNOWN_SOURCES)}")
            if set(planned) - PLANNED_SOURCES:
                errors.append(f"{nid}: unknown planned sources {sorted(set(planned) - PLANNED_SOURCES)}")
            info[nid] = Node(nid, str(entry.get("name") or nid), parent, len(segs),
                             data, planned, str(entry.get("note") or ""))
            ids.append(nid)
        children: dict[str, list[str]] = {n: [] for n in ids}
        for n in ids:
            if info[n].parent in children:
                children[info[n].parent].append(n)
        for n in ids:
            node = info[n]
            if children[n] and (node.data or node.planned):
                errors.append(f"{n}: coverage (data/planned) is only allowed on leaves")
            if not children[n] and not (node.data or node.planned):
                errors.append(f"{n}: leaf has no data source and no planned source (rule b)")
        if not ids:
            errors.append("no nodes")
        if errors:
            raise TaxonomyError("invalid taxonomy:\n  " + "\n  ".join(errors))
        return cls(version, tuple(ids), info, sha256,
                   {k: tuple(v) for k, v in children.items()})

    # ------------------------------------------------------------------ queries
    def __contains__(self, node: object) -> bool:
        return node in self.info

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])

    def children(self, node: str) -> tuple[str, ...]:
        return self._children[self._check(node)]

    def is_leaf(self, node: str) -> bool:
        return not self.children(node)

    def level(self, node: str) -> int:
        return self.info[self._check(node)].level

    def parent(self, node: str) -> str | None:
        return self.info[self._check(node)].parent

    def ancestors(self, node: str, include_self: bool = False) -> tuple[str, ...]:
        """Root-first ancestors of `node`."""
        segs = self._check(node).split(".")
        out = tuple(".".join(segs[:i]) for i in range(1, len(segs)))
        return out + (node,) if include_self else out

    def descendants(self, node: str, include_self: bool = False) -> tuple[str, ...]:
        """Pre-order descendants of `node`."""
        out: list[str] = [node] if include_self else []
        for c in self.children(node):
            out.extend(self.descendants(c, include_self=True))
        return tuple(out)

    def leaves(self) -> tuple[str, ...]:
        return tuple(n for n in self.nodes if not self._children[n])

    def roots(self) -> tuple[str, ...]:
        return tuple(n for n in self.nodes if self.info[n].parent is None)

    def nodes_at_level(self, level: int) -> tuple[str, ...]:
        return tuple(n for n in self.nodes if self.info[n].level == level)

    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.nodes)}

    # ------------------------------------------------------------------ consistency
    def close_upward(self, nodes: Iterable[str]) -> frozenset[str]:
        """Ancestor closure: every node plus all of its ancestors."""
        out: set[str] = set()
        for n in nodes:
            out.update(self.ancestors(n, include_self=True))
        return frozenset(out)

    def close_downward(self, nodes: Iterable[str]) -> frozenset[str]:
        """Subtree closure: every node plus all of its descendants."""
        out: set[str] = set()
        for n in nodes:
            out.update(self.descendants(n, include_self=True))
        return frozenset(out)

    def ancestor_matrix(self, nodes: Sequence[str] | None = None) -> np.ndarray:
        """A[i, j] = nodes[j] is an ancestor-or-self of nodes[i] (restricted to `nodes`)."""
        nodes = tuple(nodes) if nodes is not None else self.nodes
        pos = {n: i for i, n in enumerate(nodes)}
        a = np.zeros((len(nodes), len(nodes)), dtype=bool)
        for i, n in enumerate(nodes):
            for anc in self.ancestors(n, include_self=True):
                if anc in pos:
                    a[i, pos[anc]] = True
        return a

    def max_propagate(self, scores: np.ndarray, nodes: Sequence[str] | None = None) -> np.ndarray:
        """Return scores with parent := max(parent, children), over the columns present.

        `scores` is (N, K) aligned to `nodes` (default: all taxonomy nodes). Processed
        deepest-first so the max flows all the way up. Columns whose parent is absent
        are left as-is.
        """
        nodes = tuple(nodes) if nodes is not None else self.nodes
        pos = {n: i for i, n in enumerate(nodes)}
        out = np.array(scores, dtype=float, copy=True)
        for n in sorted(nodes, key=lambda x: -self.level(x)):
            # nearest ancestor that is present (skips missing intermediate columns)
            for anc in reversed(self.ancestors(n)):
                if anc in pos:
                    out[:, pos[anc]] = np.maximum(out[:, pos[anc]], out[:, pos[n]])
                    break
        return out

    def violations(self, scores: np.ndarray, nodes: Sequence[str] | None = None,
                   eps: float = 1e-9) -> tuple[int, int]:
        """(#child>parent violations, #checked pairs) over present parent/child columns."""
        nodes = tuple(nodes) if nodes is not None else self.nodes
        pos = {n: i for i, n in enumerate(nodes)}
        bad = total = 0
        for n in nodes:
            p = self.info[n].parent
            if p in pos:
                bad += int(np.sum(scores[:, pos[n]] > scores[:, pos[p]] + eps))
                total += scores.shape[0]
        return bad, total

    def render(self) -> str:
        """Indented tree, leaves marked with their coverage."""
        lines = []
        for n in self.nodes:
            node = self.info[n]
            tag = ""
            if not self._children[n]:
                cov = list(node.data) + [f"{p}*" for p in node.planned]
                tag = f"  [{', '.join(cov)}]"
            lines.append(f"{'  ' * (node.level - 1)}{n.split('.')[-1]}{tag}")
        return "\n".join(lines)

    def _check(self, node: str) -> str:
        if node not in self.info:
            raise KeyError(f"unknown taxonomy node {node!r}")
        return node


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    """The packaged taxonomy (cached)."""
    return Taxonomy.load()
