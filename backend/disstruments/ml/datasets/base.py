"""Common record type + source-label mapping shared by all dataset loaders.

Label semantics (the part that must stay honest):
- `positive`: taxonomy nodes present, ancestor-closed.
- `observed`: nodes whose presence/absence is KNOWN. `observed - positive` = known
  negatives. Everything else is masked out of every metric.
- A source label mapped to an internal node (source coarser than our leaves) marks that
  subtree unknown, never negative. Unknown subtrees also make their ancestors unknown
  unless positive (if a child may be present, so may its parent).
- Dataset-wide `unobserved` subtrees (mappings.yaml, per dataset): a pure observation
  mask, unobserved on every record and stem of that dataset (`apply_unobserved`). Used
  where a source cannot label a node honestly at all (taxonomy 3.0.0: cymbals on
  MedleyDB/Slakh). Validated (`unobserved_roots`) so it can never strip a positive nor
  create label-dependent missingness: roots must be level-1 (no ancestors to leave
  observed-only-when-positive) and no rule of that dataset may target a node inside them.
- Unmapped source labels are never dropped: strict mode raises with the full list;
  non-strict records them on the record and masks every non-positive node, unless the
  loader documents a coarser fallback for that field (Slakh: unknown `plugin_name` falls
  back to the GM `inst_class` rule, whose own masking then applies).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..taxonomy import Taxonomy, TaxonomyError, get_taxonomy

MAPPINGS_PATH = Path(__file__).resolve().parent.parent / "mappings.yaml"
SPLITS = ("train", "val", "test")
ALL = "*"


class UnknownLabelsError(ValueError):
    """Strict mode: source labels with no mapping and no ignore entry."""

    def __init__(self, dataset: str, labels: Mapping[str, int]):
        self.dataset, self.labels = dataset, dict(labels)
        listing = ", ".join(f"{k!r} (x{v})" for k, v in sorted(self.labels.items()))
        super().__init__(f"{dataset}: {len(self.labels)} unmapped source label(s): {listing}. "
                         f"Add them to mappings.yaml ({dataset}.map or .ignore).")


# ---------------------------------------------------------------------- records
@dataclass(frozen=True)
class StemLabels:
    stem_id: str
    positive: frozenset[str]
    observed: frozenset[str]
    source_labels: tuple[str, ...]
    audio: Path | None = None
    bleed: bool | None = None       # MedleyDB `has_bleed` (track-level flag); None = unknown


@dataclass(frozen=True)
class Record:
    """One evaluable item (a multitrack or a clip)."""
    dataset: str
    item_id: str
    split: str                      # train | val | test (| omitted for Slakh redux)
    artist: str | None              # opaque grouping key for artist-disjointness; None = unknown
    positive: frozenset[str]
    observed: frozenset[str]
    source_labels: tuple[str, ...]  # provenance: raw labels as found in the source
    unknown_labels: tuple[str, ...] = ()   # unmapped labels (non-strict mode only)
    audio: dict[str, Path] = field(default_factory=dict)   # "mix" -> path, only existing files
    stems: tuple[StemLabels, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def negative(self) -> frozenset[str]:
        return self.observed - self.positive


@dataclass
class DatasetIndex:
    """All records of one dataset (every split) plus the loader config and stats."""
    name: str
    records: list[Record]
    config: dict[str, Any]          # hashable loader params (no absolute paths)
    stats: dict[str, Any] = field(default_factory=dict)

    def split(self, name: str | None) -> list[Record]:
        if name in (None, "all"):
            return list(self.records)
        return [r for r in self.records if r.split == name]

    def split_counts(self) -> dict[str, int]:
        return dict(Counter(r.split for r in self.records))


# ---------------------------------------------------------------------- mapping
@dataclass(frozen=True)
class Rule:
    nodes: tuple[str, ...]          # positive nodes (empty for ignore rules)
    unknown: tuple[str, ...]        # subtree roots made unknown; ("*",) = everything
    ignored: bool = False
    reason: str = ""
    stem_unknown: tuple[str, ...] = ()   # like `unknown`, but only for the carrying stem


@dataclass(frozen=True)
class Resolution:
    positive: frozenset[str]        # ancestor-closed
    unknown_roots: frozenset[str]
    unknown_labels: tuple[str, ...]
    ignored_labels: tuple[str, ...]
    mapped: frozenset[str] = frozenset()        # rule nodes before ancestor closure
    stem_unknown_roots: frozenset[str] = frozenset()


class LabelMap:
    """Source-label -> taxonomy rules for one vocabulary (validated on construction)."""

    def __init__(self, source: str, mapped: Mapping[str, Any] | None,
                 ignored: Mapping[str, Any] | None, taxonomy: Taxonomy):
        self.source, self.taxonomy = source, taxonomy
        self.rules: dict[str, Rule] = {}
        errors: list[str] = []
        for label, spec in (mapped or {}).items():
            try:
                self.rules[str(label)] = self._parse_map(spec)
            except (TypeError, ValueError, KeyError) as e:
                errors.append(f"{source}.map[{label!r}]: {e}")
        for label, spec in (ignored or {}).items():
            if str(label) in self.rules:
                errors.append(f"{source}: {label!r} is both mapped and ignored")
                continue
            try:
                self.rules[str(label)] = self._parse_ignore(spec)
            except (TypeError, ValueError, KeyError) as e:
                errors.append(f"{source}.ignore[{label!r}]: {e}")
        if errors:
            raise TaxonomyError("invalid mappings:\n  " + "\n  ".join(errors))

    def _nodes(self, value: Any) -> tuple[str, ...]:
        vals = [value] if isinstance(value, str) else list(value or [])
        for v in vals:
            if v != ALL and v not in self.taxonomy:
                raise KeyError(f"unknown node {v!r}")
        return tuple(vals)

    def _parse_map(self, spec: Any) -> Rule:
        if isinstance(spec, (str, list)):
            spec = {"node": spec}
        if not isinstance(spec, dict) or "node" not in spec:
            raise ValueError("expected node id, list, or {node: ...}")
        if set(spec) - {"node", "children_unknown", "unknown"}:
            raise ValueError(f"unexpected keys {sorted(set(spec) - {'node', 'children_unknown', 'unknown'})}")
        nodes = self._nodes(spec["node"])
        if not nodes or ALL in nodes:
            raise ValueError("node must be one or more taxonomy ids")
        unknown = list(self._nodes(spec.get("unknown")))
        if spec.get("children_unknown", True):
            unknown += [n for n in nodes if not self.taxonomy.is_leaf(n)]
        return Rule(nodes, tuple(dict.fromkeys(unknown)))

    def _parse_ignore(self, spec: Any) -> Rule:
        if not isinstance(spec, dict) or not spec.get("reason"):
            raise ValueError("ignore entries need {reason: ..., unknown: [...]}")
        if set(spec) - {"reason", "unknown", "stem_unknown"}:
            raise ValueError(f"unexpected keys {sorted(set(spec) - {'reason', 'unknown', 'stem_unknown'})}")
        return Rule((), self._nodes(spec.get("unknown")), ignored=True, reason=str(spec["reason"]),
                    stem_unknown=self._nodes(spec.get("stem_unknown")))

    def __contains__(self, label: object) -> bool:
        return label in self.rules

    def labels(self) -> frozenset[str]:
        return frozenset(self.rules)

    def mapped_nodes(self) -> dict[str, set[str]]:
        """node -> source labels resolving exactly to it (coverage bookkeeping)."""
        out: dict[str, set[str]] = defaultdict(set)
        for label, rule in self.rules.items():
            for n in rule.nodes:
                out[n].add(label)
        return out

    def resolve(self, labels: Iterable[str]) -> Resolution:
        pos: set[str] = set()
        unk: set[str] = set()
        stem_unk: set[str] = set()
        unknown_labels: list[str] = []
        ignored: list[str] = []
        for label in labels:
            rule = self.rules.get(label)
            if rule is None:
                unknown_labels.append(label)
                unk.add(ALL)            # something unmapped is present: mask non-positives
                continue
            if rule.ignored:
                ignored.append(label)
            pos.update(rule.nodes)
            unk.update(rule.unknown)
            stem_unk.update(rule.stem_unknown)
        return Resolution(self.taxonomy.close_upward(pos), frozenset(unk),
                          tuple(dict.fromkeys(unknown_labels)), tuple(dict.fromkeys(ignored)),
                          frozenset(pos), frozenset(stem_unk))


def observed_exhaustive(taxonomy: Taxonomy, positive: frozenset[str],
                        unknown_roots: Iterable[str]) -> frozenset[str]:
    """Observed set for exhaustively-annotated sources (MedleyDB, Slakh).

    Everything not positive is a known negative, except unknown subtrees and their
    ancestors. Positives are always observed.
    """
    roots = set(unknown_roots)
    if ALL in roots:
        return frozenset(positive)
    unknown = set(taxonomy.close_downward(roots)) | set(taxonomy.close_upward(roots))
    return frozenset(n for n in taxonomy.nodes if n not in unknown or n in positive)


# Rule tables per dataset section (tables not listed here default to `map` / `ignore`).
RULE_TABLES = {"openmic": ("classes",),
               "slakh": ("plugins", "plugins_ignore", "inst_class", "inst_class_ignore")}
_DEFAULT_RULE_TABLES = ("map", "ignore")


def _rule_targets(spec: Any) -> list[str]:
    """Every node id a map/ignore/classes entry names (node, unknown, stem_unknown)."""
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return [str(v) for v in spec]
    if isinstance(spec, dict):
        out: list[str] = []
        for key in ("node", "unknown", "stem_unknown"):
            out += _rule_targets(spec.get(key) or [])
        return out
    return []


def unobserved_roots(doc: Mapping[str, Any], dataset: str, taxonomy: Taxonomy) -> frozenset[str]:
    """Validated `<dataset>.unobserved` roots ({node: reason}) from a mappings doc.

    Rules (each violation is an error, all reported at once):
    - known node, non-empty reason;
    - level-1 root (parentless): a deeper root would leave its ancestors observed only
      when positive (`observed_exhaustive` closes unknown upward), i.e. label-dependent
      missingness the audit cannot see;
    - no rule table of the dataset may name a node inside an unobserved subtree (as
      `node`, `unknown` or `stem_unknown`): the mask never strips a positive, so a rule
      that could produce one there is a contradiction, and a redundant mask is a smell.
    """
    section = doc.get(dataset) or {}
    spec = section.get("unobserved") or {}
    if not isinstance(spec, dict):
        raise TaxonomyError(f"invalid mappings: {dataset}.unobserved must be {{node: reason}}")
    errors = []
    for n, why in spec.items():
        if n not in taxonomy:
            errors.append(f"{dataset}.unobserved[{n!r}]: unknown node")
            continue
        if not str(why or "").strip():
            errors.append(f"{dataset}.unobserved[{n!r}]: needs a reason")
        if taxonomy.parent(n) is not None:
            errors.append(f"{dataset}.unobserved[{n!r}]: must be a level-1 node (a deeper root "
                          f"makes its ancestors observed only when positive)")
    if not errors and spec:
        masked = taxonomy.close_downward(spec)
        for table in RULE_TABLES.get(dataset, _DEFAULT_RULE_TABLES):
            for label, rule in (section.get(table) or {}).items():
                hit = sorted({t for t in _rule_targets(rule) if t in masked})
                if hit:
                    errors.append(f"{dataset}.{table}[{label!r}] targets {hit}, inside "
                                  f"{dataset}.unobserved; use an ignore entry instead")
    if errors:
        raise TaxonomyError("invalid mappings:\n  " + "\n  ".join(errors))
    return frozenset(spec)


def apply_unobserved(taxonomy: Taxonomy, positive: frozenset[str], unknown_roots: Iterable[str],
                     unobserved: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(positive, observed) with the dataset-wide `unobserved` subtrees masked.

    A pure observation mask: positives are returned unchanged (positives always win, as
    for every other unknown root). `unobserved_roots` guarantees no rule of the dataset
    targets those subtrees, so in practice there are no positives inside them.
    """
    pos = frozenset(positive)
    return pos, observed_exhaustive(taxonomy, pos, set(unknown_roots) | set(unobserved))


@lru_cache(maxsize=4)
def _load_mappings_cached(path: str) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def load_mappings(path: Path | str | None = None, taxonomy: Taxonomy | None = None) -> tuple[dict, str]:
    """(mappings doc, sha256): a private copy. Checks the declared taxonomy major version
    and validates every dataset section's `unobserved` block (`unobserved_roots`)."""
    doc, sha = _load_mappings_cached(str(path or MAPPINGS_PATH))
    tax = taxonomy or get_taxonomy()
    declared = str(doc.get("taxonomy_version", ""))
    if declared.split(".")[0] != str(tax.major):
        raise TaxonomyError(f"mappings target taxonomy {declared}, loaded taxonomy is {tax.version}")
    key = (sha, tax.version, tuple(tax.nodes))
    if key not in _VALIDATED:                  # every dataset section, in one place
        for name, section in doc.items():
            if isinstance(section, dict):
                unobserved_roots(doc, name, tax)
        _VALIDATED.add(key)
    return copy.deepcopy(doc), sha             # callers may mutate; the cache must not change


_VALIDATED: set[tuple] = set()


def collect_unknown(dataset: str, counts: Counter, strict: bool) -> None:
    if counts and strict:
        raise UnknownLabelsError(dataset, counts)


# ---------------------------------------------------------------------- artists & splits
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "`": "'"})


def normalize_artist(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Grouping key: NFKC, casefold, unified quotes, collapsed whitespace, then aliases."""
    s = unicodedata.normalize("NFKC", str(name)).translate(_QUOTES).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    aliases = {normalize_artist(k): normalize_artist(v) for k, v in (aliases or {}).items()}
    return aliases.get(s, s)


def artist_disjoint_split(groups: Mapping[str, int], fractions: Mapping[str, float],
                          seed: int = 0) -> dict[str, str]:
    """Assign whole artist groups to splits, deterministically, balancing item counts.

    Greedy: groups sorted by size (desc; ties broken by a seeded hash), each assigned to
    the split with the largest remaining deficit vs. its target fraction. Deterministic
    for a given (group set, fractions, seed). NOT stable under adding new groups —
    reports store a split hash so any change is visible; pin with a split file if needed.
    """
    total_frac = sum(fractions.values())
    if total_frac <= 0 or any(f < 0 for f in fractions.values()):
        raise ValueError(f"bad split fractions {dict(fractions)}")
    total = sum(groups.values())
    target = {s: total * f / total_frac for s, f in fractions.items()}
    have = {s: 0 for s in fractions}

    def tiebreak(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()

    out: dict[str, str] = {}
    for g in sorted(groups, key=lambda g: (-groups[g], tiebreak(g))):
        # split order is part of the key so equal deficits resolve deterministically
        best = max(fractions, key=lambda s: (target[s] - have[s], -list(fractions).index(s)))
        out[g] = best
        have[best] += groups[g]
    return out


def artist_leakage(records: Iterable[Record]) -> dict[str, list[str]]:
    """Artists appearing in more than one split (records with artist=None are skipped)."""
    by_artist: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.artist is not None:
            by_artist[r.artist].add(r.split)
    return {a: sorted(s) for a, s in sorted(by_artist.items()) if len(s) > 1}


def split_hash(records: Sequence[Record]) -> str:
    """Hash of (item, split) assignments — changes whenever a split changes."""
    payload = sorted((r.item_id, r.split) for r in records)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def stratified_artist_split(items: Mapping[str, tuple[str, Iterable[str]]],
                            fractions: Mapping[str, float], seed: int = 0,
                            n_iter: int = 20000, min_count: int = 2) -> dict[str, str]:
    """Artist-grouped split that also balances strata (genre, per-leaf positives).

    `items`: item_id -> (group, strata labels). Whole groups move together, so the result
    is group-disjoint by construction. Starts from `artist_disjoint_split` and hill-climbs
    (seeded random group moves and swaps, accept only strict improvements) on
        sum over strata f with >= min_count items, splits s:
            ((count_fs - frac_s * count_f) / count_f)^2
      + 5 * sum_s ((n_s - frac_s * N) / N)^2          (item counts stay near target)
      + 1 per (f, s) with count_f >= 3 and count_fs == 0 (every split sees every
        stratum that has >= 3 items)
    Deterministic for (items, fractions, seed, n_iter). Used once to produce a pinned
    split file; loaders read the pin, they do not re-run this.
    """
    import numpy as np

    total_frac = sum(fractions.values())
    splits = list(fractions)
    frac = np.array([fractions[s] / total_frac for s in splits])
    groups = sorted({g for g, _ in items.values()})
    gi = {g: i for i, g in enumerate(groups)}
    feats = sorted({f for _, fs in items.values() for f in fs})
    fi = {f: i for i, f in enumerate(feats)}
    G = np.zeros((len(groups), len(feats) + 1))        # last column = item count
    for g, fs in items.values():
        G[gi[g], -1] += 1
        for f in set(fs):
            G[gi[g], fi[f]] += 1
    tot = G.sum(0)
    keep = np.r_[tot[:-1] >= min_count, True]
    w = np.where(keep, 1.0, 0.0)
    w[-1] = 5.0
    need = np.r_[tot[:-1] >= 3, False]

    def cost(C: np.ndarray) -> float:
        dev = (C - frac[:, None] * tot[None, :]) / np.maximum(tot, 1)[None, :]
        return float((w[None, :] * dev ** 2).sum() + ((C == 0) & need[None, :]).sum())

    start = artist_disjoint_split({g: int(G[gi[g], -1]) for g in groups}, fractions, seed)
    a = np.array([splits.index(start[g]) for g in groups])
    C = np.zeros((len(splits), G.shape[1]))
    for k in range(len(groups)):
        C[a[k]] += G[k]
    best = cost(C)
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        if rng.random() < 0.5:                         # move one group
            k = int(rng.integers(len(groups)))
            src, dst = a[k], int(rng.integers(len(splits)))
            if dst == src:
                continue
            C[src] -= G[k]; C[dst] += G[k]             # noqa: E702
            c = cost(C)
            if c < best - 1e-12:
                best, a[k] = c, dst
            else:
                C[dst] -= G[k]; C[src] += G[k]         # noqa: E702
        else:                                          # swap two groups
            k, j = (int(x) for x in rng.integers(len(groups), size=2))
            if a[k] == a[j]:
                continue
            sk, sj = a[k], a[j]
            C[sk] += G[j] - G[k]; C[sj] += G[k] - G[j]  # noqa: E702
            c = cost(C)
            if c < best - 1e-12:
                best, a[k], a[j] = c, sj, sk
            else:
                C[sk] -= G[j] - G[k]; C[sj] -= G[k] - G[j]  # noqa: E702
    return {item: splits[a[gi[g]]] for item, (g, _) in items.items()}


def check_unique_ids(dataset: str, ids: Iterable[str], where: str = "") -> None:
    """Raise on duplicate item ids (a duplicated item would be double-weighted)."""
    dup = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dup:
        raise ValueError(f"{dataset}: duplicate item ids{(' in ' + where) if where else ''}: {dup[:10]}")


def assert_no_leakage(dataset: str, records: Iterable[Record], what: str) -> None:
    """Raise if any artist spans splits (used for pinned/official splits)."""
    leak = artist_leakage(records)
    if leak:
        raise ValueError(f"{dataset}: {what} is not artist-disjoint; {len(leak)} artist(s) span "
                         f"splits, e.g. {dict(list(leak.items())[:5])}")
