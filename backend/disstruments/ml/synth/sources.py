"""Source registry (`sources.yaml`, M2 S0): which sample library / synth renders which leaf,
under which licence. Every rendered clip records the ids (and licences) of its sources, so
a commercially clean subset can always be filtered out (docs/DIRECTION_NOTES.md)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..taxonomy import Taxonomy

REGISTRY_PATH = Path(__file__).with_name("sources.yaml")
KINDS = {"sfz": "sfizz", "sf2": "fluidsynth", "synth": None, "notes": "notes"}
# Licences we accept at all, and whether each is commercially clean. Anything else is an
# error: a new licence must be reviewed and added here deliberately.
LICENCES: dict[str, bool] = {
    "CC0-1.0": True, "PDDL-1.0": True, "LicenseRef-PublicDomain": True,
    "CC-BY-3.0": True, "CC-BY-4.0": True, "MIT": True,
    "GPL-3.0-or-later-with-exception": True,     # FreePats exception: renders are free
    "LicenseRef-IowaMIS-any-use": True,
    "LicenseRef-engine-output": True,            # GPL synth output with our own patches
    "CC-BY-SA-3.0": True, "CC-BY-SA-4.0": True,  # usable; derived *samples* share-alike
    "CC-BY-NC-3.0": False, "CC-BY-NC-4.0": False, "CC-BY-NC-SA-4.0": False,
}


class SourceError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    kind: str                       # sfz | sf2 | synth
    engine: str                     # sfizz | fluidsynth | surge_xt | dexed
    leaves: tuple[str, ...]
    license: str
    commercial_clean: bool
    path: str | None = None
    programs: dict[str, tuple[int, ...]] = field(default_factory=dict)   # sf2: leaf -> GM programs
    license_url: str = ""
    attribution: str = ""
    download: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    weight: float = 1.0                    # relative pick probability among a leaf's sources
    note_map: dict[int, int] = field(default_factory=dict)   # GM drum note -> source key
    label_as: str | None = None    # coarser node to label with (e.g. bass.electric when the
                                   # playing technique is undocumented); its subtree is unknown
    instrument: str | None = None  # notes: file prefix `<instrument>-<pitch>-<velocity>.wav`
    native_sr: int | None = None   # recording sample rate when band-limited (e.g. NSynth 16 kHz)

    def file(self, root: Path) -> Path | None:
        return (root / self.path) if self.path else None


@dataclass(frozen=True)
class Registry:
    version: int
    root: Path
    v1_leaves: tuple[str, ...]
    sources: tuple[Source, ...]

    def for_leaf(self, leaf: str, *, commercial_only: bool = False) -> list[Source]:
        return [s for s in self.sources if leaf in s.leaves
                and (s.commercial_clean or not commercial_only)]

    def by_id(self, sid: str) -> Source:
        for s in self.sources:
            if s.id == sid:
                return s
        raise KeyError(sid)


def load_registry(path: Path | str | None = None, taxonomy: Taxonomy | None = None,
                  root: Path | str | None = None) -> Registry:
    """Load and validate. `root` overrides the yaml `root` (also `DISS_SYNTH_SOURCES`)."""
    doc = yaml.safe_load(Path(path or REGISTRY_PATH).read_text())
    tax = taxonomy or Taxonomy.load()
    leaves = set(tax.leaves())
    errs: list[str] = []
    r = Path(os.path.expanduser(str(root or os.environ.get("DISS_SYNTH_SOURCES") or doc["root"])))
    v1 = tuple(doc.get("v1_leaves") or ())
    errs += [f"v1_leaves: {l} is not a taxonomy leaf" for l in v1 if l not in leaves]
    out, seen = [], set()
    for raw in _expand(doc.get("sources") or [], tax, errs):
        sid = raw.get("id", "?")
        if sid in seen:
            errs.append(f"{sid}: duplicate id")
        seen.add(sid)
        kind = raw.get("kind")
        if kind not in KINDS:
            errs.append(f"{sid}: kind must be one of {sorted(KINDS)}")
            continue
        lic = raw.get("license")
        if lic not in LICENCES:
            errs.append(f"{sid}: licence {lic!r} not in the reviewed list (sources.py LICENCES)")
            continue
        clean = bool(raw.get("commercial_clean"))
        if clean and not LICENCES[lic]:
            errs.append(f"{sid}: commercial_clean=true contradicts licence {lic}")
        src_leaves = tuple(raw.get("leaves") or ())
        errs += [f"{sid}: {l} is not a taxonomy leaf" for l in src_leaves if l not in leaves]
        if kind != "synth" and not raw.get("path"):
            errs.append(f"{sid}: {kind} source needs a path")
        la = raw.get("label_as")
        if la is not None and (la not in tax or any(la not in tax.ancestors(l) for l in src_leaves)):
            errs.append(f"{sid}: label_as {la!r} must be a taxonomy ancestor of all its leaves")
        if float(raw.get("weight", 1.0)) <= 0:
            errs.append(f"{sid}: weight must be > 0")
        progs = {k: tuple(v) for k, v in (raw.get("programs") or {}).items()}
        errs += [f"{sid}: programs key {k} not in its leaves" for k in progs if k not in src_leaves]
        out.append(Source(id=sid, name=raw.get("name", sid), kind=kind,
                          engine=raw.get("engine") or KINDS[kind] or "", leaves=src_leaves,
                          license=lic, commercial_clean=clean, path=raw.get("path"),
                          programs=progs, license_url=raw.get("license_url", ""),
                          attribution=raw.get("attribution", ""),
                          download=raw.get("download") or {}, notes=raw.get("notes", ""),
                          weight=float(raw.get("weight", 1.0)),
                          note_map={int(k): int(v) for k, v in (raw.get("note_map") or {}).items()},
                          label_as=raw.get("label_as"), instrument=raw.get("instrument"),
                          native_sr=raw.get("native_sr")))
    reg = Registry(int(doc.get("version", 1)), r, v1, tuple(out))
    uncovered = [l for l in v1 if not reg.for_leaf(l, commercial_only=True)]
    errs += [f"v1 leaf {l} has no commercially clean source" for l in uncovered]
    if errs:
        raise SourceError("sources.yaml invalid:\n  " + "\n  ".join(errs))
    return reg


def _expand(raws: list[dict], tax: Taxonomy, errs: list[str]) -> list[dict]:
    """Entries with `expand_labels: <json>` become one source per labelled instrument
    (multi-instrument note collections such as NSynth). Label values: a taxonomy leaf; a
    coarser node (rendered for all its v1-relevant leaves, labelled coarsely); or "skip".
    The entry's `weight` is split evenly across the instruments serving each leaf, so a
    collection with 40 basses doesn't swamp the leaf's other sources."""
    out: list[dict] = []
    for raw in raws:
        lab_file = raw.get("expand_labels")
        if not lab_file:
            out.append(raw)
            continue
        p = Path(lab_file) if Path(lab_file).is_absolute() else Path(__file__).parent / lab_file
        if not p.exists():          # labels not made yet: the collection contributes nothing
            continue
        labels = json.loads(p.read_text()).get("labels", {})
        per_leaf: dict[str, list[tuple[str, tuple[str, ...], str | None]]] = {}
        for inst, node in sorted(labels.items()):
            if node in (None, "skip"):
                continue
            if node not in tax:
                errs.append(f"{raw.get('id')}: {inst}: {node!r} is not a taxonomy node")
                continue
            def covered(n: str) -> tuple[str, ...]:
                return tuple(l for l in raw.get("leaves") or () if n == l or n in tax.ancestors(l))
            leaves = covered(node)
            if not leaves and tax.parent(node):     # e.g. alto sax -> saxophone (serves tenor parts)
                node = tax.parent(node)
                leaves = covered(node)
            if not leaves:
                errs.append(f"{raw.get('id')}: {inst}: {node!r} covers none of the entry's leaves")
                continue
            coarse = None if node in leaves and len(leaves) == 1 and node == leaves[0] else node
            for l in leaves:
                per_leaf.setdefault(l, []).append((inst, leaves, coarse))
        n_for = {l: len(v) for l, v in per_leaf.items()}
        done = set()
        for l, items in per_leaf.items():
            for inst, leaves, coarse in items:
                if inst in done:
                    continue
                done.add(inst)
                w = float(raw.get("weight", 1.0)) / max(n_for[x] for x in leaves)
                out.append({k: v for k, v in raw.items() if k not in ("expand_labels", "leaves", "weight")}
                           | {"id": f"{raw['id']}:{inst}", "leaves": list(leaves), "instrument": inst,
                              "weight": w, "label_as": coarse})
    return out
