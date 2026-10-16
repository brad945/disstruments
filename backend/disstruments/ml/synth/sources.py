"""Source registry (`sources.yaml`, M2 S0): which sample library / synth renders which leaf,
under which licence. Every rendered clip records the ids (and licences) of its sources, so
a commercially clean subset can always be filtered out (docs/DIRECTION_NOTES.md)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
    # Physical instrument / recording identity, for counting *distinct instruments* per leaf
    # (M4: leaves with 1-2 instruments memorize them). Articulation variants of one recording
    # (VSCO trumpet sus/vib/stac, FSBS clean/dist) share a group. Default: the source id.
    group: str = ""
    status: str = ""               # "" = in use; "pending_approval" = download not approved yet

    def file(self, root: Path) -> Path | None:
        return (root / self.path) if self.path else None

    def available(self, root: Path) -> bool:
        """Its audio is on disk (synths always are). A pending source with no files simply
        doesn't count; the build never picks it (the engine would fail on a missing file)."""
        if self.kind == "synth":
            return True
        f = self.file(root)
        if f is None or not f.exists():
            return False
        if self.kind == "notes":
            return (self.instrument or "") in notes_instruments(f)
        return True


_NOTES_DIRS: dict[str, frozenset[str]] = {}


def notes_instruments(d: Path) -> frozenset[str]:
    """Instrument prefixes present in a notes directory (`<inst>-<pitch>-<vel>.wav`); one
    directory scan, cached (the shared notes dir holds ~300k files)."""
    key = str(d)
    if key not in _NOTES_DIRS:
        names = set()
        if Path(d).is_dir():
            for e in os.scandir(d):
                n = e.name
                if n.endswith(".wav") and n.count("-") >= 2:
                    names.add(n.rsplit("-", 2)[0])
        _NOTES_DIRS[key] = frozenset(names)
    return _NOTES_DIRS[key]


@dataclass(frozen=True)
class Registry:
    version: int
    root: Path
    v1_leaves: tuple[str, ...]
    sources: tuple[Source, ...]
    v8_leaves: tuple[str, ...] = ()        # candidate new leaves; gated by `leaf_gate`

    def for_leaf(self, leaf: str, *, commercial_only: bool = False) -> list[Source]:
        return [s for s in self.sources if leaf in s.leaves
                and (s.commercial_clean or not commercial_only)]

    def by_id(self, sid: str) -> Source:
        for s in self.sources:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def instruments(self, leaf: str, *, commercial_only: bool = False,
                    present_only: bool = True) -> dict[str, list[str]]:
        """Distinct instruments (group -> source ids) that render `leaf` with an *exact*
        label. Coarse sources (label_as) don't count: they never produce this leaf's label."""
        out: dict[str, list[str]] = {}
        for s in self.for_leaf(leaf, commercial_only=commercial_only):
            if s.label_as is not None:
                continue
            if present_only and (s.status or not s.available(self.root)):
                continue
            out.setdefault(s.group or s.id, []).append(s.id)
        return out


def leaf_gate(reg: Registry, leaves: Iterable[str], *, min_instruments: int = 2,
              commercial_only: bool = False) -> dict[str, dict[str, Any]]:
    """Per candidate leaf: distinct instruments on disk, pending ones, and whether it is
    trainable *fairly* (>= `min_instruments` distinct instruments; one source = the model
    learns that sampler, not the instrument - M4 leave-source-out)."""
    report = {}
    for leaf in leaves:
        have = reg.instruments(leaf, commercial_only=commercial_only)
        every = reg.instruments(leaf, commercial_only=commercial_only, present_only=False)
        report[leaf] = {
            "instruments": sorted(have),
            "pending": sorted(set(every) - set(have)),
            "coarse_sources": sorted(s.id for s in reg.for_leaf(leaf, commercial_only=commercial_only)
                                     if s.label_as is not None),
            "trainable": len(have) >= min_instruments,
        }
    return report


SHARED_BANKS = ("fluidr3_gm",)     # GM fallback banks: never held out (they span every family)


def v8_holdout(reg: Registry, *, min_instruments: int = 2,
               commercial_only: bool = False) -> list[str]:
    """Leave-source-out pairs (`source_id:leaf`) for the v8 LSO model, fixed by rule before
    any v8 result: for every gated new leaf with >= min_instruments + 1 distinct instruments,
    hold out the alphabetically first instrument group that isn't a shared GM bank (all of
    that group's sources). Leaves with fewer instruments are not held out (training would
    keep < min_instruments)."""
    gate = leaf_gate(reg, reg.v8_leaves, min_instruments=min_instruments,
                     commercial_only=commercial_only)
    out = []
    for leaf, g in gate.items():
        if not g["trainable"] or len(g["instruments"]) < min_instruments + 1:
            continue
        groups = reg.instruments(leaf, commercial_only=commercial_only)
        pick = next((k for k in sorted(groups) if k not in SHARED_BANKS), None)
        if pick is not None:
            out += [f"{sid}:{leaf}" for sid in groups[pick]]
    return out


def resolve_leaf_set(reg: Registry, name: str, *, min_instruments: int = 2,
                     commercial_only: bool = False) -> tuple[tuple[str, ...], dict[str, Any]]:
    """`v1` = the original 20 leaves (unchanged behaviour). `v8` = v1 plus every v8 leaf
    that passes `leaf_gate`; the rest are left out (their MIDI parts are not rendered)."""
    if name == "v1":
        return reg.v1_leaves, {}
    if name != "v8":
        raise ValueError(f"unknown leaf set {name!r} (v1 | v8)")
    gate = leaf_gate(reg, reg.v8_leaves, min_instruments=min_instruments,
                     commercial_only=commercial_only)
    new = tuple(l for l in reg.v8_leaves if gate[l]["trainable"] and l not in reg.v1_leaves)
    return reg.v1_leaves + new, gate


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
    v8 = tuple(doc.get("v8_leaves") or ())
    errs += [f"v8_leaves: {l} is not a taxonomy leaf" for l in v8 if l not in leaves]
    errs += [f"v8_leaves: {l} is already a v1 leaf" for l in v8 if l in v1]
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
        # Without a program the GM bank would play the *MIDI part's* program, which after
        # cross-rendering within a role can be another instrument than the label says.
        if kind == "sf2":
            errs += [f"{sid}: sf2 leaf {l} has no programs entry" for l in src_leaves if l not in progs]
        status = raw.get("status") or ""
        if status not in ("", "pending_approval"):
            errs.append(f"{sid}: status must be empty or pending_approval")
        out.append(Source(id=sid, name=raw.get("name", sid), kind=kind,
                          engine=raw.get("engine") or KINDS[kind] or "", leaves=src_leaves,
                          license=lic, commercial_clean=clean, path=raw.get("path"),
                          programs=progs, license_url=raw.get("license_url", ""),
                          attribution=raw.get("attribution", ""),
                          download=raw.get("download") or {}, notes=raw.get("notes", ""),
                          weight=float(raw.get("weight", 1.0)),
                          note_map={int(k): int(v) for k, v in (raw.get("note_map") or {}).items()},
                          label_as=raw.get("label_as"), instrument=raw.get("instrument"),
                          native_sr=raw.get("native_sr"), group=str(raw.get("group") or sid),
                          status=status))
    reg = Registry(int(doc.get("version", 1)), r, v1, tuple(out), v8)
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
                              "weight": w, "label_as": coarse,
                              "group": f"{raw['id']}:{inst}"})      # each instrument is distinct
    return out
