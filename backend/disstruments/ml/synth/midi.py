"""MIDI selection (M2 S2): pick 10 s windows and decide which taxonomy leaf each track becomes.

A MIDI track only tells us its *role* (GM program family / drum channel). We keep the role
(a bass line is rendered as a bass) and choose the leaf within the role ourselves, so the
label is exact. Tracks whose family has no v1 leaf are dropped, not rendered: every audible
stem has a known label, so every other taxonomy node is a true negative.

Splits are by MIDI *composition* (artist/title for the Lakh clean subset, file md5
otherwise), so the same song never lands in train and test.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# GM program (0-based) family -> role. Drums come from the drum channel, not the program.
ROLE_OF_PROGRAM: dict[range, str] = {
    range(0, 8): "keys_piano",      # acoustic + electric pianos
    range(16, 21): "organ",         # drawbar / percussive / rock / church / reed organ
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(56, 64): "brass",
    range(64, 72): "sax",           # soprano/alto/tenor/baritone sax + other reeds
    range(80, 88): "synth_lead",
    range(88, 96): "synth_pad",
}

# role -> candidate leaves (v1). GM hints narrow the choice where the program is specific.
ROLE_LEAVES: dict[str, tuple[str, ...]] = {
    "keys_piano": ("keys.piano", "keys.electric_piano.rhodes", "keys.electric_piano.fm"),
    "organ": ("keys.organ.drawbar",),
    "guitar": ("guitar.acoustic.nylon", "guitar.acoustic.steel",
               "guitar.electric.clean", "guitar.electric.distorted"),
    "bass": ("bass.electric.picked", "bass.electric.fingered", "bass.upright",
             "bass.synth.analog", "bass.synth.sub_808"),
    "brass": ("brass.trumpet",),
    "sax": ("woodwinds.saxophone.tenor",),
    "synth_lead": ("keys.synth.lead", "keys.synth.pluck"),
    "synth_pad": ("keys.synth.pad",),
    "drums": ("drums.acoustic_kit", "drums.electronic.tr808"),
}

# Programs that pin a leaf (or a subset) because the GM name is specific.
PROGRAM_HINTS: dict[int, tuple[str, ...]] = {
    0: ("keys.piano",), 1: ("keys.piano",), 2: ("keys.piano",), 3: ("keys.piano",),
    4: ("keys.electric_piano.rhodes",), 5: ("keys.electric_piano.fm",),
    24: ("guitar.acoustic.nylon",), 25: ("guitar.acoustic.steel",),
    26: ("guitar.electric.clean",), 27: ("guitar.electric.clean",),
    28: ("guitar.electric.clean",), 29: ("guitar.electric.distorted",),
    30: ("guitar.electric.distorted",),
    32: ("bass.upright",), 33: ("bass.electric.fingered",), 34: ("bass.electric.picked",),
    38: ("bass.synth.analog", "bass.synth.sub_808"), 39: ("bass.synth.analog", "bass.synth.sub_808"),
    56: ("brass.trumpet",), 66: ("woodwinds.saxophone.tenor",),
}


def role_of(program: int, is_drum: bool) -> str | None:
    if is_drum:
        return "drums"
    for r, role in ROLE_OF_PROGRAM.items():
        if program in r:
            return role
    return None


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float     # seconds, relative to window start
    end: float
    velocity: int


@dataclass
class Part:
    """One MIDI track inside a window, with its role and (once assigned) its leaf."""
    track_index: int
    program: int
    is_drum: bool
    role: str
    notes: list[Note]
    leaf: str | None = None
    gm_name: str = ""


@dataclass
class Window:
    midi_id: str           # md5 of the file
    composition: str       # split key (artist/title or md5)
    path: str              # relative to the MIDI root
    start: float
    duration: float
    tempo: float | None
    parts: list[Part] = field(default_factory=list)


def composition_key(path: Path, root: Path) -> str:
    """Lakh clean_midi layout is `<Artist>/<Title>[.N].mid`; versions of one song share
    a key. Anything else falls back to the file's md5 (computed by the caller)."""
    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        title = rel.stem.split(".")[0].strip().lower()
        return f"{rel.parts[-2].strip().lower()}/{title}"
    return ""


def split_of(composition: str, fractions: Mapping[str, float]) -> str:
    """Deterministic split from a stable hash of the composition key."""
    u = (zlib.crc32(composition.encode()) & 0xFFFFFFFF) / 2 ** 32
    acc = 0.0
    total = sum(fractions.values())
    for name, f in fractions.items():
        acc += f / total
        if u < acc:
            return name
    return list(fractions)[-1]


def load_windows(path: Path, root: Path, *, window_s: float = 10.0, hop_s: float = 5.0,
                 min_parts: int = 1, min_notes_per_part: int = 4,
                 max_windows: int | None = None) -> list[Window]:
    """Parse one MIDI file into candidate windows (only parts with a v1 role are kept)."""
    import pretty_midi

    data = path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    comp = composition_key(path, root) or md5
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:  # noqa: BLE001 - Lakh contains invalid files; skip them
        return []
    end = pm.get_end_time()
    if end < window_s:
        return []
    tempos = pm.get_tempo_changes()[1]
    tempo = float(tempos[0]) if len(tempos) else None
    out: list[Window] = []
    t = 0.0
    while t + window_s <= end:
        parts = []
        for k, inst in enumerate(pm.instruments):
            role = role_of(inst.program, inst.is_drum)
            if role is None:
                continue
            notes = [Note(n.pitch, max(0.0, n.start - t), min(window_s, n.end - t), n.velocity)
                     for n in inst.notes if n.end > t and n.start < t + window_s]
            notes = [n for n in notes if n.end - n.start > 0.01]
            if len(notes) >= min_notes_per_part:
                parts.append(Part(k, inst.program, inst.is_drum, role, notes,
                                  gm_name=pretty_midi.program_to_instrument_name(inst.program)
                                  if not inst.is_drum else "Drums"))
        if len(parts) >= min_parts:
            out.append(Window(md5, comp, str(path.relative_to(root)), round(t, 3), window_s,
                              tempo, parts))
            if max_windows and len(out) >= max_windows:
                break
        t += hop_s
    return out


class LeafBalancer:
    """Choose a leaf for each part: GM hint if specific, else the least-used candidate
    (ties broken by the seeded RNG). Class balance is a config knob, not a crawl problem."""

    def __init__(self, leaves: Iterable[str], rng: np.random.Generator,
                 hint_prob: float = 0.7):
        self.counts = {l: 0 for l in leaves}
        self.rng = rng
        self.hint_prob = hint_prob

    def choose(self, part: Part) -> str | None:
        cands = [l for l in ROLE_LEAVES.get(part.role, ()) if l in self.counts]
        if not cands:
            return None
        hint = [] if part.is_drum else [l for l in PROGRAM_HINTS.get(part.program, ())
                                        if l in cands]
        pool = hint if hint and self.rng.random() < self.hint_prob else cands
        low = min(self.counts[l] for l in pool)
        best = [l for l in pool if self.counts[l] == low]
        leaf = best[int(self.rng.integers(len(best)))]
        self.counts[leaf] += 1
        return leaf


def assign_leaves(window: Window, balancer: LeafBalancer, max_parts: int = 5) -> Window:
    """Assign leaves in place; keep at most `max_parts` parts and at most one part per
    leaf (two stems of the same leaf add nothing to the label). Parts left without a
    leaf are removed, so they are never rendered."""
    rng = balancer.rng
    order = list(rng.permutation(len(window.parts)))
    kept: list[Part] = []
    used: set[str] = set()
    for k in order:
        if len(kept) >= max_parts:
            break
        p = window.parts[k]
        leaf = balancer.choose(p)
        if leaf is None or leaf in used:
            if leaf is not None:
                balancer.counts[leaf] -= 1
            continue
        p.leaf = leaf
        used.add(leaf)
        kept.append(p)
    window.parts = sorted(kept, key=lambda p: p.track_index)
    return window


def notes_to_midi(notes: Sequence[Note], program: int, is_drum: bool, path: Path,
                  tempo: float = 120.0, pitch_shift: int = 0) -> Path:
    """Write one part as a single-track MIDI file (input for sfizz/FluidSynth renders)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    for n in notes:
        pitch = n.pitch if is_drum else int(np.clip(n.pitch + pitch_shift, 0, 127))
        inst.notes.append(pretty_midi.Note(n.velocity, pitch, n.start, n.end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path
