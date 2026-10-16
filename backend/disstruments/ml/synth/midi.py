"""MIDI selection (M2 S2): pick 10 s windows and decide which taxonomy leaf each track becomes.

A MIDI track only tells us its *role* (GM program family / drum channel). We keep the role
(a bass line is rendered as a bass) and choose the leaf within the role ourselves, so the
label is exact. Tracks whose family has no leaf in the active leaf set are dropped, not
rendered: every audible stem has a known label, so every other taxonomy node is a true
negative. The drum channel is split by GM note into the kit part and non-kit percussion
parts (tambourine, shakers, claps, hand drums; v8).

Splits are by MIDI *artist* (Lakh clean subset is `Artist/Title.N.mid`; other files fall
back to their md5), so no artist (and therefore no song) lands in both train and test.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# GM program (0-based) family -> role. Drums come from the drum channel, not the program.
# v8 added the orchestral / folk families. Programs with no row are dropped (never
# rendered): celesta 8, music box 10, tubular bells 14, dulcimer 15, contrabass 43 (bowed
# double bass: no leaf of its own; bass.upright is learned from pizzicato), pizzicato
# strings 45, timpani 47, synth strings 50-51, choir/voice 52-54, orchestra hit 55,
# recorder/pan flute/bottle/shakuhachi/whistle/ocarina 74-79, sitar/koto/shamisen etc.,
# and synth brass 62/63 (Bradley, 2026-10-16: a synth patch is not an acoustic brass
# instrument, so it never renders under a brass.* label; v1-v7 rendered it as trumpet).
ROLE_OF_PROGRAM: dict[range, str] = {
    range(0, 8): "keys_piano",      # acoustic + electric pianos
    range(9, 10): "mallet",         # glockenspiel
    range(11, 14): "mallet",        # vibraphone, marimba, xylophone
    range(16, 21): "organ",         # drawbar / percussive / rock / church / reed organ
    range(21, 22): "accordion",
    range(22, 23): "harmonica",
    range(23, 24): "accordion",     # tango accordion
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(40, 43): "strings_solo",  # violin, viola, cello
    range(44, 45): "strings_section",   # tremolo strings
    range(46, 47): "harp",
    range(48, 50): "strings_section",   # string ensembles 1 + 2
    range(56, 62): "brass",         # trumpet .. brass section; 62/63 synth brass dropped
    range(64, 72): "reed",          # soprano/alto/tenor/baritone sax, oboe, English horn, bassoon, clarinet
    range(72, 74): "flute",         # piccolo, flute
    range(80, 88): "synth_lead",
    range(88, 96): "synth_pad",
    range(105, 106): "plucked",     # banjo
}

# role -> candidate leaves. GM hints narrow the choice where the program is specific; the
# label is exact because the part is rendered with a source of the chosen leaf. Which of
# these leaves a build may use is the active leaf set (`--leaf-set`); v1 = the original 20.
ROLE_LEAVES: dict[str, tuple[str, ...]] = {
    "keys_piano": ("keys.piano", "keys.electric_piano.rhodes", "keys.electric_piano.fm"),
    "organ": ("keys.organ.drawbar",),
    "guitar": ("guitar.acoustic.nylon", "guitar.acoustic.steel",
               "guitar.electric.clean", "guitar.electric.distorted"),
    "bass": ("bass.electric.picked", "bass.electric.fingered", "bass.upright",
             "bass.synth.analog", "bass.synth.sub_808"),
    "brass": ("brass.trumpet", "brass.trombone", "brass.french_horn", "brass.tuba",
              "brass.section"),
    "reed": ("woodwinds.saxophone.tenor", "woodwinds.saxophone.soprano",
             "woodwinds.saxophone.alto", "woodwinds.saxophone.baritone",
             "woodwinds.clarinet", "woodwinds.oboe", "woodwinds.bassoon"),
    "flute": ("woodwinds.flute",),
    "harmonica": ("woodwinds.harmonica",),
    "accordion": ("keys.accordion",),
    "strings_solo": ("strings.bowed.violin", "strings.bowed.viola", "strings.bowed.cello",
                     "strings.bowed.section"),          # a solo line may be played by a section
    "strings_section": ("strings.bowed.section",),      # never a solo instrument (chords)
    "harp": ("strings.plucked.harp",),
    "plucked": ("strings.plucked.banjo", "strings.plucked.mandolin"),
    "mallet": ("percussion.mallet.glockenspiel", "percussion.mallet.vibraphone",
               "percussion.mallet.marimba", "percussion.mallet.xylophone"),
    "synth_lead": ("keys.synth.lead", "keys.synth.pluck"),
    "synth_pad": ("keys.synth.pad",),
    "drums": ("drums.acoustic_kit", "drums.electronic.tr808"),
    # Non-kit percussion: split out of the drum channel by GM note (PERC_ROLE_NOTES).
    "perc_tambourine": ("percussion.tambourine",),
    "perc_shaker": ("percussion.shaker",),
    "perc_claps": ("percussion.claps",),
    "perc_hand_drums": ("percussion.hand_drums",),
}

# Programs that pin a leaf (or a subset) because the GM name is specific.
PROGRAM_HINTS: dict[int, tuple[str, ...]] = {
    0: ("keys.piano",), 1: ("keys.piano",), 2: ("keys.piano",), 3: ("keys.piano",),
    4: ("keys.electric_piano.rhodes",), 5: ("keys.electric_piano.fm",),
    9: ("percussion.mallet.glockenspiel",), 11: ("percussion.mallet.vibraphone",),
    12: ("percussion.mallet.marimba",), 13: ("percussion.mallet.xylophone",),
    21: ("keys.accordion",), 22: ("woodwinds.harmonica",), 23: ("keys.accordion",),
    24: ("guitar.acoustic.nylon",), 25: ("guitar.acoustic.steel",),
    26: ("guitar.electric.clean",), 27: ("guitar.electric.clean",),
    28: ("guitar.electric.clean",), 29: ("guitar.electric.distorted",),
    30: ("guitar.electric.distorted",),
    32: ("bass.upright",), 33: ("bass.electric.fingered",), 34: ("bass.electric.picked",),
    38: ("bass.synth.analog", "bass.synth.sub_808"), 39: ("bass.synth.analog", "bass.synth.sub_808"),
    40: ("strings.bowed.violin",), 41: ("strings.bowed.viola",), 42: ("strings.bowed.cello",),
    44: ("strings.bowed.section",), 46: ("strings.plucked.harp",),
    48: ("strings.bowed.section",), 49: ("strings.bowed.section",),
    56: ("brass.trumpet",), 57: ("brass.trombone",), 58: ("brass.tuba",),
    59: ("brass.trumpet",), 60: ("brass.french_horn",), 61: ("brass.section",),
    64: ("woodwinds.saxophone.soprano",), 65: ("woodwinds.saxophone.alto",),
    66: ("woodwinds.saxophone.tenor",), 67: ("woodwinds.saxophone.baritone",),
    68: ("woodwinds.oboe",), 69: ("woodwinds.oboe",),           # English horn -> oboe leaf
    70: ("woodwinds.bassoon",), 71: ("woodwinds.clarinet",),
    72: ("woodwinds.flute",), 73: ("woodwinds.flute",),         # piccolo -> flute leaf
    105: ("strings.plucked.banjo",),
}

# Leaves named "(solo)" in the taxonomy. A part with more than MAX_SOLO_POLYPHONY
# simultaneous notes (chords, divisi) is never rendered as a solo instrument: a solo-violin
# sampler playing 4-note chords sounds like a section, so the solo label would be wrong.
SOLO_LEAVES = frozenset({"strings.bowed.violin", "strings.bowed.viola", "strings.bowed.cello"})
MAX_SOLO_POLYPHONY = 2      # double stops are idiomatic on bowed strings


# GM drum notes that belong to a drum *kit*. Other GM percussion is either its own
# percussion.* part (PERC_ROLE_NOTES) or dropped (cowbell 56, vibraslap 58, timbales 65/66,
# agogo 67/68, whistles 71/72, guiro 73/74, claves 75, wood blocks 76/77, cuica 78/79,
# triangles 80/81: no leaf), never rendered unlabelled inside a kit part.
KIT_NOTES = frozenset({35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                       53, 55, 57, 59})
# Hi-hats and cymbals: rendering any of these makes the level-1 `cymbals` leaf audible.
# Only kit parts (`drums.*` leaves) can imply cymbals; percussion parts never contain them.
CYMBAL_NOTES = frozenset({42, 44, 46, 49, 51, 52, 53, 55, 57, 59})
# Non-kit percussion parts split off the drum channel (v8). Timbales (65/66) are played
# with sticks, so they are not hand drums: dropped (Bradley's call; see docs/debriefs/v8-plan.md).
PERC_ROLE_NOTES: dict[str, frozenset[int]] = {
    "perc_tambourine": frozenset({54}),
    "perc_shaker": frozenset({69, 70, 82}),          # cabasa, maracas, shaker
    "perc_claps": frozenset({39}),                    # hand clap
    "perc_hand_drums": frozenset({60, 61, 62, 63, 64}),   # bongos hi/lo, congas mute/open/low
}
assert not any(KIT_NOTES & n for n in PERC_ROLE_NOTES.values())
assert not any(CYMBAL_NOTES & n for n in PERC_ROLE_NOTES.values())


def role_of(program: int, is_drum: bool) -> str | None:
    """Role of a *melodic* program, or "drums" for the drum channel (whose percussion
    roles are split off by note in `load_windows`)."""
    if is_drum:
        return "drums"
    for r, role in ROLE_OF_PROGRAM.items():
        if program in r:
            return role
    return None


def active_roles(leaves: Iterable[str]) -> frozenset[str]:
    """Roles that can produce at least one of `leaves` (others need not be parsed)."""
    s = set(leaves)
    return frozenset(r for r, ls in ROLE_LEAVES.items() if s & set(ls))


def max_polyphony(notes: Sequence["Note"]) -> int:
    """Most notes sounding at once (a note ending exactly when another starts doesn't overlap)."""
    ev = sorted([(n.start, 1) for n in notes] + [(n.end, -1) for n in notes])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


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


def artist_key(composition: str) -> str:
    """Grouping/split key: the artist part of `artist/title` (artist-disjoint splits, the
    project-wide rule); a bare md5 key is its own group."""
    return composition.split("/", 1)[0]


def split_of(composition: str, fractions: Mapping[str, float]) -> str:
    """Deterministic split from a stable hash of the composition's *artist*, so splits are
    artist-disjoint (and therefore also song-disjoint)."""
    u = (zlib.crc32(artist_key(composition).encode()) & 0xFFFFFFFF) / 2 ** 32
    acc = 0.0
    total = sum(fractions.values())
    for name, f in fractions.items():
        acc += f / total
        if u < acc:
            return name
    return list(fractions)[-1]


def _role_groups(program: int, is_drum: bool) -> list[tuple[str, frozenset[int] | None]]:
    """(role, allowed GM notes or None = all) for one MIDI track. A drum track yields the
    kit part plus one part per non-kit percussion role."""
    if is_drum:
        return [("drums", KIT_NOTES)] + list(PERC_ROLE_NOTES.items())
    role = role_of(program, False)
    return [(role, None)] if role else []


def load_windows(path: Path, root: Path, *, window_s: float = 10.0, hop_s: float = 5.0,
                 min_parts: int = 1, min_notes_per_part: int = 4,
                 max_windows: int | None = None,
                 roles: Iterable[str] | None = None) -> list[Window]:
    """Parse one MIDI file into candidate windows (only parts with a role are kept; with
    `roles`, only those roles, so a v1 build parses exactly what it did before v8)."""
    keep_roles = None if roles is None else frozenset(roles)
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
            groups = [(r, ok) for r, ok in _role_groups(inst.program, inst.is_drum)
                      if keep_roles is None or r in keep_roles]
            if not groups:
                continue
            in_win = [Note(n.pitch, max(0.0, n.start - t), min(window_s, n.end - t), n.velocity)
                      for n in inst.notes if n.end > t and n.start < t + window_s]
            in_win = [n for n in in_win if n.end - n.start > 0.01]
            for role, allowed in groups:
                notes = [n for n in in_win if allowed is None or n.pitch in allowed]
                if len(notes) < min_notes_per_part:
                    continue
                if not inst.is_drum:
                    name = pretty_midi.program_to_instrument_name(inst.program)
                else:
                    name = "Drums" if role == "drums" else f"Drums ({role.removeprefix('perc_')})"
                parts.append(Part(k, inst.program, inst.is_drum, role, notes, gm_name=name))
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
        if any(l in SOLO_LEAVES for l in cands) and max_polyphony(part.notes) > MAX_SOLO_POLYPHONY:
            cands = [l for l in cands if l not in SOLO_LEAVES]
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
