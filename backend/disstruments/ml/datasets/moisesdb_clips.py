"""MoisesDB 10-second mixture clips — TEST-ONLY (never trained on; Bradley's approval).

Licence: CC BY-NC-SA 4.0, non-commercial research. Audio and the download URL never go in
the repo; clips are built locally by `moisesdb_fetch` (one window per song, mono 24 kHz
FLAC = sum of ALL the song's track wavs over the window) and live outside the repo at
`<root>/clips/<song_uuid>_w<start>.{flac,json}`. Every record is split="test"; `artist` is
the MoisesDB artist (47 artists, Firefly alone has 60 of 240 songs, so the bootstrap MUST
resample artists — the harness groups by `artist`).

Why a mapping dict here instead of a `moisesdb:` section in mappings.yaml:
- mappings.yaml's sha256 is part of the medleydb/openmic/slakh loader configs (and so of
  every existing report's config hash); adding a section would change those hashes for
  no reason.
- The labels are contextual (window activity, has_bleed, song-level drum/cymbal context,
  genre), so half of the logic could not live in YAML anyway. The per-trackType table is
  still validated through `base.LabelMap` (unknown node ids raise).

Per-track activity (from raw levels stored by the fetcher; thresholds are loader params,
so they can change without re-downloading):
  rms_db        RMS over the window, all channels, dBFS (full-scale square = 0 dB);
                None = digital silence.
  peak_frame_db max RMS over 50 ms frames, dBFS.
  ACTIVE     rms_db >= ABS_FLOOR_DB (-50) and rms_db >= loudest track's rms_db - REL_DB (35)
  SILENT     peak_frame_db < SILENCE_DB (-70): no 50 ms frame reaches -70 dBFS
  UNCERTAIN  anything else (present but quiet/buried): its labels become UNKNOWN, never
             positive and never negative.
A silent track contributes nothing; a leaf with no active/uncertain track and no other
unknown flag is a true NEGATIVE (the multitrack is exhaustive).

Track -> label rules (TRACK_RULES; node = positive when the track is ACTIVE and clean):
  bass_guitar -> bass.electric (fingered/picked/slap/fretless unknown)
  bass_synthesizer_(moog_etc) -> bass.synth (analog/sub_808 unknown)
  contrabass/double_bass -> bass.upright
  acoustic_guitar -> guitar.acoustic (nylon/steel unknown)
  clean_electric_guitar -> guitar.electric.clean; distorted_electric_guitar -> .distorted
  grand_piano -> keys.piano
  electric_piano_(rhodes,wurlitzer,piano sound alike) -> keys.electric_piano (children
      unknown) + keys.piano unknown (the class literally includes "piano sound alike")
      + keys.clavinet unknown (an electric keyboard MoisesDB has no class for)
  organ,_electric_organ -> keys.organ (drawbar/combo/pipe unknown)
  synth_lead -> keys.synth.lead; synth_pad -> keys.synth.pad
  other_sounds_(harpsichord, mellotron...) -> no positive; keys subtree unknown
  drum_machine -> drums.electronic (tr808/tr909 unknown)
  kick/snare/toms/full_acoustic_drumkit/overheads -> drums.acoustic_kit
      context: if a drum_machine track is not silent, or the genre is rap/electronic
      (where "kick/snare" stems are routinely samples), and no full_acoustic_drumkit or
      overheads track is ACTIVE (room/overhead mics only exist for a real kit), then
      kick/snare/toms -> drums positive with acoustic_kit AND electronic unknown.
  cymbals, hi_hat -> cymbals
  overheads -> drums.acoustic_kit (NOT cymbals positive: overheads carry the whole kit
      and are active on kick/snare-only bars too; deviation from the brief)
      cymbals context (song level): cymbals UNKNOWN if a full_acoustic_drumkit or
      drum_machine track is not silent (hats are inside them), or if
      kick/snare/toms/overheads are not silent and the SONG has no dedicated
      cymbals/hi_hat track (cymbals are then inside the close-mic bleed / overheads).
      If the song has dedicated cymbal/hat tracks and they are silent in the window, the
      drummer is not playing cymbals: true negative.
  lead_male_singer / lead_female_singer / background_vocals -> voice.sung; voice.rap,
      voice.spoken, voice.screamed UNKNOWN whenever a vocal track is active/uncertain
      (MoisesDB labels the singer, never the delivery; rap verses on pop songs are filed
      as "singer"); background_vocals also make voice.choir unknown.
      Rap-genre songs: vocal tracks -> voice positive, ALL voice children unknown (the
      "lead singer" of a rap song is mostly rapping; voice.sung would be a guess).
  other_(vocoder, beatboxing) -> no positive; voice + keys.synth unknown
  a-tonal_percussion -> percussion; shaker/tambourine/hand_drums/claps unknown
      (percussion.mallet stays observed: pitched percussion has its own class)
  pitched_percussion -> percussion.mallet (children unknown)
  brass -> brass (children unknown); reeds -> woodwinds (children unknown)
  flutes_(piccolo, bamboo flute, panpipes) -> woodwinds.flute (panpipes are lumped in by
      the source; 2 tracks total)
  other_wind -> no positive; woodwinds, brass, keys.accordion unknown
  cello_(solo) -> strings.bowed.cello; viola_(solo) -> strings.bowed.viola
  string_section / cello_section / viola_section -> strings.bowed.section (as MedleyDB)
  other_strings -> strings.bowed (children unknown; solo violin lives here)
  other_plucked (banjo, mandolin, ukulele, harp) -> strings.plucked (children unknown)
  fx/processed_sound (scratches, risers, impacts) -> no positive; keys.synth,
      drums.electronic and cymbals (reverse-cymbal swells) unknown
Bleed: an ACTIVE track with has_bleed=true contributes its nodes as UNKNOWN (its level
may be bleed); another clean active track of the same label still makes them positive.
Dataset-wide leaf mask (no upward closure — the mask is constant, so it is not
label-dependent and keys.synth/keys stay observed negatives on keyless songs):
  keys.synth.pluck, keys.synth.brass — MoisesDB has no class for them (plucks may be
  filed as synth_lead or other_sounds, synth brass as synth_lead or brass).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..taxonomy import Taxonomy, get_taxonomy
from .base import (DatasetIndex, LabelMap, Record, UnknownLabelsError, check_unique_ids,
                   normalize_artist, observed_exhaustive)

LICENCE = "CC BY-NC-SA 4.0 (MoisesDB); non-commercial research, test-only, no redistribution"
SCHEMA = "moisesdb_clip/1"

ABS_FLOOR_DB = -50.0      # active: window RMS at least this (dBFS)
REL_DB = 35.0             # active: within this many dB of the loudest track in the window
SILENCE_DB = -70.0        # silent: no 50 ms frame reaches this (dBFS)

ACTIVE, UNCERTAIN, SILENT = "active", "uncertain", "silent"

# ---------------------------------------------------------------------- source labels
BASS_GUITAR = "bass_guitar"
BASS_SYNTH = "bass_synthesizer_(moog_etc)"
CONTRABASS = "contrabass/double_bass_(bass_of_instrings)"
CELLO = "cello_(solo)"
CELLO_SECTION = "cello_section"
OTHER_STRINGS = "other_strings"
STRING_SECTION = "string_section"
VIOLA = "viola_(solo)"
VIOLA_SECTION = "viola_section"
CYMBALS = "cymbals"
DRUM_MACHINE = "drum_machine"
FULL_KIT = "full_acoustic_drumkit"
HI_HAT = "hi_hat"
KICK = "kick_drum"
OVERHEADS = "overheads"
SNARE = "snare_drum"
TOMS = "toms"
ACOUSTIC_GUITAR = "acoustic_guitar"
CLEAN_GUITAR = "clean_electric_guitar"
DIST_GUITAR = "distorted_electric_guitar"
FX = "fx/processed_sound,_scratches,_gun_shots,_explosions_etc"
ORGAN = "organ,_electric_organ"
OTHER_KEYS = "other_sounds_(hapischord,_melotron_etc)"
SYNTH_LEAD = "synth_lead"
SYNTH_PAD = "synth_pad"
OTHER_PLUCKED = "banjo,_mandolin,_ukulele,_harp_etc"
ATONAL_PERC = "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"
PITCHED_PERC = "pitched_percussion_(mallets,_glockenspiel,_...)"
EPIANO = "electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)"
GRAND_PIANO = "grand_piano"
BACKING_VOX = "background_vocals"
LEAD_FEMALE = "lead_female_singer"
LEAD_MALE = "lead_male_singer"
OTHER_VOX = "other_(vocoder,_beatboxing_etc)"
BRASS = "brass_(trumpet,_trombone,_french_horn,_brass_etc)"
FLUTES = "flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)"
OTHER_WIND = "other_wind"
REEDS = "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)"

_DELIVERY = ["voice.rap", "voice.spoken", "voice.screamed"]

# trackType -> LabelMap spec (`node` positive when active+clean; internal nodes make their
# children unknown unless children_unknown: false; `unknown` extra subtree roots).
TRACK_RULES: dict[str, Any] = {
    BASS_GUITAR: "bass.electric",
    BASS_SYNTH: "bass.synth",
    CONTRABASS: "bass.upright",
    CELLO: "strings.bowed.cello",
    VIOLA: "strings.bowed.viola",
    CELLO_SECTION: "strings.bowed.section",
    VIOLA_SECTION: "strings.bowed.section",
    STRING_SECTION: "strings.bowed.section",
    OTHER_STRINGS: "strings.bowed",
    CYMBALS: "cymbals",
    HI_HAT: "cymbals",
    DRUM_MACHINE: "drums.electronic",
    FULL_KIT: "drums.acoustic_kit",
    KICK: "drums.acoustic_kit",
    SNARE: "drums.acoustic_kit",
    TOMS: "drums.acoustic_kit",
    OVERHEADS: "drums.acoustic_kit",
    ACOUSTIC_GUITAR: "guitar.acoustic",
    CLEAN_GUITAR: "guitar.electric.clean",
    DIST_GUITAR: "guitar.electric.distorted",
    ORGAN: "keys.organ",
    SYNTH_LEAD: "keys.synth.lead",
    SYNTH_PAD: "keys.synth.pad",
    OTHER_PLUCKED: "strings.plucked",
    ATONAL_PERC: {"node": "percussion", "children_unknown": False,
                  "unknown": ["percussion.shaker", "percussion.tambourine",
                              "percussion.hand_drums", "percussion.claps"]},
    PITCHED_PERC: "percussion.mallet",
    EPIANO: {"node": "keys.electric_piano", "unknown": ["keys.piano", "keys.clavinet"]},
    GRAND_PIANO: "keys.piano",
    LEAD_MALE: {"node": "voice.sung", "unknown": _DELIVERY},
    LEAD_FEMALE: {"node": "voice.sung", "unknown": _DELIVERY},
    BACKING_VOX: {"node": "voice.sung", "unknown": _DELIVERY + ["voice.choir"]},
    BRASS: "brass",
    FLUTES: "woodwinds.flute",
    REEDS: "woodwinds",
}
TRACK_IGNORE: dict[str, Any] = {
    OTHER_KEYS: {"reason": "harpsichord/mellotron/other keyboards: keyboard type unknown",
                 "unknown": ["keys"]},
    OTHER_VOX: {"reason": "vocoder/beatbox: voice or synth, delivery unknown",
                "unknown": ["voice", "keys.synth"]},
    OTHER_WIND: {"reason": "unspecified wind instrument",
                 "unknown": ["woodwinds", "brass", "keys.accordion"]},
    FX: {"reason": "processed sound/scratches/risers/impacts: may be synth, drum machine or "
                   "reverse cymbals", "unknown": ["keys.synth", "drums.electronic", "cymbals"]},
}
# Song-context overrides (see module docstring).
RAP_VOCALS = {"node": "voice"}                                      # all children unknown
KIT_AMBIGUOUS = {"node": "drums", "children_unknown": False,
                 "unknown": ["drums.acoustic_kit", "drums.electronic"]}
VOCALS = (LEAD_MALE, LEAD_FEMALE, BACKING_VOX)
CLOSE_MICS = (KICK, SNARE, TOMS)
DEDICATED_CYMBALS = (CYMBALS, HI_HAT)
KIT_GENRES = frozenset({"rap", "electronic"})
CONSTANT_LEAF_MASK = ("keys.synth.pluck", "keys.synth.brass")


def label_map(taxonomy: Taxonomy | None = None) -> LabelMap:
    tax = taxonomy or get_taxonomy()
    return LabelMap("moisesdb", TRACK_RULES, TRACK_IGNORE, tax)


def _context_map(taxonomy: Taxonomy) -> LabelMap:
    return LabelMap("moisesdb.context", {"rap_vocals": RAP_VOCALS, "kit_ambiguous": KIT_AMBIGUOUS},
                    None, taxonomy)


# ---------------------------------------------------------------------- activity
def classify_activity(levels: Sequence[tuple[float | None, float | None]], *,
                      abs_floor_db: float = ABS_FLOOR_DB, rel_db: float = REL_DB,
                      silence_db: float = SILENCE_DB) -> list[str]:
    """(rms_db, peak_frame_db) per track -> ACTIVE / UNCERTAIN / SILENT.

    None means digital silence (-inf dB)."""
    def f(x: float | None) -> float:
        return -math.inf if x is None else float(x)
    loudest = max((f(r) for r, _ in levels), default=-math.inf)
    out = []
    for rms, peak in levels:
        r, p = f(rms), f(peak)
        if r >= abs_floor_db and r >= loudest - rel_db:
            out.append(ACTIVE)
        elif p < silence_db:
            out.append(SILENT)
        else:
            out.append(UNCERTAIN)
    return out


# ---------------------------------------------------------------------- labels
def song_labels(tracks: Sequence[Mapping[str, Any]], states: Sequence[str], genre: str | None,
                taxonomy: Taxonomy | None = None, lmap: LabelMap | None = None,
                ) -> tuple[frozenset[str], frozenset[str], list[str], dict[str, Any]]:
    """(positive, observed, unmapped trackTypes, provenance) for one window.

    `tracks`: [{trackType, has_bleed, ...}] (every track of the song, in any order);
    `states`: activity per track (ACTIVE / UNCERTAIN / SILENT).
    """
    tax = taxonomy or get_taxonomy()
    lmap = lmap or label_map(tax)
    ctx = _context_map(tax)
    types = [str(t["trackType"]) for t in tracks]
    song_types = set(types)
    sounding = {tt for tt, s in zip(types, states) if s != SILENT}
    active_clean = {tt for tt, s, t in zip(types, states, tracks)
                    if s == ACTIVE and not t.get("has_bleed")}
    bleed_only = {tt for tt in song_types
                  if all(t.get("has_bleed") for t, x in zip(tracks, types) if x == tt)}
    rap = (genre or "").casefold() == "rap"
    kit_ambiguous = ((DRUM_MACHINE in sounding or (genre or "").casefold() in KIT_GENRES)
                     and not ({FULL_KIT, OVERHEADS} & active_clean))

    pos: set[str] = set()
    unk: set[str] = set()
    unmapped: list[str] = []
    why: Counter = Counter()
    for tt, state, t in zip(types, states, tracks):
        if tt in bleed_only and tt in lmap:
            # Song-level, label-independent: if every track of this type has bleed, its
            # nodes are UNKNOWN whether or not it sounds in the window. (Masking only the
            # active ones made e.g. `voice` observed only when absent on 66 songs, which is
            # label-dependent missingness, M1 debrief.)
            res = lmap.resolve([tt])
            unk |= set(res.mapped) | set(res.unknown_roots)
            why["bleed_only_type"] += 1
            continue
        if state == SILENT:
            continue
        if tt not in lmap:
            unmapped.append(tt)
            continue
        if rap and tt in VOCALS:
            res = ctx.resolve(["rap_vocals"])
        elif kit_ambiguous and tt in CLOSE_MICS:
            res = ctx.resolve(["kit_ambiguous"])
            why["kit_ambiguous"] += 1
        else:
            res = lmap.resolve([tt])
        nodes, extra_unknown = set(res.mapped), set(res.unknown_roots)
        if state == ACTIVE and not t.get("has_bleed"):
            pos |= nodes
            unk |= extra_unknown
        else:                                      # uncertain level, or level may be bleed
            unk |= nodes | extra_unknown
            why["bleed" if state == ACTIVE else "uncertain"] += 1
    # cymbals: song-level context
    if (FULL_KIT in sounding or DRUM_MACHINE in sounding or
            ((set(CLOSE_MICS) | {OVERHEADS}) & sounding and
             not (set(DEDICATED_CYMBALS) & song_types))):
        unk.add("cymbals")
        why["cymbals_in_kit"] += 1
    # Bradley's taxonomy decision Q1 (2026-09-28): cymbals are scored on OpenMIC only.
    # Kept consistent here even though MoisesDB has dedicated cymbal tracks.
    unk.add("cymbals")
    pos.discard("cymbals")
    positive = tax.close_upward(pos)
    if unmapped:
        unk.add("*")
    observed = observed_exhaustive(tax, positive, unk)
    observed = frozenset(n for n in observed if n not in CONSTANT_LEAF_MASK or n in positive)
    prov = {"rap_genre": rap, "kit_ambiguous": kit_ambiguous, "unknown_reasons": dict(why),
            "unknown_roots": sorted(unk)}
    return positive, observed, unmapped, prov


# ---------------------------------------------------------------------- loader
def window_tag(window_start: float) -> str:
    return f"w{float(window_start):g}"


def _per_leaf(records: Iterable[Record], tax: Taxonomy) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for leaf in tax.leaves():
        p = sum(leaf in r.positive for r in records)
        u = sum(leaf not in r.observed for r in records)
        out[leaf] = {"pos": p, "unknown": u}
    return out


def load(root: Path | str, *, window_start: float = 20.0, strict: bool = True,
         abs_floor_db: float = ABS_FLOOR_DB, rel_db: float = REL_DB,
         silence_db: float = SILENCE_DB, taxonomy: Taxonomy | None = None) -> DatasetIndex:
    """Records for every `<root>/clips/*_w<start>.json` (one per song)."""
    tax = taxonomy or get_taxonomy()
    root = Path(root).expanduser()
    tag = window_tag(window_start)
    files = sorted((root / "clips").glob(f"*_{tag}.json"))
    if not files:
        raise FileNotFoundError(f"moisesdb: no clips/*_{tag}.json under {root} "
                                f"(run python -m disstruments.ml.datasets.moisesdb_fetch)")
    lmap = label_map(tax)
    records: list[Record] = []
    unknown_types: Counter = Counter()
    states_count: Counter = Counter()
    for f in files:
        doc = json.loads(f.read_text())
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"moisesdb: {f.name}: schema {doc.get('schema')!r} != {SCHEMA!r}")
        tracks = doc["tracks"]
        states = classify_activity([(t.get("rms_db"), t.get("peak_frame_db")) for t in tracks],
                                   abs_floor_db=abs_floor_db, rel_db=rel_db, silence_db=silence_db)
        states_count.update(states)
        pos, obs, unmapped, prov = song_labels(tracks, states, doc.get("genre"), tax, lmap)
        unknown_types.update(unmapped)
        flac = f.with_suffix(".flac")
        records.append(Record(
            dataset="moisesdb", item_id=f"{doc['song_id']}_{tag}", split="test",
            artist=normalize_artist(doc["artist"]), positive=pos, observed=obs,
            source_labels=tuple(sorted({t["trackType"] for t, s in zip(tracks, states)
                                        if s != SILENT})),
            unknown_labels=tuple(dict.fromkeys(unmapped)),
            audio={"mix": flac} if flac.exists() else {},
            extra={"song": doc.get("song"), "genre": doc.get("genre"),
                   "window": doc.get("window"),
                   "activity": {t["track_id"]: s for t, s in zip(tracks, states)}, **prov}))
    collect = dict(unknown_types)
    if collect and strict:
        raise UnknownLabelsError("moisesdb", collect)
    check_unique_ids("moisesdb", [r.item_id for r in records])
    stats = {"n_items": len(records), "n_artists": len({r.artist for r in records}),
             "activity_states": dict(states_count), "unknown_track_types": collect,
             "per_leaf": _per_leaf(records, tax)}
    return DatasetIndex("moisesdb", records,
                        {"dataset": "moisesdb", "window_start": float(window_start),
                         "abs_floor_db": abs_floor_db, "rel_db": rel_db, "silence_db": silence_db,
                         "taxonomy_version": tax.version, "licence": LICENCE}, stats)
