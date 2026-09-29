"""Slakh2100 loader (orig / split2 / redux splits, MIDI-duplicate leakage checks).

Format (ethman/slakh-utils README, mirdata fixtures): `TrackXXXXX/metadata.yaml` with
`UUID` (= md5 of the source Lakh MIDI), `lmd_midi_dir` (contains the MSD track id), and
`stems.Sxx.{inst_class, program_num, is_drum, midi_program_name, plugin_name,
audio_rendered}`; audio at `mix.flac` and `stems/Sxx.flac` (or .wav after conversion).
Track dirs sit under `{train,validation,test,omitted}/` (redux release) or flat
(BabySlakh). Unrendered stems (see `_rendered`) are not in the mix and are skipped.

Labels come from `plugin_name` (the patch that rendered the audio), not the MIDI
program — see mappings.yaml `slakh` for the evidence. `inst_class` is the fallback
(non-strict mode only; strict mode raises on any unknown patch).

Near-silent stems: a rendered stem whose `integrated_loudness` is below
`min_loudness_lufs` (default -60 LUFS, ~40 dB under a typical -15..-20 LUFS stem) is
effectively inaudible in the mix. Its nodes become UNKNOWN (not positive: a model cannot
hear it; not negative: it is there). None disables the floor.

`slakh.unobserved` subtrees (3.0.0: cymbals) are unobserved on every track and stem.

Splits: computed from the track id (orig: 1-1500 train, 1501-1875 val, 1876-2100 test)
plus vendored slakh-utils tables (`slakh_splits.json`), so the directory layout the user
has does not matter; a layout that matches no known split scheme raises.
- redux (default): duplicate-MIDI tracks omitted; each MIDI appears once.
- split2: all 2100 tracks, duplicates moved so none crosses splits.
- orig: original split; 146 MIDI groups leak across splits (reported in stats).
Leakage is re-checked at load time from the metadata `UUID`.

Artist-disjointness: NOT provided and not claimable. Lakh MIDI has no reliable artist
field; LMD-matched only links to Million Song Dataset track ids (noisy audio matching).
We record the MSD id per track and report cross-split MSD-id collisions between distinct
MIDI files (same song, different MIDI) as a measured song-level leakage number.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from ..taxonomy import Taxonomy, get_taxonomy
from .base import (DatasetIndex, LabelMap, Record, StemLabels, apply_unobserved,
                   check_unique_ids, collect_unknown, load_mappings, unobserved_roots)

NAME = "slakh"
SPLIT_MODES = ("redux", "split2", "orig")
_DIR_SPLITS = {"train": "train", "validation": "val", "test": "test", "omitted": "omitted"}
_TRACK = re.compile(r"^Track\d{5}$")
_MSD = re.compile(r"TR[A-Z0-9]{16}")
SPLITS_PATH = Path(__file__).with_name("slakh_splits.json")
DEFAULT_MIN_LOUDNESS_LUFS = -60.0


def _loudness(stem: dict) -> float | None:
    try:
        return float(stem["integrated_loudness"])
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _tables() -> dict:
    t = json.loads(SPLITS_PATH.read_text())
    t["redux_omitted"] = frozenset(t["redux_omitted"])
    return t


def orig_split(track_id: str) -> str:
    n = int(track_id[5:])
    if not 1 <= n <= 2100:
        raise ValueError(f"slakh: {track_id} outside Slakh2100 id range")
    return "train" if n <= 1500 else "val" if n <= 1875 else "test"


def canonical_split(track_id: str, mode: str) -> str:
    """Split of a Slakh2100 track under `mode` ('omitted' for redux-dropped tracks)."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"slakh split mode must be one of {SPLIT_MODES}")
    base = orig_split(track_id)
    if mode == "redux" and track_id in _tables()["redux_omitted"]:
        return "omitted"
    if mode == "split2":
        dest = _tables()["split2_moves"].get(track_id)
        return _DIR_SPLITS[dest] if dest else base
    return base


def label_maps(taxonomy: Taxonomy | None = None) -> tuple[LabelMap, LabelMap]:
    tax = taxonomy or get_taxonomy()
    doc, _ = load_mappings(taxonomy=tax)
    d = doc[NAME]
    return (LabelMap(f"{NAME}.plugins", d["plugins"], d.get("plugins_ignore"), tax),
            LabelMap(f"{NAME}.inst_class", d["inst_class"], d.get("inst_class_ignore"), tax))


def _find_tracks(root: Path) -> list[tuple[str, Path, str | None]]:
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name in _DIR_SPLITS:
            out += [(t.name, t, _DIR_SPLITS[d.name]) for t in sorted(d.iterdir())
                    if t.is_dir() and _TRACK.match(t.name)]
        elif d.is_dir() and _TRACK.match(d.name):
            out.append((d.name, d, None))
    return out


def _audio(p: Path) -> Path | None:
    for ext in (".flac", ".wav"):
        if p.with_suffix(ext).exists():
            return p.with_suffix(ext)
    return None


def _rendered(stem: dict, audio: Path | None) -> bool:
    """Is this stem in the mix? BabySlakh ships `audio_rendered: false` on every stem
    even though the stem audio exists, so the flag alone is not trusted: a stem also
    counts as rendered if its audio file exists, or if it has a real patch AND a measured
    loudness (truly unrendered stems have `plugin_name: None` and no loudness)."""
    if stem.get("audio_rendered") is True or audio is not None:
        return True
    return str(stem.get("plugin_name")) not in ("None", "") and "integrated_loudness" in stem


def load(root: Path | str, *, split_mode: str = "redux", include_omitted: bool = False,
         strict: bool = True, taxonomy: Taxonomy | None = None,
         min_loudness_lufs: float | None = DEFAULT_MIN_LOUDNESS_LUFS) -> DatasetIndex:
    """Load every TrackXXXXX under `root` (a slakh2100 split dir tree or a flat dir)."""
    tax = taxonomy or get_taxonomy()
    doc, mappings_sha = load_mappings(taxonomy=tax)
    pmap, cmap = label_maps(tax)
    masked = unobserved_roots(doc, NAME, tax)
    root = Path(root)
    tracks = _find_tracks(root)
    if not tracks:
        raise FileNotFoundError(f"slakh: no TrackXXXXX dirs under {root}")
    check_unique_ids(NAME, (t for t, _, _ in tracks), f"track dirs under {root}")

    unknown = Counter()
    layout_errors: list[str] = []
    records: list[Record] = []
    n_omitted = n_unrendered = n_fallback = n_flag_overrides = n_quiet = 0
    for track_id, tdir, dir_split in tracks:
        split = canonical_split(track_id, split_mode)
        if dir_split is not None and dir_split not in {canonical_split(track_id, m) for m in SPLIT_MODES}:
            layout_errors.append(f"{track_id} in {dir_split}/ matches no known split scheme")
        if split == "omitted":
            n_omitted += 1
            if not include_omitted:
                continue
        meta = yaml.safe_load((tdir / "metadata.yaml").read_text())
        stems, labels, unknown_labels = [], [], []
        pos: set[str] = set()
        unk: set[str] = set()
        for stem_id, s in sorted((meta.get("stems") or {}).items()):
            stem_audio = _audio(tdir / "stems" / str(stem_id))
            rendered = _rendered(s, stem_audio)
            if not rendered:
                n_unrendered += 1
                continue
            n_flag_overrides += int(not s.get("audio_rendered", False))
            plugin = Path(str(s.get("plugin_name"))).name
            inst_class = str(s.get("inst_class"))
            src = (f"plugin:{plugin}", f"inst_class:{inst_class}", f"program:{s.get('program_num')}")
            if plugin in pmap:
                res = pmap.resolve([plugin])
            else:
                unknown[f"plugin:{plugin}"] += 1
                unknown_labels.append(f"plugin:{plugin}")
                n_fallback += 1
                res = cmap.resolve([inst_class])
                if res.unknown_labels:
                    unknown[f"inst_class:{inst_class}"] += 1
                    unknown_labels.append(f"inst_class:{inst_class}")
            lufs = _loudness(s)
            if min_loudness_lufs is not None and lufs is not None and lufs < min_loudness_lufs:
                n_quiet += 1                   # inaudible: its nodes are unknown, not positive
                unk |= res.mapped | res.unknown_roots
                labels += src + (f"quiet:{lufs:.1f}LUFS",)
                _, q_obs = apply_unobserved(tax, frozenset(), res.mapped | res.unknown_roots, masked)
                stems.append(StemLabels(str(stem_id), frozenset(), q_obs, src, stem_audio))
                continue
            s_pos, s_obs = apply_unobserved(tax, res.positive, res.unknown_roots, masked)
            pos |= s_pos
            unk |= res.unknown_roots
            labels += src
            stems.append(StemLabels(str(stem_id), s_pos, s_obs, src, stem_audio))
        positive, observed = apply_unobserved(tax, frozenset(pos), unk, masked)
        audio = {"mix": a} if (a := _audio(tdir / "mix")) else {}
        audio.update({st.stem_id: st.audio for st in stems if st.audio is not None})
        msd = _MSD.search(str(meta.get("lmd_midi_dir", "")))
        records.append(Record(
            dataset=NAME, item_id=track_id, split=split, artist=None,
            positive=positive, observed=observed,
            source_labels=tuple(dict.fromkeys(labels)),
            unknown_labels=tuple(dict.fromkeys(unknown_labels)),
            audio=audio, stems=tuple(stems),
            extra={"midi_md5": str(meta.get("UUID", "")), "msd_track_id": msd.group(0) if msd else None,
                   "orig_split": orig_split(track_id), "dir_split": dir_split},
        ))
    if layout_errors:
        raise ValueError("slakh: unexpected directory layout:\n  " + "\n  ".join(layout_errors[:20]))
    collect_unknown(NAME, unknown, strict)

    # MIDI-duplicate leakage (exact md5) and song-level leakage (same MSD id, distinct MIDI)
    kept = [r for r in records if r.split != "omitted"]
    md5_splits: dict[str, set[str]] = defaultdict(set)
    md5_count = Counter(r.extra["midi_md5"] for r in kept if r.extra["midi_md5"])
    for r in kept:
        if r.extra["midi_md5"]:
            md5_splits[r.extra["midi_md5"]].add(r.split)
    cross_md5 = sorted(m for m, s in md5_splits.items() if len(s) > 1)
    if split_mode in ("redux", "split2") and cross_md5:
        raise ValueError(f"slakh: {len(cross_md5)} MIDI files span splits under {split_mode} "
                         f"(data disagrees with slakh-utils tables), e.g. {cross_md5[:3]}")
    if split_mode == "redux" and any(c > 1 for c in md5_count.values()):
        raise ValueError("slakh: duplicate MIDI within redux (expected each MIDI once)")
    msd_splits: dict[str, set[str]] = defaultdict(set)
    for r in kept:
        if r.extra["msd_track_id"]:
            msd_splits[r.extra["msd_track_id"]].add(r.split)

    config = {"dataset": NAME, "split_mode": split_mode, "include_omitted": include_omitted,
              "strict": strict, "min_loudness_lufs": min_loudness_lufs,
              "unobserved": sorted(masked),
              "mappings_sha256": mappings_sha}
    stats = {"n_tracks": len(records), "n_omitted_seen": n_omitted,
             "n_unrendered_stems": n_unrendered, "n_rendered_flag_overrides": n_flag_overrides,
             "n_plugin_fallbacks": n_fallback, "n_quiet_stems_masked": n_quiet,
             "unknown_labels": dict(unknown),
             "midi_md5_cross_split": len(cross_md5),
             "msd_id_cross_split": sum(1 for s in msd_splits.values() if len(s) > 1),
             "artist_disjoint": "unknown (Lakh MIDI has no reliable artist field)"}
    return DatasetIndex(NAME, records, config, stats)
