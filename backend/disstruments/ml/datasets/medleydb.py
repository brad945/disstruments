"""MedleyDB 1.0 + 2.0 loader (track-level labels from per-stem + raw-track metadata).

Format (marl/medleydb): one `<TrackId>_METADATA.yaml` per track, shipped in the GitHub
repo (`medleydb/data/Metadata/`), NOT in the gated audio archive. Fields used: `artist`,
`genre`, `instrumental`, `has_bleed`, `mix_filename`, `stem_dir`, and
`stems.Sxx.{instrument, filename, raw.Ryy.{instrument}}`; `instrument` may be a string or
a list. Audio (optional) lives at `<audio_root>/<TrackId>/<mix_filename>` and
`<audio_root>/<TrackId>/<stem_dir>/<filename>` (MEDLEYDB_PATH/Audio layout).

Label rules (see mappings.yaml `medleydb`):
- Stem labels map to nodes; raw-track labels may only REFINE them (descendant-or-self of
  a node the stem label maps to). A non-refining raw label marks its nodes unknown.
- `stem_unknown` ignore rules (Main System room mics; 3.0.0 hi-hat/cymbal stems) mask the
  stem, not the track. From a raw-track label they apply only when the stem's own label
  yields no positive (a `drum set` stem with a raw `high hat` track stays fully observed).
- A stem with no instrument label is unknown content: masks everything (stem and track),
  like the `Unlabeled` label.
- `medleydb.unobserved` subtrees (3.0.0: cymbals) are unobserved on every track and stem
  (a pure mask; `high hat`/`cymbal` are ignore entries, so nothing maps there).

Splits: MedleyDB has no official split. Default = the pinned, artist-disjoint split over
the 196 public v1+v2 tracks (`medleydb_split_v1.json`, genre/per-leaf stratified; see
`base.stratified_artist_split`). Loading tracks the pin does not know raises; loading a
strict subset of the pin raises unless `allow_partial_split` (e.g. audio for part of the
release). `generated_split=True` opts out and generates a seeded artist-disjoint split
over whatever was loaded (NOT stable under adding tracks — for fixtures / experiments).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Mapping

import yaml

from ..taxonomy import Taxonomy, get_taxonomy
from .base import (ALL, SPLITS, DatasetIndex, LabelMap, Record, StemLabels, apply_unobserved,
                   artist_disjoint_split, assert_no_leakage, check_unique_ids, collect_unknown,
                   load_mappings, normalize_artist, unobserved_roots)

NAME = "medleydb"
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
PINNED_SPLIT_PATH = Path(__file__).with_name("medleydb_split_v1.json")
_SUFFIX = "_METADATA.yaml"


def label_map(taxonomy: Taxonomy | None = None) -> LabelMap:
    tax = taxonomy or get_taxonomy()
    doc, _ = load_mappings(taxonomy=tax)
    return LabelMap(NAME, doc[NAME]["map"], doc[NAME].get("ignore"), tax)


def _as_list(v) -> list[str]:
    if v is None:
        return []
    return [str(x) for x in v if x is not None] if isinstance(v, list) else [str(v)]


def artist_key(artist: str | None, track_id: str, aliases: Mapping[str, str]) -> str:
    """Grouping key: normalized artist with non-alphanumerics dropped, so a missing
    `artist` (fallback: the track-id prefix, e.g. `FooBar`) groups with `Foo Bar`."""
    name = normalize_artist(artist if artist else track_id.split("_")[0], aliases)
    return re.sub(r"[\W_]+", "", name)


def read_split_file(path: Path | str) -> dict[str, str]:
    """{track_id: split}; accepts a flat mapping or {"tracks": {...}, ...metadata}."""
    doc = json.loads(Path(path).read_text())
    tracks = doc["tracks"] if isinstance(doc, dict) and isinstance(doc.get("tracks"), dict) else doc
    bad = {v for v in tracks.values() if v not in SPLITS}
    if bad:
        raise ValueError(f"split file {Path(path).name}: bad split names {sorted(map(str, bad))}")
    return dict(tracks)


def _resolve_stem(lmap: LabelMap, tax: Taxonomy, stem: dict):
    """(positive, unknown_roots, stem_only_unknown, labels, unknown_labels, n_nonrefining)."""
    stem_labels = _as_list(stem.get("instrument"))
    raw_labels = [l for raw in (stem.get("raw") or {}).values()
                  for l in _as_list((raw or {}).get("instrument"))]
    labels = list(dict.fromkeys(stem_labels + raw_labels))
    if not stem_labels:                        # unknown content: like `Unlabeled`
        res = lmap.resolve(raw_labels)
        return (frozenset(), frozenset({ALL}), frozenset(), labels, res.unknown_labels, 0)
    base = lmap.resolve(stem_labels)
    pos, unk, stem_unk = set(base.positive), set(base.unknown_roots), set(base.stem_unknown_roots)
    unknown_labels = list(base.unknown_labels)
    refinable = tax.close_downward(base.mapped)
    nonrefining = 0
    for label in dict.fromkeys(raw_labels):
        if label in stem_labels:
            continue
        r = lmap.resolve([label])
        unknown_labels += r.unknown_labels
        unk |= r.unknown_roots
        # raw `stem_unknown` applies only when the stem label asserts nothing (a kit stem with
        # a raw hi-hat track keeps its known negatives) -- except a full mask ("*", e.g. a raw
        # `Main System` room mic), which always applies: that audio can contain anything.
        if not base.positive or ALL in r.stem_unknown_roots:
            stem_unk |= r.stem_unknown_roots
        for n in r.mapped:
            if n in refinable:
                pos |= tax.close_upward([n])
            else:                              # contradicts / is outside the stem label
                unk.add(n)
                nonrefining += 1
    return (frozenset(pos), frozenset(unk), frozenset(stem_unk), labels,
            tuple(dict.fromkeys(unknown_labels)), nonrefining)


def _bleed(v) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in ("yes", "true") else False if s in ("no", "false") else None


def load(metadata_root: Path | str, audio_root: Path | str | None = None, *,
         seed: int = 0, fractions: Mapping[str, float] | None = None,
         split_file: Path | str | None = None, generated_split: bool = False,
         allow_partial_split: bool = False, strict: bool = True,
         taxonomy: Taxonomy | None = None) -> DatasetIndex:
    """Load every `*_METADATA.yaml` under `metadata_root` (recursive).

    If `audio_root` is given, tracks without a track directory there are skipped (the
    public metadata repo lists more tracks than the audio release) and counted in stats.
    `split_file` defaults to the packaged pin; `generated_split` ignores it.
    """
    tax = taxonomy or get_taxonomy()
    doc, mappings_sha = load_mappings(taxonomy=tax)
    lmap = LabelMap(NAME, doc[NAME]["map"], doc[NAME].get("ignore"), tax)
    aliases = doc[NAME].get("artist_aliases") or {}
    masked = unobserved_roots(doc, NAME, tax)
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if set(fractions) - set(SPLITS):
        raise ValueError(f"fractions keys must be within {SPLITS}")
    if generated_split and split_file is not None:
        raise ValueError("pass either split_file or generated_split, not both")

    root = Path(metadata_root)
    files = sorted(root.rglob(f"*{_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"no *{_SUFFIX} files under {root}")
    check_unique_ids(NAME, (f.name[: -len(_SUFFIX)] for f in files), f"metadata files under {root}")
    aroot = Path(audio_root) if audio_root else None

    unknown = Counter()
    skipped_no_audio = n_nonrefining = n_unlabeled = 0
    pending: list[dict] = []
    for f in files:
        track_id = f.name[: -len(_SUFFIX)]
        meta = yaml.safe_load(f.read_text()) or {}
        track_dir = aroot / track_id if aroot else None
        if aroot and not track_dir.is_dir():
            skipped_no_audio += 1
            continue

        bleed = _bleed(meta.get("has_bleed"))
        stems: list[StemLabels] = []
        track_pos: set[str] = set()
        track_unk: set[str] = set()
        track_labels: list[str] = []
        track_unknown: list[str] = []
        for stem_id, stem in sorted((meta.get("stems") or {}).items()):
            stem = stem or {}
            pos, unk, stem_unk, labels, unk_labels, nonref = _resolve_stem(lmap, tax, stem)
            pos, stem_obs = apply_unobserved(tax, pos, unk | stem_unk, masked)
            n_nonrefining += nonref
            n_unlabeled += not _as_list(stem.get("instrument"))
            unknown.update(unk_labels)
            track_pos |= pos
            track_unk |= unk
            track_labels += labels
            track_unknown += unk_labels
            stem_audio = None
            if track_dir and meta.get("stem_dir") and stem.get("filename"):
                p = track_dir / meta["stem_dir"] / stem["filename"]
                stem_audio = p if p.exists() else None
            stems.append(StemLabels(str(stem_id), pos, stem_obs, tuple(labels), stem_audio, bleed))
        positive, observed = apply_unobserved(tax, frozenset(track_pos), track_unk, masked)
        audio = {}
        if track_dir and meta.get("mix_filename") and (track_dir / meta["mix_filename"]).exists():
            audio["mix"] = track_dir / meta["mix_filename"]
        for s in stems:
            if s.audio is not None:
                audio[s.stem_id] = s.audio
        pending.append(dict(
            item_id=track_id,
            artist=artist_key(meta.get("artist"), track_id, aliases),
            positive=positive,
            observed=observed,
            source_labels=tuple(dict.fromkeys(track_labels)),
            unknown_labels=tuple(dict.fromkeys(track_unknown)),
            audio=audio, stems=tuple(stems),
            extra={"artist_raw": meta.get("artist"), "genre": meta.get("genre"),
                   "instrumental": meta.get("instrumental"), "has_bleed": bleed,
                   "excerpt": meta.get("excerpt"), "version": str(meta.get("version"))},
        ))
    collect_unknown(NAME, unknown, strict)

    if generated_split:
        sizes = Counter(p["artist"] for p in pending)
        by_artist = artist_disjoint_split(sizes, fractions, seed)
        split_of = {p["item_id"]: by_artist[p["artist"]] for p in pending}
        split_name = None
    else:
        sf = Path(split_file) if split_file is not None else PINNED_SPLIT_PATH
        assignment = read_split_file(sf)
        loaded = {p["item_id"] for p in pending}
        unpinned = sorted(loaded - set(assignment))
        if unpinned:
            raise ValueError(
                f"medleydb: {len(unpinned)} loaded track(s) are not in split file {sf.name}, e.g. "
                f"{unpinned[:5]}. The pin covers the public v1+v2 release; for other track sets "
                f"use generated_split=True (--generated-split).")
        absent = sorted(set(assignment) - loaded)
        if absent and not allow_partial_split:
            raise ValueError(
                f"medleydb: {len(absent)} track(s) of split file {sf.name} were not loaded, e.g. "
                f"{absent[:5]} (missing metadata or audio). Pass allow_partial_split=True "
                f"(--allow-partial-split) to evaluate on the subset; reports record the split hash.")
        split_of = {t: assignment[t] for t in loaded}
        split_name = sf.name

    records = [Record(dataset=NAME, split=split_of[p["item_id"]], **p) for p in pending]
    if not generated_split:
        assert_no_leakage(NAME, records, f"split file {split_name}")
    config = {"dataset": NAME, "seed": seed if generated_split else None,
              "fractions": fractions if generated_split else None,
              "split_file": split_name, "allow_partial_split": allow_partial_split,
              "audio_required": aroot is not None, "strict": strict,
              "unobserved": sorted(masked),
              "mappings_sha256": mappings_sha}
    stats = {"n_metadata_files": len(files), "n_tracks": len(records),
             "skipped_no_audio": skipped_no_audio, "unknown_labels": dict(unknown),
             "n_artists": len({r.artist for r in records}),
             "n_nonrefining_raw_labels": n_nonrefining, "n_unlabeled_stems": n_unlabeled}
    return DatasetIndex(NAME, records, config, stats)


def make_split(index: DatasetIndex, taxonomy: Taxonomy | None = None, *, seed: int = 0,
               fractions: Mapping[str, float] | None = None, n_iter: int = 20000) -> dict:
    """Build a pinnable split document for the loaded tracks: artist-disjoint, stratified
    by genre and per-leaf positives (`base.stratified_artist_split`). Run once; commit."""
    from .base import split_hash, stratified_artist_split
    tax = taxonomy or get_taxonomy()
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    leaves = set(tax.leaves())
    items = {r.item_id: (r.artist, [f"genre:{r.extra.get('genre')}"] + sorted(r.positive & leaves))
             for r in index.records}
    assign = stratified_artist_split(items, fractions, seed=seed, n_iter=n_iter)
    recs = [Record(**{**r.__dict__, "split": assign[r.item_id]}) for r in index.records]
    assert_no_leakage(NAME, recs, "generated split")
    return {"_about": "Pinned artist-disjoint MedleyDB split over the public v1+v2 tracks "
                      "(track ids only; no audio/metadata). Stratified by genre and per-leaf "
                      "positives; generated by disstruments.ml.datasets.medleydb.make_split. "
                      "Never regenerate in place: a new pin is a new file/version.",
            "version": "medleydb_split_v1", "seed": seed, "n_iter": n_iter,
            "fractions": fractions, "taxonomy_version": tax.version,
            "split_hash": split_hash(recs),
            "counts": dict(sorted(Counter(assign.values()).items())),
            "tracks": dict(sorted(assign.items()))}
