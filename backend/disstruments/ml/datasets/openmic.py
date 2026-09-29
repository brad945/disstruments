"""OpenMIC-2018 loader + the OpenMIC <-> taxonomy compatibility layer.

Format (cosmir/openmic-2018 README + modeling-baseline notebook): archive expands to
`openmic-2018/` with `openmic-2018.npz` (X, Y_true, Y_mask, sample_key), `class-map.json`
(class -> column), `openmic-2018-metadata.csv` (artist_id etc., joined on `sample_key`),
`partitions/` and `audio/<key[:3]>/<sample_key>.ogg`. Partition files: the README lists
`train01.txt`/`test01.txt`, the notebook and the mirdata fixtures use
`split01_train.csv`/`split01_test.csv` (headerless, one sample_key per line) — both
accepted. Y_true holds annotator-agreement probabilities; binarized at >= 0.5 as in the
official notebook. Y_mask says which labels were annotated (partial labels).

Official split is train/test only and artist-disjoint per the notebook ("artists are not
represented in both sides of the split"; built with de-duplicated artist groups, which are
coarser than `artist_id`). The loader re-checks that via metadata.csv and RAISES on any
overlap (`check_split_leakage=False` to opt out; unverified on the full release here).
Without metadata.csv the check cannot run and stats say so.
An optional artist-disjoint `val` carve-out of train is available (default off).

Taxonomy compatibility (PRD F27 "level 1 stays OpenMIC-compatible"): literal level-1
equality is impossible with 9 families (OpenMIC puts piano, organ, synth, violin, ... at
the top level), so compatibility is made precise instead:
- `CLASS_TO_NODE`: total map, every OpenMIC class -> exactly one node (antichain).
- `openmic_class_of(node)`: projection node -> OpenMIC class via the deepest mapped
  ancestor-or-self (None for nodes above/outside the mapped subtrees, e.g. `keys`).
- `project_scores`: taxonomy scores -> 20 OpenMIC columns, used to score against OpenMIC
  and against draft-1 reports (which use the OpenMIC-20 names).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..taxonomy import Taxonomy, get_taxonomy
from .base import (DatasetIndex, Record, artist_disjoint_split, assert_no_leakage,
                   check_unique_ids, load_mappings, unobserved_roots)

NAME = "openmic"
OPENMIC_CLASSES = ("accordion", "banjo", "bass", "cello", "clarinet", "cymbals", "drums",
                   "flute", "guitar", "mallet_percussion", "mandolin", "organ", "piano",
                   "saxophone", "synthesizer", "trombone", "trumpet", "ukulele", "violin",
                   "voice")
# README errata: audio for these FMA track ids is corrupted (labels are kept; flagged).
CORRUPT_TRACK_IDS = frozenset({"071826", "071827", "087435", "095253", "095259",
                               "095263", "102144", "113025", "113604", "138485"})
_PARTITIONS = {"train": ("split01_train.csv", "train01.txt"),
               "test": ("split01_test.csv", "test01.txt")}


def class_to_node(taxonomy: Taxonomy | None = None) -> dict[str, str]:
    doc, _ = load_mappings(taxonomy=taxonomy or get_taxonomy())
    return dict(doc[NAME]["classes"])


@lru_cache(maxsize=1)
def _default_projection() -> dict[str, str | None]:
    tax = get_taxonomy()
    mapped = {v: k for k, v in class_to_node(tax).items()}
    out: dict[str, str | None] = {}
    for n in tax.nodes:
        hit = [mapped[a] for a in tax.ancestors(n, include_self=True) if a in mapped]
        out[n] = hit[-1] if hit else None
    return out


def openmic_class_of(node: str) -> str | None:
    """OpenMIC class a taxonomy node counts as (deepest mapped ancestor-or-self)."""
    return _default_projection()[node]


def project_scores(scores: np.ndarray, nodes: Sequence[str]) -> np.ndarray:
    """(N, K) taxonomy scores -> (N, 20) OpenMIC scores (column order = OPENMIC_CLASSES).

    Class score = max over provided nodes projecting to that class (== the mapped node's
    score once scores are max-propagated). NaN where no provided node projects there.
    """
    out = np.full((scores.shape[0], len(OPENMIC_CLASSES)), np.nan)
    col = {c: i for i, c in enumerate(OPENMIC_CLASSES)}
    for j, n in enumerate(nodes):
        c = openmic_class_of(n)
        if c is not None:
            out[:, col[c]] = np.fmax(out[:, col[c]], scores[:, j])
    return out


_KEY = re.compile(r"^\d+_\d+$")          # sample_key = <fma track id>_<start sample>


def _read_partition(root: Path, split: str) -> list[str] | None:
    """Keys of the first existing partition file for `split` (csv style preferred)."""
    for name in _PARTITIONS[split]:
        p = root / "partitions" / name
        if p.exists():
            keys = [ln.strip().split(",")[0] for ln in p.read_text().splitlines() if ln.strip()]
            bad = [k for k in keys if not _KEY.match(k)]
            if bad:
                raise ValueError(f"openmic: {p.name} has {len(bad)} line(s) that are not sample keys "
                                 f"(<track>_<start>), e.g. {bad[:3]}; partition files are headerless")
            return keys
    return None


def _read_artists(root: Path) -> dict[str, tuple[str, str]]:
    p = root / "openmic-2018-metadata.csv"
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {row["sample_key"]: (row.get("artist_id", ""), row.get("artist_name", ""))
                for row in csv.DictReader(f)}


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """(Y_true, Y_mask, sample_keys, keys_were_pickled).

    The official build script (cosmir/openmic-2018 scripts/helper_numpy.py) documents
    `sample_key` as dtype=object (np.unique over a pandas column), which np.savez pickles.
    Label arrays are always read with pickling off; only `sample_key` falls back to
    allow_pickle=True, and only for this official file.
    """
    with np.load(path, allow_pickle=False) as npz:
        y_true = np.asarray(npz["Y_true"], dtype=float)
        y_mask = np.asarray(npz["Y_mask"], dtype=bool)
        try:
            raw, pickled = npz["sample_key"], False
        except ValueError:
            raw = None
    if raw is None:
        with np.load(path, allow_pickle=True) as npz:
            raw, pickled = npz["sample_key"], True
    keys = [k.decode() if isinstance(k, bytes) else str(k) for k in raw]
    return y_true, y_mask, keys, pickled


def load(root: Path | str, *, val_fraction: float = 0.0, seed: int = 0,
         binarize_threshold: float | None = None, taxonomy: Taxonomy | None = None,
         strict: bool = True, check_split_leakage: bool = True) -> DatasetIndex:
    """Load OpenMIC-2018 from the extracted `openmic-2018/` directory."""
    tax = taxonomy or get_taxonomy()
    doc, mappings_sha = load_mappings(taxonomy=tax)
    c2n = dict(doc[NAME]["classes"])
    # `openmic.unobserved` is applied, never ignored. Validation rejects any root with a
    # class mapped inside it, so it can only mask derived negatives.
    masked = tax.close_downward(unobserved_roots(doc, NAME, tax))
    thr = float(binarize_threshold if binarize_threshold is not None
                else doc[NAME].get("binarize_threshold", 0.5))
    root = Path(root)

    class_map = json.loads((root / "class-map.json").read_text())
    unmapped = sorted(set(class_map) - set(c2n))
    if unmapped and strict:
        raise ValueError(f"openmic: class-map.json classes without a node mapping: {unmapped}")
    y_true, y_mask, keys, pickled_keys = _read_npz(root / "openmic-2018.npz")
    check_unique_ids(NAME, keys, "openmic-2018.npz sample_key")
    if y_true.shape != y_mask.shape or y_true.shape != (len(keys), len(class_map)):
        raise ValueError(f"openmic: shape mismatch Y_true{y_true.shape} Y_mask{y_mask.shape} "
                         f"keys={len(keys)} classes={len(class_map)}")

    split_of: dict[str, str] = {}
    for split in ("train", "test"):
        members = _read_partition(root, split)
        if members is None:
            raise FileNotFoundError(f"openmic: no partition file for {split} in {root / 'partitions'}")
        for k in members:
            if k in split_of:
                raise ValueError(f"openmic: sample {k} in both partitions")
            split_of[k] = split
    not_in_npz = set(split_of) - set(keys)
    if not_in_npz:
        raise ValueError(f"openmic: {len(not_in_npz)} partition keys missing from npz, e.g. {sorted(not_in_npz)[:3]}")

    artists = _read_artists(root)
    if val_fraction > 0:
        train_keys = [k for k in keys if split_of.get(k) == "train"]
        if any(k not in artists for k in train_keys):
            raise ValueError("openmic: val carve-out needs artist ids from openmic-2018-metadata.csv")
        sizes = Counter(artists[k][0] for k in train_keys)
        assign = artist_disjoint_split(sizes, {"train": 1 - val_fraction, "val": val_fraction}, seed)
        for k in train_keys:
            split_of[k] = assign[artists[k][0]]

    records: list[Record] = []
    unassigned = 0
    for i, key in enumerate(keys):
        if key not in split_of:
            unassigned += 1
            continue
        pos_nodes, neg_roots, labels = [], [], []
        for cls, col in class_map.items():
            if cls not in c2n or not y_mask[i, col]:
                continue
            is_pos = y_true[i, col] >= thr
            labels.append(f"{cls}={'1' if is_pos else '0'}")
            (pos_nodes if is_pos else neg_roots).append(c2n[cls])
        positive = tax.close_upward(pos_nodes)
        negative = tax.close_downward(neg_roots) - positive - masked
        audio_path = root / "audio" / key[:3] / f"{key}.ogg"
        artist_id, artist_name = artists.get(key, ("", ""))
        records.append(Record(
            dataset=NAME, item_id=key, split=split_of[key],
            artist=f"fma:{artist_id}" if artist_id else None,
            positive=positive, observed=positive | negative, source_labels=tuple(labels),
            audio={"mix": audio_path} if audio_path.exists() else {},
            extra={"artist_name": artist_name,
                   "audio_corrupt": key.split("_")[0] in CORRUPT_TRACK_IDS},
        ))
    leakage_checked = check_split_leakage and bool(artists)
    if leakage_checked:
        assert_no_leakage(NAME, [r for r in records if r.split in ("train", "val", "test")],
                          "official split01 (+ val carve-out)")
    config = {"dataset": NAME, "binarize_threshold": thr, "val_fraction": val_fraction,
              "seed": seed, "unobserved": sorted(unobserved_roots(doc, NAME, tax)),
              "mappings_sha256": mappings_sha}
    stats = {"n_clips": len(records), "unassigned_in_npz": unassigned,
             "unmapped_classes": unmapped, "sample_key_pickled": pickled_keys, "n_missing_artist": sum(r.artist is None for r in records),
             "n_audio_corrupt_flagged": sum(r.extra["audio_corrupt"] for r in records),
             "split_leakage_checked": leakage_checked}
    return DatasetIndex(NAME, records, config, stats)
