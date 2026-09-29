"""Real-data audits on the committed 196-track public MedleyDB digest: pinned split,
`data:` coverage tags, and label-dependent missingness (the v1 `percussion` bug class)."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from disstruments.ml.datasets import load_dataset
from disstruments.ml.datasets.base import Record, artist_leakage
from disstruments.ml.datasets.medleydb import PINNED_SPLIT_PATH, read_split_file
from disstruments.ml.eval import cli
from disstruments.ml.eval.audit import coverage, missingness
from disstruments.ml.taxonomy import get_taxonomy

MDB_PUBLIC_DIGEST = Path(__file__).resolve().parents[1] / "fixtures" / "ml" / "medleydb_public_labels.json"


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


@pytest.fixture(scope="module")
def pub(mdb_public_root):
    return load_dataset("medleydb", mdb_public_root)          # default = pinned split


# ---------------------------------------------------------------------- pinned split
def test_pin_covers_exactly_the_canonical_public_tracks():
    canonical = set(json.loads(MDB_PUBLIC_DIGEST.read_text())["tracks"])
    pin = read_split_file(PINNED_SPLIT_PATH)
    assert len(canonical) == 196 and set(pin) == canonical
    doc = json.loads(PINNED_SPLIT_PATH.read_text())
    assert set(doc) >= {"version", "seed", "split_hash", "tracks"}   # ids + splits only
    assert all(isinstance(v, str) for v in doc["tracks"].values())


def test_pin_is_default_artist_disjoint_and_sized(pub):
    assert pub.config["split_file"] == PINNED_SPLIT_PATH.name
    assert artist_leakage(pub.records) == {}
    c = pub.split_counts()
    assert c == json.loads(PINNED_SPLIT_PATH.read_text())["counts"]
    assert 0.6 < c["train"] / 196 < 0.8 and c["val"] >= 25 and c["test"] >= 25


def test_pin_is_genre_balanced(pub):
    """Stratified pin (critic #11): every genre with >= 5 tracks appears in test."""
    genre = Counter(r.extra["genre"] for r in pub.records)
    test = Counter(r.extra["genre"] for r in pub.split("test"))
    for g, n in genre.items():
        if n >= 5:
            assert test[g] >= 1, g


def test_pin_mismatch_fails_loudly(mdb_public_root, tmp_path):
    part = tmp_path / "part"
    shutil.copytree(mdb_public_root, part)
    next(iter(sorted(part.glob("*.yaml")))).unlink()
    with pytest.raises(ValueError, match="allow_partial_split"):
        load_dataset("medleydb", part)
    sub = load_dataset("medleydb", part, allow_partial_split=True)
    assert len(sub.records) == 195 and sub.config["allow_partial_split"] is True
    extra = tmp_path / "extra"
    shutil.copytree(mdb_public_root, extra)
    shutil.copy(next(iter(sorted(extra.glob("*.yaml")))), extra / "Unpinned_Track_METADATA.yaml")
    with pytest.raises(ValueError, match="generated_split"):
        load_dataset("medleydb", extra)
    assert len(load_dataset("medleydb", extra, generated_split=True).records) == 197


# ---------------------------------------------------------------------- coverage tags
def test_medleydb_data_tags_match_real_public_examples(pub, tax):
    """`data: [medleydb]` on a leaf <=> >= 1 positive in the public release."""
    cov = coverage(pub.records, tax, "medleydb")
    assert cov["claims_without_examples"] == []
    assert cov["examples_without_claim"] == []
    s = cov["summary"]
    # 3.0.0: -1 (cymbals now OpenMIC-only) +3 (saxophone -> 4 subtype leaves) vs 2.0.0's 35.
    # Test split under the unchanged pin: 31/15 -> 30/14, both drops are cymbals.
    assert s["leaves_with_any_positive"] == 37
    assert s["leaves_test_ge1"] == 30 and s["leaves_test_ge3"] == 14


def test_every_leaf_has_a_source(tax):
    for leaf in tax.leaves():
        info = tax.info[leaf]
        assert info.data or info.planned, leaf


# ---------------------------------------------------------------------- missingness
def test_no_label_dependent_missingness_on_real_metadata(pub, tax):
    flagged = {n: v for n, v in missingness(pub.records, tax).items() if v["flagged"]}
    assert flagged == {}, {n: (round(v["rate_observed"], 3), round(v["rate_overall"], 3))
                           for n, v in flagged.items()}


def test_percussion_observed_on_kit_tracks(pub):
    kit = [r for r in pub.records if "drums.acoustic_kit" in r.positive]
    assert len(kit) > 100
    assert sum("percussion" in r.observed for r in kit) / len(kit) > 0.9


def _recs(pos_flags, obs_flags, node="percussion"):
    return [Record("x", f"i{i}", "test", None, frozenset({node} if p else ()),
                   frozenset({node} if o or p else ()), ())
            for i, (p, o) in enumerate(zip(pos_flags, obs_flags))]


def test_audit_flags_v1_percussion_pattern(tax):
    # v1 shape: 39% positive overall; negatives masked on ~half of them (kit tracks)
    pos = [True] * 39 + [False] * 61
    obs = [True] * 39 + [False] * 30 + [True] * 31
    m = missingness(_recs(pos, obs), tax)["percussion"]
    assert m["flagged"] and m["rate_observed"] == pytest.approx(39 / 70)


def test_audit_ignores_label_independent_masking(tax):
    # same masking fraction on positives and negatives -> observed rate == overall rate
    pos = [True] * 40 + [False] * 60
    obs = ([False] * 4 + [True] * 36) + ([False] * 6 + [True] * 54)
    assert not missingness(_recs(pos, obs), tax)["percussion"]["flagged"]


def test_coverage_cli(mdb_public_root, tmp_path, capsys):
    out = tmp_path / "cov.json"
    assert cli.main(["coverage", "--dataset", "medleydb", "--root", str(mdb_public_root),
                     "--out", str(out), "--strict"]) == 0
    doc = json.loads(out.read_text())
    assert doc["coverage"]["summary"]["n_items"] == 196
    assert "label-dependent missingness flagged: none" in capsys.readouterr().out


def test_stratified_split_deterministic_and_group_disjoint():
    from disstruments.ml.datasets.base import stratified_artist_split
    rng = np.random.default_rng(0)
    items = {f"t{i}": (f"g{i % 17}", [f"genre:{i % 4}", f"leaf{rng.integers(6)}"]) for i in range(80)}
    f = {"train": .7, "val": .15, "test": .15}
    a = stratified_artist_split(items, f, seed=3, n_iter=2000)
    assert a == stratified_artist_split(dict(reversed(list(items.items()))), f, seed=3, n_iter=2000)
    by_group = {}
    for t, (g, _) in items.items():
        by_group.setdefault(g, set()).add(a[t])
    assert all(len(s) == 1 for s in by_group.values())
    assert set(a.values()) == {"train", "val", "test"}


def test_make_split_cli_never_overwrites(mdb_public_root, tmp_path):
    out = tmp_path / "pin.json"
    assert cli.main(["make-split", "--root", str(mdb_public_root), "--out", str(out),
                     "--n-iter", "200"]) == 0
    doc = json.loads(out.read_text())
    assert len(doc["tracks"]) == 196
    idx = load_dataset("medleydb", mdb_public_root, split_file=out)
    assert artist_leakage(idx.records) == {}
    with pytest.raises(SystemExit):
        cli.main(["make-split", "--root", str(mdb_public_root), "--out", str(out)])
