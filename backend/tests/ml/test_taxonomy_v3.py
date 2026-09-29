"""Taxonomy 3.0.0: one block per decision Bradley made on 2026-09-28 (see mappings.yaml /
taxonomy.yaml headers). Real-data checks use the committed 196-track MedleyDB digest, the
real 166-patch Slakh vocabulary, and the OpenMIC/Slakh fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from disstruments.ml.datasets import load_dataset, medleydb, openmic, slakh
from disstruments.ml.datasets.base import (LabelMap, apply_unobserved, load_mappings,
                                           observed_exhaustive, unobserved_roots)
from disstruments.ml.eval.audit import coverage, missingness
from disstruments.ml.eval.harness import baseline_predictions, evaluate, label_items
from disstruments.ml.eval.metrics import SKIP_NO_POSITIVES, SKIP_UNOBSERVED
from disstruments.ml.eval.predictions import Predictions
from disstruments.ml.taxonomy import TaxonomyError, get_taxonomy

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ml"
VOCAB = FIX / "vocab"
SLAKH = FIX / "slakh" / "slakh2100_flac_redux"
OM = FIX / "openmic" / "openmic-2018"


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


@pytest.fixture(scope="module")
def pub(mdb_public_root):
    return load_dataset("medleydb", mdb_public_root)


@pytest.fixture(scope="module")
def mdb_map(tax):
    return medleydb.label_map(tax)


@pytest.fixture(scope="module")
def slakh_maps(tax):
    return slakh.label_maps(tax)


def _mdb_track(root: Path, stems: dict[str, str], tid: str = "A_One") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    meta = {"artist": "A", "genre": "Rock", "has_bleed": "no",
            "stems": {sid: {"instrument": inst} for sid, inst in stems.items()}}
    (root / f"{tid}_METADATA.yaml").write_text(yaml.safe_dump(meta))
    return root


def _load_mdb1(root: Path):
    return load_dataset("medleydb", root, generated_split=True).records[0]


def _slakh_track(root: Path, stems: dict[str, tuple[str, str, float]]) -> Path:
    d = root / "train" / "Track00010"
    d.mkdir(parents=True)
    st = {sid: {"audio_rendered": True, "inst_class": cls, "is_drum": cls == "Drums",
                "midi_program_name": "x", "program_num": 0, "plugin_name": plugin,
                "integrated_loudness": lufs}
          for sid, (plugin, cls, lufs) in stems.items()}
    (d / "metadata.yaml").write_text(yaml.safe_dump({"UUID": "u", "lmd_midi_dir": "x", "stems": st}))
    return root


def test_version_and_shape(tax):
    assert tax.version == "3.0.0"
    assert len(tax) == 86 and len(tax.leaves()) == 65
    doc, _ = load_mappings(taxonomy=tax)
    assert doc["taxonomy_version"] == "3.0.0"


# ============================================================ Q1: cymbals OpenMIC-only
def test_cymbals_declared_unobserved_on_medleydb_and_slakh_only(tax):
    doc, _ = load_mappings(taxonomy=tax)
    assert unobserved_roots(doc, "medleydb", tax) == {"cymbals"}
    assert unobserved_roots(doc, "slakh", tax) == {"cymbals"}
    assert unobserved_roots(doc, "openmic", tax) == frozenset()
    assert tax.info["cymbals"].data == ("openmic",)


def test_no_kit_implies_cymbals_rule_left(mdb_map, slakh_maps):
    """The v2 kit => cymbals assumption is gone from every table. Arbiter (Alt A): nothing
    maps to cymbals on MedleyDB either; `high hat`/`cymbal` are ignore entries that mask
    drums/percussion on the carrying stem only."""
    plugins, classes = slakh_maps
    assert "cymbals" not in plugins.mapped_nodes() and "cymbals" not in classes.mapped_nodes()
    assert "cymbals" not in mdb_map.mapped_nodes()
    for label in ("high hat", "cymbal"):
        rule = mdb_map.rules[label]
        assert rule.ignored and rule.nodes == () and rule.unknown == ()
        assert rule.stem_unknown == ("drums", "percussion") and "OpenMIC only" in rule.reason


def test_cymbals_unobserved_on_every_real_medleydb_record_and_stem(pub):
    assert any({"high hat", "cymbal"} & set(r.source_labels) for r in pub.records)  # mask exercised
    for r in pub.records:
        assert "cymbals" not in r.positive and "cymbals" not in r.observed, r.item_id
        for s in r.stems:
            assert "cymbals" not in s.positive and "cymbals" not in s.observed, (r.item_id, s.stem_id)
    assert pub.config["unobserved"] == ["cymbals"]


def test_cymbal_only_stem_is_masked_not_positive(tmp_path):
    r = _load_mdb1(_mdb_track(tmp_path, {"S01": "high hat", "S02": "violin"}))
    hh = {s.stem_id: s for s in r.stems}["S01"]
    assert hh.positive == frozenset() and "cymbals" not in hh.observed
    assert "strings.bowed.violin" in r.positive and "cymbals" not in r.observed


def test_cymbals_unobserved_on_every_slakh_record_and_stem(tmp_path):
    idx = load_dataset("slakh", SLAKH)
    assert any("drums.acoustic_kit" in r.positive for r in idx.records)
    for r in idx.records:
        assert "cymbals" not in r.observed and all("cymbals" not in s.observed for s in r.stems)
    # quiet-stem path and GM `Drums` fallback path are masked too
    root = _slakh_track(tmp_path, {"S00": ("pop_kit.nkm", "Drums", -75.0),
                                   "S01": ("mystery_kit.nkm", "Drums", -18.0)})
    r = load_dataset("slakh", root, strict=False).records[0]
    assert "cymbals" not in r.observed and all("cymbals" not in s.observed for s in r.stems)
    assert "drums" in r.positive


def test_cymbals_scored_on_openmic():
    idx = load_dataset("openmic", OM)
    r = next(x for x in idx.records if x.item_id == "000139_119040")        # cymbals 0.6
    assert "cymbals" in r.positive
    assert idx.config["unobserved"] == []                  # openmic applies `unobserved` (none)
    # every clip: cymbals observed iff the annotation says so; positive iff Y_true >= 0.5
    for x in idx.records:
        lab = {l.split("=")[0]: l.split("=")[1] for l in x.source_labels}
        assert ("cymbals" in x.observed) == ("cymbals" in lab), x.item_id
        assert ("cymbals" in x.positive) == (lab.get("cymbals") == "1"), x.item_id
    # (the fixture has no cymbals=0 clip; test_adversarial_v3 covers negatives + mask)


def test_cymbals_mask_is_label_independent_and_skipped_by_harness(pub, tax):
    m = missingness(pub.records, tax)["cymbals"]
    assert m["n_observed"] == 0 and m["n_positive"] == 0 and not m["flagged"]
    test = pub.split("test")
    ids = [i for i, _, _ in label_items(test)]
    s = baseline_predictions("prior", ids, tax, train_records=pub.split("train"))
    res = evaluate(test, Predictions(ids, list(tax.nodes), s, tax.version), tax)
    assert res["per_node"]["cymbals"]["skipped"] == SKIP_UNOBSERVED
    stem_items = label_items(pub.records, "stem")
    assert not any("cymbals" in obs for _, _, obs in stem_items)


def test_unobserved_validation_and_semantics(tax):
    with pytest.raises(TaxonomyError, match="unknown node"):
        unobserved_roots({"x": {"unobserved": {"guitar.banjo": "r"}}}, "x", tax)
    with pytest.raises(TaxonomyError, match="needs a reason"):
        unobserved_roots({"x": {"unobserved": {"cymbals": ""}}}, "x", tax)
    # arbiter (Alt A): only level-1 roots; a pure mask that never strips positives
    with pytest.raises(TaxonomyError, match="level-1"):
        unobserved_roots({"x": {"unobserved": {"guitar.electric": "r"}}}, "x", tax)
    pos = tax.close_upward(["bass.electric"])
    p, obs = apply_unobserved(tax, pos, [], frozenset({"guitar"}))
    assert p == pos and pos <= obs
    assert not set(tax.close_downward(["guitar"])) & obs
    assert set(tax.close_downward(["keys"])) <= obs - p                 # rest stays known-negative


# ============================================================ Q3: sitar / beatbox collapsed
@pytest.mark.parametrize("label,parent", [("sitar", "strings.plucked"), ("beatboxing", "voice")])
def test_sitar_and_beatbox_collapse_to_parent(tax, mdb_map, label, parent, tmp_path):
    assert "strings.plucked.sitar" not in tax and "voice.beatbox" not in tax
    assert tax.info["voice.screamed"].planned == ("golden",)          # screamed kept
    res = mdb_map.resolve([label])
    assert res.mapped == {parent} and res.unknown_roots == frozenset()
    # same rule as the other out-of-scope plucked strings: parent positive, children negative
    rule, oud = mdb_map.rules[label], mdb_map.rules["oud"]
    assert (rule.nodes, rule.unknown, rule.ignored) == ((parent,), (), False)
    assert (oud.unknown, oud.ignored) == ((), False)                    # the rule it mirrors
    r = _load_mdb1(_mdb_track(tmp_path, {"S01": label}))
    assert parent in r.positive
    assert set(tax.children(parent)) <= r.negative


# ============================================================ Q4: picked/fingered synthetic-only
def test_picked_and_fingered_bass_skipped_on_real_data(pub, tax):
    for leaf in ("bass.electric.picked", "bass.electric.fingered"):
        assert tax.info[leaf].planned == ("synthetic",) and tax.info[leaf].data == ()
    cov = coverage(pub.records, tax, "medleydb")["per_leaf"]
    assert cov["bass.electric.picked"]["total"] == 0 and cov["bass.electric.fingered"]["total"] == 0
    recs = pub.split("all")
    ids = [i for i, _, _ in label_items(recs)]
    s = baseline_predictions("random", ids, tax, seed=0)
    res = evaluate(recs, Predictions(ids, list(tax.nodes), s, tax.version), tax)
    for leaf in ("bass.electric.picked", "bass.electric.fingered"):
        assert res["per_node"][leaf]["skipped"] in (SKIP_NO_POSITIVES, SKIP_UNOBSERVED), leaf


# ============================================================ Q8a: saxophone subtypes
SAX = ("soprano", "alto", "tenor", "baritone")


def test_sax_subtypes_are_leaves(tax):
    assert tax.children("woodwinds.saxophone") == tuple(f"woodwinds.saxophone.{s}" for s in SAX)
    assert all(tax.is_leaf(f"woodwinds.saxophone.{s}") for s in SAX)


@pytest.mark.parametrize("sub", SAX)
def test_medleydb_sax_labels_map_to_subtype_leaves(mdb_map, sub):
    assert mdb_map.resolve([f"{sub} saxophone"]).mapped == {f"woodwinds.saxophone.{sub}"}


def test_real_medleydb_sax_positives_per_subtype(pub):
    n = {s: sum(f"woodwinds.saxophone.{s}" in r.positive for r in pub.records) for s in SAX}
    assert n == {"soprano": 2, "alto": 2, "tenor": 16, "baritone": 1}, n
    # horn sections still leave the whole sax subtree unknown
    horn = [r for r in pub.records if "horn section" in r.source_labels
            and "woodwinds.saxophone" not in r.positive]
    assert horn and all("woodwinds.saxophone.tenor" not in r.observed for r in horn)


def test_openmic_saxophone_is_coarse_subtype_unknown():
    assert openmic.class_to_node()["saxophone"] == "woodwinds.saxophone"
    r = next(x for x in load_dataset("openmic", OM).records if x.item_id == "000312_184320")
    assert "woodwinds.saxophone" in r.positive
    assert not any(f"woodwinds.saxophone.{s}" in r.observed for s in SAX)


SLAKH_SAX = {
    "alto_sax_vintage_solo.nkm": "woodwinds.saxophone.alto",
    "alto_saxophone.nkm": "woodwinds.saxophone.alto",
    "tenor_sax.nkm": "woodwinds.saxophone.tenor",
    "tenor_saxophone.nkm": "woodwinds.saxophone.tenor",
    "baritone_sax_vintage_solo.nkm": "woodwinds.saxophone.baritone",
    "baritone_saxophone.nkm": "woodwinds.saxophone.baritone",
    "saxophone_essential.nkm": "woodwinds.saxophone",
    "saxophones_essential.nkm": "woodwinds.saxophone",
    "saxophone_section.nkm": "woodwinds.saxophone",
}


def test_slakh_sax_patches(slakh_maps, tax):
    plugins, _ = slakh_maps
    vocab = json.loads((VOCAB / "slakh_vocab.json").read_text())["patches"]
    assert {p for p in vocab if "sax" in p} == set(SLAKH_SAX)          # no soprano patch exists
    for patch, node in SLAKH_SAX.items():
        res = plugins.resolve([patch])
        assert res.mapped == {node}, patch
        obs = observed_exhaustive(tax, res.positive, res.unknown_roots)
        if node == "woodwinds.saxophone":                              # subtype unknown
            assert not any(f"{node}.{s}" in obs for s in SAX), patch
        else:                                                           # siblings negative
            assert {f"woodwinds.saxophone.{s}" for s in SAX} - {node} <= obs - res.positive
    assert "slakh" not in tax.info["woodwinds.saxophone.soprano"].data


# ============================================================ Q8b: electric pianos
def test_electric_piano_group_and_acoustic_piano_leaf(tax):
    assert tax.is_leaf("keys.piano")
    assert tax.children("keys.electric_piano") == tuple(
        f"keys.electric_piano.{s}" for s in ("rhodes", "wurlitzer", "fm"))
    assert not any(n.startswith("keys.piano.") for n in tax.nodes)


def test_openmic_piano_maps_to_acoustic_piano_with_confound_documented(tax):
    assert openmic.class_to_node(tax)["piano"] == "keys.piano"
    assert "electric piano" in tax.info["keys.piano"].note.lower()     # label-noise confound
    assert openmic.openmic_class_of("keys.electric_piano") is None


def test_medleydb_piano_labels(mdb_map, tax, tmp_path):
    assert mdb_map.resolve(["piano"]).mapped == {"keys.piano"}
    assert mdb_map.resolve(["tack piano"]).mapped == {"keys.piano"}
    r = _load_mdb1(_mdb_track(tmp_path, {"S01": "electric piano"}))
    assert "keys.electric_piano" in r.positive and "keys.piano" in r.negative
    assert not any(c in r.observed for c in tax.children("keys.electric_piano"))   # subtype unknown


def test_real_medleydb_electric_piano_tracks(pub):
    ep = [r for r in pub.records if "electric piano" in r.source_labels]
    assert len(ep) >= 5 and all("keys.electric_piano" in r.positive for r in ep)


SLAKH_EP = {
    "scarbee_a_200.nkm": "keys.electric_piano.wurlitzer",
    "wurly_ep.nkm": "keys.electric_piano.wurlitzer",
    "scarbee_mark_I.nkm": "keys.electric_piano.rhodes",
    "scarbee_pianet.nkm": "keys.electric_piano",
}


def test_slakh_electric_piano_patches(slakh_maps, tax):
    plugins, classes = slakh_maps
    for patch, node in SLAKH_EP.items():
        res = plugins.resolve([patch])
        assert res.mapped == {node}, patch
        obs = observed_exhaustive(tax, res.positive, res.unknown_roots)
        assert "keys.piano" in obs - res.positive, patch                # EP is not acoustic piano
    # Pianet: an EP that is none of our leaves -> children known negative
    res = plugins.resolve(["scarbee_pianet.nkm"])
    obs = observed_exhaustive(tax, res.positive, res.unknown_roots)
    assert set(tax.children("keys.electric_piano")) <= obs - res.positive
    # GM `Piano` fallback (programs 0-7 include EPs 4-5): e-piano unknown, not negative
    res = classes.resolve(["Piano"])
    obs = observed_exhaustive(tax, res.positive, res.unknown_roots)
    assert not set(tax.descendants("keys.electric_piano", include_self=True)) & obs
    assert "keys.piano" not in obs


# --- final-review follow-ups (2026-09-28) -------------------------------------------------

def test_raw_main_system_masks_labeled_stem_fully():
    """A raw room mic under a labeled stem masks that stem entirely (v2 behaviour kept):
    the `not base.positive` narrowing is for hi-hat/cymbal only, never for a full mask."""
    from disstruments.ml.datasets import medleydb as mdb
    from disstruments.ml.datasets.base import ALL
    from disstruments.ml.taxonomy import Taxonomy
    tax = Taxonomy.load()
    lmap = mdb.label_map(tax)
    stem = {"instrument": "violin", "raw": {"R01": {"instrument": "violin"},
                                            "R02": {"instrument": "Main System"}}}
    pos, _unk, stem_unk, *_ = mdb._resolve_stem(lmap, tax, stem)
    assert "strings.bowed.violin" in pos
    assert ALL in stem_unk


def test_raw_hihat_under_kit_keeps_stem_observed():
    from disstruments.ml.datasets import medleydb as mdb
    from disstruments.ml.taxonomy import Taxonomy
    tax = Taxonomy.load()
    stem = {"instrument": "drum set", "raw": {"R01": {"instrument": "high hat"}}}
    pos, _unk, stem_unk, *_ = mdb._resolve_stem(mdb.label_map(tax), tax, stem)
    assert "drums" in pos and not stem_unk


def test_piano_incl_ep_only_on_openmic(mdb_public_root):
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.eval import harness
    from disstruments.ml.eval.predictions import Predictions
    from disstruments.ml.taxonomy import Taxonomy
    tax = Taxonomy.load()
    idx = load_dataset("medleydb", mdb_public_root)
    test = idx.split("test")
    ids = [r.item_id for r in test]
    scores = harness.baseline_predictions("prior", ids, tax, train_records=idx.split("train"))
    preds = Predictions(ids, list(tax.nodes), scores, tax.version)
    rep = harness.evaluate(test, preds, tax)
    assert rep["openmic20"]["piano_incl_ep"] == {"ap": None, "skipped": "not_openmic"}
