"""Regression tests for the arbiter's M1 fixes: raw-label refinement, stem-scoped masking,
bleed, Slakh loudness floor / GM fallback, gate comparability + per-metric tolerances,
paired bootstrap, new summary metrics."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from disstruments.ml.datasets import load_dataset
from disstruments.ml.eval import cli
from disstruments.ml.eval.harness import (EvalArrays, evaluate, label_items, label_matrices,
                                          paired_bootstrap, promotion_gate, summarize)
from disstruments.ml.eval.predictions import Predictions
from disstruments.ml.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


def _mdb(d, track, stems, artist="A", **meta):
    """stems: {sid: (instrument, [raw instruments])}"""
    st = {}
    for sid, (inst, raws) in stems.items():
        st[sid] = {"instrument": inst,
                   "raw": {f"R{i:02d}": {"instrument": r} for i, r in enumerate(raws, 1)}}
    (d / f"{track}_METADATA.yaml").write_text(yaml.safe_dump({"artist": artist, "stems": st, **meta}))


def _load1(d):
    return load_dataset("medleydb", d, generated_split=True).records[0]


# ---------------------------------------------------------------------- raw-track refinement
@pytest.mark.parametrize("stem,raw,node", [
    ("distorted electric guitar", "clean electric guitar", "guitar.electric.clean"),   # DI / reamp
    ("string section", "cello", "strings.bowed.cello"),
    ("brass section", "trumpet", "brass.trumpet"),
    ("drum machine", "kick drum", "drums.acoustic_kit"),
])
def test_non_refining_raw_label_is_unknown_not_positive(tmp_path, stem, raw, node):
    _mdb(tmp_path, "A_One", {"S01": (stem, [raw])})
    r = _load1(tmp_path)
    assert node not in r.positive and node not in r.observed
    assert node not in r.stems[0].positive and node not in r.stems[0].observed


@pytest.mark.parametrize("stem,raw,node", [
    ("auxiliary percussion", "tambourine", "percussion.tambourine"),
    ("vocalists", "female singer", "voice.sung"),
    ("woodwind section", "tenor saxophone", "woodwinds.saxophone.tenor"),   # 3.0.0 sax leaves
])
def test_refining_raw_label_is_positive(tmp_path, stem, raw, node):
    _mdb(tmp_path, "A_One", {"S01": (stem, [raw])})
    assert node in _load1(tmp_path).positive


def test_non_refining_raw_does_not_hide_a_real_positive(tmp_path):
    _mdb(tmp_path, "A_One", {"S01": ("distorted electric guitar", ["clean electric guitar"]),
                             "S02": ("clean electric guitar", [])})
    assert "guitar.electric.clean" in _load1(tmp_path).positive


# ---------------------------------------------------------------------- Main System / bleed
def test_main_system_masks_its_stem_not_the_track(tmp_path, tax):
    _mdb(tmp_path, "A_One", {"S01": ("violin", []), "S02": ("Main System", [])})
    r = _load1(tmp_path)
    (tmp_path / "A_One_METADATA.yaml").unlink()
    _mdb(tmp_path, "A_One", {"S01": ("violin", [])})
    r0 = _load1(tmp_path)
    assert r.observed == r0.observed and r.positive == r0.positive      # track untouched
    ms = {s.stem_id: s for s in r.stems}["S02"]
    assert ms.positive == frozenset() and ms.observed == frozenset()    # stem fully masked


def test_has_bleed_flag_and_exclude(tmp_path, tax):
    _mdb(tmp_path, "A_One", {"S01": ("violin", [])}, artist="A", has_bleed="yes")
    _mdb(tmp_path, "B_Two", {"S01": ("cello", [])}, artist="B", has_bleed="no")
    idx = load_dataset("medleydb", tmp_path, generated_split=True)
    by = {r.item_id: r for r in idx.records}
    assert by["A_One"].stems[0].bleed is True and by["B_Two"].stems[0].bleed is False
    assert [i for i, _, _ in label_items(idx.records, "stem", exclude_bleed=True)] == ["B_Two/S01"]
    with pytest.raises(ValueError):
        label_items(idx.records, "mix", exclude_bleed=True)


# ---------------------------------------------------------------------- Slakh
def _slakh(root, stems):
    d = root / "train" / "Track00010"
    d.mkdir(parents=True)
    (d / "metadata.yaml").write_text(yaml.safe_dump({"UUID": "u", "lmd_midi_dir": "x", "stems": stems}))


def _st(plugin, inst_class, lufs=-18.0, program=0):
    return {"audio_rendered": True, "inst_class": inst_class, "is_drum": False,
            "midi_program_name": "x", "program_num": program, "plugin_name": plugin,
            "integrated_loudness": lufs}


def test_slakh_quiet_stem_is_unknown(tmp_path):
    _slakh(tmp_path, {"S00": _st("harp.nkm", "Strings", lufs=-75.0),
                      "S01": _st("tonewheel_organ_b3.nkm", "Organ")})
    idx = load_dataset("slakh", tmp_path)
    r = idx.records[0]
    assert idx.stats["n_quiet_stems_masked"] == 1
    assert "strings.plucked.harp" not in r.positive and "strings.plucked.harp" not in r.observed
    assert "keys.organ.drawbar" in r.positive
    loud = load_dataset("slakh", tmp_path, min_loudness_lufs=-80.0).records[0]
    assert "strings.plucked.harp" in loud.positive


def test_slakh_gm_chromatic_percussion_fallback_does_not_claim_mallets(tmp_path):
    _slakh(tmp_path, {"S00": _st("unknown_dulcimer.nkm", "Chromatic Percussion", program=15)})
    r = load_dataset("slakh", tmp_path, strict=False).records[0]
    assert "percussion.mallet" not in r.positive and "percussion.mallet" not in r.observed


def test_slakh_kit_does_not_imply_cymbals_and_palm_muted_bass_not_picked(tmp_path):
    _slakh(tmp_path, {"S00": _st("pop_kit.nkm", "Drums"),
                      "S01": _st("scarbee_rickenbacker_bass_palm_muted.nkm", "Bass")})
    r = load_dataset("slakh", tmp_path).records[0]
    assert "drums.acoustic_kit" in r.positive
    assert "cymbals" not in r.positive and "cymbals" not in r.observed   # 3.0.0: OpenMIC-only
    assert "bass.electric.picked" not in r.positive and "bass.electric.picked" not in r.observed
    assert "bass.electric.slap" in r.negative                          # Scarbee: fretted, not slap


# ---------------------------------------------------------------------- summary metrics
def _recs(tax, n=40, seed=0):
    from disstruments.ml.datasets.base import Record
    rng = np.random.default_rng(seed)
    leaves = list(tax.leaves())
    out = []
    for i in range(n):
        pos = tax.close_upward(rng.choice(leaves, 3, replace=False))
        out.append(Record("medleydb", f"t{i:02d}", "test", f"a{i}", pos, frozenset(tax.nodes), ()))
    return out


def _preds(recs, tax, noise, seed):
    ids, y, _ = label_matrices(label_items(recs), tax)
    s = np.clip(y * 0.8 + 0.1 + np.random.default_rng(seed).normal(0, noise, y.shape), 0, 1)
    return Predictions(ids, list(tax.nodes), s, tax.version)


def test_summary_has_new_metrics(tax):
    recs = _recs(tax)
    s = evaluate(recs, _preds(recs, tax, 0.2, 1), tax)["summary"]
    for k in ("ece_classwise", "brier", "f1_macro_leaf_t60", "f1_micro_leaf_t60", "hier_f1_t60",
              "openmic20_map", "openmic20_ece"):
        assert s[k] is not None, k


def test_summarize_reproduces_evaluate_exactly(tax):
    recs = _recs(tax)
    res, arrays = evaluate(recs, _preds(recs, tax, 0.3, 2), tax, return_arrays=True)
    assert summarize(arrays, tax) == res["summary"]


def test_constant_prior_is_calibrated_pooled_but_not_classwise(tax):
    """Critic #10: pooled ECE rewards a useless constant; classwise ECE does not."""
    recs = _recs(tax, n=60)
    ids, y, m = label_matrices(label_items(recs), tax)
    pooled_rate = y[m].mean()
    p = Predictions(ids, list(tax.nodes), np.full(y.shape, pooled_rate), tax.version)
    s = evaluate(recs, p, tax)["summary"]
    assert s["ece"] < 0.001 and s["ece_classwise"] > 0.03


# ---------------------------------------------------------------------- bootstrap + gate
def _arrays(recs, tax, noise, seed):
    return evaluate(recs, _preds(recs, tax, noise, seed), tax, return_arrays=True)


def test_paired_bootstrap_significant_vs_not(tax):
    recs = _recs(tax, n=40)
    good, a_good = _arrays(recs, tax, 0.1, 1)
    bad, a_bad = _arrays(recs, tax, 0.6, 2)
    b = paired_bootstrap(a_good, a_bad, tax, ["map_leaf"], n_boot=200, seed=0)
    assert b["map_leaf"]["ci"][0] > 0 and b["map_leaf"]["n_valid"] == 200
    same = paired_bootstrap(a_good, a_good, tax, ["map_leaf"], n_boot=50)
    assert same["map_leaf"]["ci"] == [0.0, 0.0]
    rep = lambda r: {"config": {"dataset": "medleydb"}, **r}                    # noqa: E731
    g = promotion_gate(rep(good), rep(good), bootstrap=same)
    assert not g["promote"] and g["significance"]["significant"] is False


def test_bootstrap_rejects_different_labels(tax):
    recs = _recs(tax, n=10)
    _, a = _arrays(recs, tax, 0.1, 1)
    _, b = _arrays(_recs(tax, n=10, seed=9), tax, 0.1, 1)
    with pytest.raises(ValueError, match="labels"):
        paired_bootstrap(a, b, tax, ["map_leaf"], n_boot=5)


def _rep(summary, **cfg):
    return {"config": {"dataset": "medleydb", "split": "test", "threshold": 0.5, "bins": 15,
                       "propagate": True, "unit": "mix",
                       "loader": {"seed": 0, "mappings_sha256": "M"}, **cfg},
            "items": {"labels_hash": "L", "n_evaluated": 10},
            "predictions": {"evaluated_nodes_hash": "N"},
            "provenance": {"taxonomy_version": "2.0.0", "taxonomy_sha256": "T",
                           "mappings_sha256": "M"},
            "dataset": {"split_hash": "S"}, "summary": summary}


S = {"map_macro": 0.5, "map_leaf": 0.4, "f1_macro_leaf": 0.3, "hier_f1": 0.6, "ece": 0.1,
     "ece_classwise": 0.1, "openmic20_map": 0.5}


def test_gate_defaults_primary_by_dataset_and_ignores_loader_seed():
    cand = _rep(dict(S, map_leaf=0.45))
    inc = _rep(S, loader={"seed": 7, "mappings_sha256": "M"})
    g = promotion_gate(cand, inc)
    assert g["comparable"] and g["primary"] == "map_leaf" and g["promote"]
    om = promotion_gate(_rep(dict(S, openmic20_map=0.6), dataset="openmic"), _rep(S, dataset="openmic"))
    assert om["primary"] == "openmic20_map" and om["promote"]


@pytest.mark.parametrize("where,key,val", [
    ("provenance", "taxonomy_sha256", "T2"), ("provenance", "mappings_sha256", "M2"),
    ("dataset", "split_hash", "S2"), ("config", "unit", "stem"),
    ("config", "exclude_bleed", True),
])
def test_gate_refuses_other_eval_settings(where, key, val):
    cand = _rep(dict(S, map_leaf=0.9))
    cand[where] = dict(cand[where], **{key: val})
    assert promotion_gate(cand, _rep(S))["comparable"] is False


def test_gate_relative_tolerance_scales_with_value():
    # map_leaf primary improves; map_macro secondary may drop 2% of 0.5 = 0.010
    ok = _rep(dict(S, map_leaf=0.45, map_macro=0.49))            # -0.01 = 2% of 0.5: allowed
    bad = _rep(dict(S, map_leaf=0.45, map_macro=0.489))          # -0.011 > 0.010
    assert promotion_gate(ok, _rep(S))["promote"]
    assert not promotion_gate(bad, _rep(S))["promote"]


def test_cli_compare_requires_arrays_unless_no_bootstrap(tmp_path, tax):
    recs = _recs(tax)
    for name, noise in (("a", 0.1), ("b", 0.5)):
        res, arrays = _arrays(recs, tax, noise, 1)
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(_rep(res["summary"]), default=str))
        arrays.save(d / "arrays.npz")
    assert cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--n-boot", "50",
                     "--secondary", "map_leaf"]) == 0
    (tmp_path / "b" / "arrays.npz").unlink()
    assert cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"),
                     "--secondary", "map_leaf"]) == 1
    assert cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--no-bootstrap",
                     "--secondary", "map_leaf"]) == 0


def test_eval_arrays_roundtrip(tmp_path, tax):
    recs = _recs(tax, n=5)
    _, a = _arrays(recs, tax, 0.1, 1)
    a.save(tmp_path / "x.npz")
    b = EvalArrays.load(tmp_path / "x.npz")
    assert b.item_ids == a.item_ids and b.nodes == a.nodes and np.array_equal(b.scores, a.scores)
    assert summarize(b, tax) == summarize(a, tax)


def test_real_digest_cli_prior_vs_random(mdb_public_root, tmp_path):
    """End-to-end on real public metadata with the pinned split: prior/random baselines,
    runs with prior-reference calibration, bootstrap compare."""
    common = ["--dataset", "medleydb", "--root", str(mdb_public_root), "--split", "test"]
    for kind in ("prior", "random"):
        assert cli.main(["baseline", *common, "--kind", kind, "--out", str(tmp_path / f"{kind}.json")]) == 0
        assert cli.main(["run", *common, "--predictions", str(tmp_path / f"{kind}.json"),
                         "--runs-dir", str(tmp_path / "runs"), "--name", kind]) == 0
    runs = {d.name.split("_")[-1]: d for d in (tmp_path / "runs").iterdir()}
    rep = json.loads((runs["prior"] / "metrics.json").read_text())
    assert rep["items"]["n_evaluated"] == 29 and rep["dataset"]["artist_leakage"]["n_artists"] == 0
    ref = rep["calibration"]["prior_reference"]
    assert ref["ece"] == pytest.approx(rep["summary"]["ece"])          # prior IS the reference
    # random beats the tied-constant prior on AP (AP of a constant = prevalence) but loses
    # on calibration -> not promoted
    assert cli.main(["compare", str(runs["random"]), str(runs["prior"]), "--n-boot", "30"]) == 1


def test_stem_unit_bootstrap_resamples_whole_tracks(tax):
    """Stems of one track are correlated: resample tracks, not stems (wider, honest CI)."""
    from disstruments.ml.datasets.base import Record, StemLabels
    rng = np.random.default_rng(0)
    leaves = list(tax.leaves())
    recs = []
    for t in range(12):
        track_leaves = rng.choice(leaves, 2, replace=False)
        stems = tuple(StemLabels(f"S{k}", tax.close_upward([track_leaves[k % 2]]),
                                 frozenset(tax.nodes), ()) for k in range(8))
        pos = frozenset().union(*(s.positive for s in stems))
        recs.append(Record("medleydb", f"T{t:02d}", "test", f"a{t}", pos, frozenset(tax.nodes), (), stems=stems))
    items = label_items(recs, "stem")
    ids, y, _ = label_matrices(items, tax)
    # per-TRACK quality: a candidate that is good on some tracks, bad on others
    track_q = rng.random(12)
    noise = np.repeat(track_q, 8)[:, None] * rng.normal(0, 1, y.shape)
    s1 = np.clip(y * 0.8 + 0.1 + noise, 0, 1)
    s2 = np.clip(y * 0.8 + 0.1 + 0.8 * noise[::-1], 0, 1)
    a1 = evaluate(recs, Predictions(ids, list(tax.nodes), s1, tax.version), tax, unit="stem", return_arrays=True)[1]
    a2 = evaluate(recs, Predictions(ids, list(tax.nodes), s2, tax.version), tax, unit="stem", return_arrays=True)[1]
    b = paired_bootstrap(a1, a2, tax, ["map_leaf"], n_boot=300, seed=0)["map_leaf"]
    assert b["n_groups"] == 12 and b["n_items"] == 96
    # naive stem-level resampling for comparison (as if every stem were its own track)
    flat = lambda a: EvalArrays([i.replace("/", "_") for i in a.item_ids], a.nodes, a.y_full,   # noqa: E731
                                a.m_full, a.scores, a.threshold, a.n_bins)
    naive = paired_bootstrap(flat(a1), flat(a2), tax, ["map_leaf"], n_boot=300, seed=0)["map_leaf"]
    assert naive["n_groups"] == 96
    assert (b["ci"][1] - b["ci"][0]) > (naive["ci"][1] - naive["ci"][0])


# ---------------------------------------------------------------------- final-review fixes
from disstruments.ml.eval import metrics as M  # noqa: E402
from disstruments.ml.eval.harness import DEFAULT_SECONDARIES, MIN_VALID_FRACTION, labels_hash  # noqa: E402


def _artist_recs(tax, artists, seed=0):
    """One mix record per entry of `artists` (None = unknown artist)."""
    from disstruments.ml.datasets.base import Record
    rng = np.random.default_rng(seed)
    leaves = list(tax.leaves())
    return [Record("medleydb", f"t{i:02d}", "test", a, tax.close_upward(rng.choice(leaves, 3, replace=False)),
                   frozenset(tax.nodes), ()) for i, a in enumerate(artists)]


def test_eval_arrays_carry_artist_groups_with_track_fallback(tmp_path, tax):
    recs = _artist_recs(tax, ["A", "A", "A", "B", "B", None, None])
    _, a = _arrays(recs, tax, 0.1, 1)
    assert a.groups == ["artist:A"] * 3 + ["artist:B"] * 2 + ["track:t05", "track:t06"]
    a.save(tmp_path / "x.npz")
    assert EvalArrays.load(tmp_path / "x.npz").groups == a.groups
    b = paired_bootstrap(a, a, tax, ["map_leaf"], n_boot=20)["map_leaf"]
    assert b["n_groups"] == 4 and b["n_items"] == 7


def test_stem_unit_groups_are_artists(tax):
    from disstruments.ml.datasets.base import Record, StemLabels
    leaves = list(tax.leaves())
    recs = []
    for t, artist in enumerate(["A", "A", None]):
        stems = tuple(StemLabels(f"S{k}", tax.close_upward([leaves[k]]), frozenset(tax.nodes), ())
                      for k in range(3))
        pos = frozenset().union(*(s.positive for s in stems))
        recs.append(Record("medleydb", f"T{t}", "test", artist, pos, frozenset(tax.nodes), (), stems=stems))
    ids, y, _ = label_matrices(label_items(recs, "stem"), tax)
    p = Predictions(ids, list(tax.nodes), np.clip(y * 0.8 + 0.1, 0, 1), tax.version)
    a = evaluate(recs, p, tax, unit="stem", return_arrays=True)[1]
    assert a.groups == ["artist:A"] * 6 + ["track:T2"] * 3


def test_old_arrays_without_groups_fall_back_to_tracks(tmp_path, tax):
    recs = _artist_recs(tax, ["A"] * 6)
    _, a = _arrays(recs, tax, 0.1, 1)
    np.savez_compressed(tmp_path / "old.npz", item_ids=np.array(a.item_ids), nodes=np.array(a.nodes),
                        y_full=a.y_full, m_full=a.m_full, scores=a.scores,
                        threshold=np.array(a.threshold), n_bins=np.array(a.n_bins))
    old = EvalArrays.load(tmp_path / "old.npz")
    assert old.groups is None
    assert paired_bootstrap(old, old, tax, ["map_leaf"], n_boot=10)["map_leaf"]["n_groups"] == 6
    with pytest.raises(ValueError, match="groups"):
        paired_bootstrap(a, old, tax, ["map_leaf"], n_boot=10)     # 1 artist vs 6 tracks


def test_artist_bootstrap_is_wider_than_track_bootstrap(tax):
    """Tracks of one artist are correlated (same band, same mixing): resample artists."""
    artists = [f"a{k}" for k in range(8) for _ in range(5)]          # 8 artists x 5 tracks
    recs = _artist_recs(tax, artists, seed=3)
    ids, y, _ = label_matrices(label_items(recs), tax)
    rng = np.random.default_rng(0)
    q = np.repeat(rng.random(8), 5)[:, None]                         # per-ARTIST quality
    noise = rng.normal(0, 1, y.shape)
    s1 = np.clip(y * 0.8 + 0.1 + q * noise, 0, 1)
    s2 = np.clip(y * 0.8 + 0.1 + (1 - q) * noise, 0, 1)
    a1 = evaluate(recs, Predictions(ids, list(tax.nodes), s1, tax.version), tax, return_arrays=True)[1]
    a2 = evaluate(recs, Predictions(ids, list(tax.nodes), s2, tax.version), tax, return_arrays=True)[1]
    art = paired_bootstrap(a1, a2, tax, ["map_leaf"], n_boot=300, seed=0)["map_leaf"]
    no_groups = lambda a: EvalArrays(a.item_ids, a.nodes, a.y_full, a.m_full, a.scores,   # noqa: E731
                                     a.threshold, a.n_bins)
    trk = paired_bootstrap(no_groups(a1), no_groups(a2), tax, ["map_leaf"], n_boot=300, seed=0)["map_leaf"]
    assert art["n_groups"] == 8 and trk["n_groups"] == 40
    assert (art["ci"][1] - art["ci"][0]) > (trk["ci"][1] - trk["ci"][0])


def test_map_leaf_min3_and_micro_ap(tax):
    from disstruments.ml.datasets.base import Record
    l3, l1 = list(tax.leaves())[:2]
    # l3: 3 positives, l1: 1 positive, 8 items, everything observed
    pos = [{l3}, {l3}, {l3}, {l1}, set(), set(), set(), set()]
    recs = [Record("medleydb", f"t{i}", "test", f"a{i}", tax.close_upward(p), frozenset(tax.nodes), ())
            for i, p in enumerate(pos)]
    s = np.array([[0.9, 0.1], [0.4, 0.2], [0.6, 0.3], [0.2, 0.3],
                  [0.5, 0.8], [0.1, 0.1], [0.3, 0.2], [0.2, 0.4]])
    res = evaluate(recs, Predictions([r.item_id for r in recs], [l3, l1], s, tax.version), tax)
    y = np.array([[l3 in p, l1 in p] for p in pos])
    ap3, ap1 = M.average_precision(y[:, 0], s[:, 0]), M.average_precision(y[:, 1], s[:, 1])
    sm = res["summary"]
    assert sm["map_leaf"] == pytest.approx((ap3 + ap1) / 2)
    assert sm["map_leaf_min3"] == pytest.approx(ap3)
    assert res["macro_detail"]["map_leaf_min3"]["n_classes"] == 1
    assert res["macro_detail"]["map_leaf_min3"]["min_positives"] == 3
    assert sm["map_micro_leaf"] == pytest.approx(M.average_precision(y.ravel(), s.ravel()))
    assert res["per_node"][l3]["n_positive"] == 3 and res["per_node"][l1]["n_positive"] == 1
    assert "map_leaf_min3" not in DEFAULT_SECONDARIES and "map_micro_leaf" not in DEFAULT_SECONDARIES



def test_micro_ap_respects_mask_and_no_positives():
    y = np.array([[1, 0], [0, 1], [0, 0]], bool)
    s = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.95]])
    m = np.array([[1, 1], [1, 0], [1, 1]], bool)                  # hide (1, 1)
    assert M.micro_average_precision(y, s, m) == pytest.approx(
        M.average_precision(y[m], s[m]))
    assert M.micro_average_precision(np.zeros((2, 2), bool), s[:2], None) is None
    assert M.micro_average_precision(y, s, m, cols=[]) is None


def test_summary_md_shows_companions(tmp_path, mdb_public_root):
    common = ["--dataset", "medleydb", "--root", str(mdb_public_root), "--split", "test"]
    assert cli.main(["baseline", *common, "--kind", "random", "--out", str(tmp_path / "r.json")]) == 0
    assert cli.main(["run", *common, "--predictions", str(tmp_path / "r.json"),
                     "--runs-dir", str(tmp_path / "runs"), "--name", "r"]) == 0
    d = next((tmp_path / "runs").iterdir())
    md = (d / "summary.md").read_text()
    assert "map_leaf_min3" in md and "map_micro_leaf" in md and "observed positives" in md
    rep = json.loads((d / "metrics.json").read_text())
    n3 = rep["macro_detail"]["map_leaf_min3"]["n_classes"]
    assert 0 < n3 < rep["macro_detail"]["map_leaf"]["n_classes"]
    assert n3 == sum(1 for v in rep["per_node"].values()
                     if v["leaf"] and v["skipped"] is None and v["n_positive"] >= 3)


@pytest.mark.parametrize("n_valid,promote", [(94, False), (95, True), (100, True)])
def test_gate_requires_valid_bootstrap_fraction(n_valid, promote):
    assert MIN_VALID_FRACTION == 0.95
    boot = {"map_leaf": {"ci": [0.01, 0.05], "n_valid": n_valid, "n_boot": 100, "alpha": 0.05}}
    g = promotion_gate(_rep(dict(S, map_leaf=0.45)), _rep(S), secondaries=(), bootstrap=boot)
    assert g["promote"] is promote
    assert g["significance"]["available"] is promote and g["significance"]["significant"] is promote
    if not promote:
        assert "significance unavailable" in g["significance"]["reason"]
        assert any("unavailable" in r for r in g["reasons"])


def test_gate_zero_boot_is_unavailable():
    boot = {"map_leaf": {"ci": None, "n_valid": 0, "n_boot": 0}}
    g = promotion_gate(_rep(dict(S, map_leaf=0.45)), _rep(S), secondaries=(), bootstrap=boot)
    assert not g["promote"] and g["significance"]["available"] is False


def test_labels_hash_covers_kept_items_only(tax):
    recs = _artist_recs(tax, [f"a{i}" for i in range(6)])
    full = evaluate(recs, _preds(recs, tax, 0.1, 1), tax)
    assert full["items"]["labels_hash"] == labels_hash(label_items(recs))

    def missing(k):
        p = _preds(recs, tax, 0.1, 1)
        keep = [i for i in range(len(recs)) if i != k]
        return Predictions([p.item_ids[i] for i in keep], p.nodes, p.scores[keep], tax.version)
    r0 = evaluate(recs, missing(0), tax, allow_missing=True)
    r1 = evaluate(recs, missing(1), tax, allow_missing=True)
    assert r0["items"]["n_evaluated"] == r1["items"]["n_evaluated"] == 5
    assert r0["items"]["labels_hash"] != r1["items"]["labels_hash"]
    assert r0["items"]["labels_hash"] == labels_hash(label_items(recs[1:]))
    c0, c1 = (dict(_rep(S), items=r["items"]) for r in (r0, r1))
    assert promotion_gate(dict(c0, summary=dict(S, map_leaf=0.9)), c1)["comparable"] is False


def _two_runs(tmp_path, tax, split):
    recs = _recs(tax)
    for name, noise in (("a", 0.1), ("b", 0.5)):
        res, arrays = _arrays(recs, tax, noise, 1)
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(_rep(res["summary"], split=split), default=str))
        arrays.save(d / "arrays.npz")
    return str(tmp_path / "a"), str(tmp_path / "b")


def test_cli_no_bootstrap_and_test_split_warn(tmp_path, tax, capsys):
    a, b = _two_runs(tmp_path, tax, "test")
    assert cli.main(["compare", a, b, "--no-bootstrap", "--secondary", "map_leaf"]) == 0
    err = capsys.readouterr().err
    assert "significance was NOT tested" in err and "split=test" in err


def test_cli_val_split_with_bootstrap_does_not_warn(tmp_path, tax, capsys):
    a, b = _two_runs(tmp_path, tax, "val")
    assert cli.main(["compare", a, b, "--n-boot", "50", "--secondary", "map_leaf"]) == 0
    assert capsys.readouterr().err == ""
