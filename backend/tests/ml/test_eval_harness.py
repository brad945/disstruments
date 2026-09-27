"""Harness: predictions I/O, evaluation report, baselines, promotion gate, CLI end-to-end."""
import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from disstruments.ml.datasets import load_dataset
from disstruments.ml.datasets.base import Record
from disstruments.ml.eval import cli
from disstruments.ml.eval.harness import (baseline_predictions, complete_scores, evaluate,
                                          label_items, label_matrices, promotion_gate)
from disstruments.ml.eval.predictions import Predictions, load_predictions, save_predictions
from disstruments.ml.taxonomy import get_taxonomy

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ml"


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


@pytest.fixture(scope="module")
def mdb():
    return load_dataset("medleydb", FIX / "medleydb", generated_split=True)   # 2 non-public tracks


def _oracle(records, tax, noise=0.0, seed=0):
    ids, y, _ = label_matrices(label_items(records), tax)
    s = y.astype(float) * 0.9 + 0.05
    if noise:
        s = np.clip(s + np.random.default_rng(seed).normal(0, noise, s.shape), 0, 1)
    return Predictions(ids, list(tax.nodes), s, tax.version)


def test_perfect_predictions(mdb, tax):
    recs = mdb.split("all")
    res = evaluate(recs, _oracle(recs, tax), tax)
    s = res["summary"]
    assert s["map_macro"] == 1.0 and s["map_leaf"] == 1.0 and s["hier_f1"] == 1.0
    assert s["f1_macro_leaf"] == 1.0 and s["ece"] == pytest.approx(0.05)
    assert res["items"]["n_evaluated"] == len(recs)
    d = res["macro_detail"]["map_macro"]
    assert d["n_classes"] + d["n_skipped"] == len(tax)      # every node accounted for
    assert set(res["per_level"]) == {"1", "2", "3"}
    assert s["f1_micro_leaf"] == 1.0
    assert res["predictions"]["consistency_violations_raw"]["n"] == 0


def test_leaf_only_predictions_get_synthesized_parents(mdb, tax):
    recs = mdb.split("all")
    full = _oracle(recs, tax)
    leaves = list(tax.leaves())
    cols = [tax.nodes.index(n) for n in leaves]
    p = Predictions(full.item_ids, leaves, full.scores[:, cols], tax.version)
    res = evaluate(recs, p, tax)
    assert len(res["predictions"]["synthesized_nodes"]) == len(tax) - len(leaves)
    assert res["per_node"]["guitar"]["synthesized"] is True
    res_raw = evaluate(recs, p, tax, propagate=False)
    assert "guitar" not in res_raw["per_node"]
    assert res["predictions"]["evaluated_nodes_hash"] != res_raw["predictions"]["evaluated_nodes_hash"]


def test_complete_scores_max_propagates(tax):
    nodes, s, syn = complete_scores(["guitar.electric.clean", "guitar.acoustic.nylon"],
                                    np.array([[0.3, 0.8]]), tax)
    assert nodes == ["guitar", "guitar.acoustic", "guitar.acoustic.nylon", "guitar.electric",
                     "guitar.electric.clean"]
    np.testing.assert_allclose(s, [[0.8, 0.8, 0.8, 0.3, 0.3]])
    assert syn == ["guitar", "guitar.acoustic", "guitar.electric"]


def test_only_positive_nodes_are_skipped_and_kept_out_of_ece(tax):
    # OpenMIC-like: "keys" only ever observed as a positive ancestor
    def rec(i, pos, obs):
        return Record("x", f"i{i}", "test", None, frozenset(pos), frozenset(obs), ())
    recs = [rec(0, {"keys", "keys.piano"}, {"keys", "keys.piano"}),
            rec(1, set(), {"keys.piano", *tax.descendants("keys.piano")}),
            rec(2, {"keys", "keys.piano"}, {"keys", "keys.piano"})]
    p = Predictions(["i0", "i1", "i2"], ["keys", "keys.piano"],
                    np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.1]]), tax.version)
    res = evaluate(recs, p, tax)
    assert res["per_node"]["keys"]["skipped"] == "no_negatives"
    # keys.piano: neg at .9 ranked first (P=0,R=0), then tied positives at .1 (P=2/3,R=1) -> AP 2/3
    assert res["per_node"]["keys.piano"]["ap"] == pytest.approx(2 / 3)
    assert res["calibration"]["excluded_columns"] == 1
    assert res["calibration"]["pooled"]["n"] == 3               # keys.piano column only


def test_missing_and_extra_predictions(mdb, tax):
    recs = mdb.split("all")
    p = _oracle(recs, tax)
    p.item_ids = p.item_ids[:-1] + ["NotATrack"]
    with pytest.raises(ValueError, match="no predictions"):
        evaluate(recs, p, tax)
    res = evaluate(recs, p, tax, allow_missing=True)
    assert res["items"]["n_missing_predictions"] == 1 and res["items"]["n_extra_predictions"] == 1


def test_stem_unit(mdb, tax):
    recs = mdb.split("test")
    items = label_items(recs, "stem")
    ids, y, _ = label_matrices(items, tax)
    p = Predictions(ids, list(tax.nodes), y * 0.9 + 0.05, tax.version)
    res = evaluate(recs, p, tax, unit="stem")
    assert res["items"]["n_items"] == sum(len(r.stems) for r in recs)
    assert res["summary"]["hier_f1"] == 1.0


def test_predictions_roundtrip_and_validation(tmp_path, tax):
    s = np.array([[0.1, 0.9], [0.5, 0.5]])
    for ext in (".json", ".npz"):
        path = save_predictions(tmp_path / f"p{ext}", ["a", "b"], ["guitar", "voice"], s,
                                tax.version, "medleydb", {"model": "m"})
        p = load_predictions(path)
        assert p.item_ids == ["a", "b"] and p.nodes == ["guitar", "voice"]
        np.testing.assert_allclose(p.scores, s)
        assert p.meta == {"model": "m"} and p.dataset == "medleydb" and len(p.sha256) == 64
    bad = [(["guitar", "guitar.banjo"], s, tax.version, "not in taxonomy"),
           (["guitar", "voice"], s * 2, tax.version, r"\[0, 1\]"),
           (["guitar", "voice"], np.array([[np.nan, 1], [0, 0]]), tax.version, "non-finite"),
           (["guitar", "voice"], s, "9.0.0", "harness has")]
    for nodes, sc, ver, msg in bad:
        with pytest.raises(ValueError, match=msg):
            Predictions(["a", "b"], nodes, sc, ver).validate(tax)
    assert Predictions(["a", "b"], ["guitar", "voice"], s, f"{tax.major}.99.0").validate(tax)  # minor drift warns


def test_prior_baseline_scores(mdb, tax):
    train = mdb.split("train")
    s = baseline_predictions("prior", ["x", "y"], tax, train_records=train)
    g = tax.nodes.index("guitar")
    n_pos = sum("guitar" in r.positive for r in train)
    n_obs = sum("guitar" in r.observed for r in train)
    assert s.shape == (2, len(tax)) and s[0, g] == pytest.approx((n_pos + 1) / (n_obs + 2))
    r = baseline_predictions("random", ["x"], tax, seed=3)
    assert ((r >= 0) & (r <= 1)).all()
    with pytest.raises(ValueError):
        baseline_predictions("prior", ["x"], tax)


def _report(summary, **over):
    r = {"config": {"dataset": "medleydb", "split": "test"},
         "items": {"labels_hash": "L", "n_evaluated": 10},
         "predictions": {"evaluated_nodes_hash": "N"},
         "provenance": {"taxonomy_version": "1.0.0"}, "summary": summary}
    for k, v in over.items():
        r[k].update(v)
    return r


def test_promotion_gate():
    inc = _report({"map_macro": 0.50, "map_leaf": 0.40, "f1_macro_leaf": 0.30, "hier_f1": 0.6, "ece": 0.05})
    ok = _report({"map_macro": 0.52, "map_leaf": 0.39, "f1_macro_leaf": 0.30, "hier_f1": 0.6, "ece": 0.06})
    # Arbiter: default primary is now map_leaf and tolerances are per metric (critic #9);
    # this test keeps covering the explicit map_macro primary + absolute-tolerance mode.
    gate = functools.partial(promotion_gate, primary="map_macro", tolerance=0.02)
    res = gate(ok, inc)
    assert res["promote"] and res["comparable"] and res["deltas"]["map_macro"] == pytest.approx(0.02)
    same = _report(dict(inc["summary"]))
    assert not gate(same, inc)["promote"]                       # must strictly beat
    regress = _report({**ok["summary"], "map_leaf": 0.37})               # -0.03 > 0.02
    assert "map_leaf" in gate(regress, inc)["reasons"][0]
    ece_bad = _report({**ok["summary"], "ece": 0.08})                    # lower is better: +0.03
    assert not gate(ece_bad, inc)["promote"]
    # undefined for both -> skipped; undefined only for the candidate -> blocks
    both_na = gate(_report({**ok["summary"], "map_leaf": None}),
                   _report({**inc["summary"], "map_leaf": None}))
    assert both_na["promote"] and both_na["skipped_secondaries"] == ["map_leaf", "ece_classwise"]
    assert not gate(_report({**ok["summary"], "map_leaf": None}), inc)["promote"]
    other = _report(dict(ok["summary"]), predictions={"evaluated_nodes_hash": "other"})
    res = gate(other, inc)
    assert res["comparable"] is False and not res["promote"]


def test_cli_end_to_end(tmp_path, capsys):
    root = FIX / "medleydb"
    preds = tmp_path / "prior.json"
    assert cli.main(["baseline", "--kind", "prior", "--dataset", "medleydb", "--root", str(root),
                     "--split", "test", "--out", str(preds), "--generated-split"]) == 0
    assert cli.main(["run", "--dataset", "medleydb", "--root", str(root), "--split", "test",
                     "--predictions", str(preds), "--runs-dir", str(tmp_path / "runs"),
                     "--generated-split"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    rep = json.loads((run_dir / "metrics.json").read_text())
    assert (run_dir / "summary.md").exists() and run_dir.name.endswith("_prior")
    prov = rep["provenance"]
    assert prov["taxonomy_version"] == get_taxonomy().version and len(prov["predictions_sha256"]) == 64
    assert (run_dir / "arrays.npz").exists()
    assert len(prov["mappings_sha256"]) == 64 and "sha" in prov["git"]
    assert rep["dataset"]["artist_leakage"]["n_artists"] == 0
    assert rep["predictions"]["meta"]["model"] == "baseline-prior"
    assert set(rep["per_level"]) == {"1", "2", "3"} and "per_node" in rep

    # same eval from a different absolute path -> same config hash
    moved = tmp_path / "elsewhere"
    shutil.copytree(root, moved)
    cli.main(["run", "--dataset", "medleydb", "--root", str(moved), "--split", "test",
              "--predictions", str(preds), "--runs-dir", str(tmp_path / "runs"), "--name", "moved",
              "--generated-split"])
    rep2 = json.loads(next((tmp_path / "runs").glob("*_moved")).joinpath("metrics.json").read_text())
    assert rep2["provenance"]["config_hash"] == prov["config_hash"]

    # promotion gate: identical reports do not promote (must strictly beat) -> exit 1
    assert cli.main(["compare", str(run_dir / "metrics.json"),
                     str(next((tmp_path / "runs").glob("*_moved")) / "metrics.json"),
                     "--n-boot", "20"]) == 1


@pytest.mark.parametrize("dataset,root,extra", [
    ("openmic", FIX / "openmic" / "openmic-2018", []),
    ("slakh", FIX / "slakh" / "slakh2100_flac_redux", []),
])
def test_cli_other_datasets(tmp_path, dataset, root, extra):
    preds = tmp_path / "rand.npz"
    assert cli.main(["baseline", "--kind", "random", "--dataset", dataset, "--root", str(root),
                     "--split", "test", "--out", str(preds), *extra]) == 0
    assert cli.main(["run", "--dataset", dataset, "--root", str(root), "--split", "test",
                     "--predictions", str(preds), "--runs-dir", str(tmp_path / "runs"), *extra]) == 0
    rep = json.loads(next((tmp_path / "runs").iterdir()).joinpath("metrics.json").read_text())
    assert rep["config"]["dataset"] == dataset
    if dataset == "openmic":
        # only the 20 mapped columns: fewer pooled pairs than the all-node ECE
        assert 0 < rep["openmic20"]["ece_n"] < rep["calibration"]["pooled"]["n"]


def test_cli_default_runs_dir_is_data_dir(tmp_path):
    from disstruments.config import settings
    preds = tmp_path / "p.json"
    cli.main(["baseline", "--dataset", "medleydb", "--root", str(FIX / "medleydb"), "--split", "val",
              "--out", str(preds), "--generated-split"])
    _, out = cli.run_eval(cli.build_parser().parse_args(
        ["run", "--dataset", "medleydb", "--root", str(FIX / "medleydb"), "--split", "val",
         "--predictions", str(preds), "--generated-split"]))
    assert out.parent == settings.data_dir / "eval_runs"
