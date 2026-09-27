"""Adversarial tests for M1 (taxonomy, loaders, metrics, harness, CLI).

Plain asserts encode the CORRECT behavior: a failure here is a bug. Tests named
`test_surprising_*` pin CURRENT behavior that is arguable but not clearly wrong.

Arbiter amendments (each marked "Arbiter:" inline): MedleyDB now defaults to the pinned
196-track split, so synthetic-track tests pass `generated_split=True`; the gate's default
primary is `map_leaf` and tolerances are per metric (critic #7/#9), so gate tests pin
`primary="map_macro"` and use the new exact boundaries; three `surprising` behaviors were
changed on purpose (null-instrument stem masked, leaky split files raise).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from disstruments.ml.datasets import load_dataset
from disstruments.ml.datasets import openmic as OM
from disstruments.ml.datasets.base import (Record, UnknownLabelsError, artist_disjoint_split,
                                           artist_leakage, normalize_artist)
from disstruments.ml.eval import cli
from disstruments.ml.eval import metrics as M
from disstruments.ml.eval.harness import (complete_scores, evaluate, label_items,
                                          label_matrices, promotion_gate)
from disstruments.ml.eval.predictions import Predictions, load_predictions, save_predictions
from disstruments.ml.taxonomy import Taxonomy, TaxonomyError, get_taxonomy

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ml"


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


def _mini_tax() -> Taxonomy:
    """a -> a.x -> {a.x.p, a.x.q}; a -> a.y ; b (leaf)."""
    L = {"planned": ["synthetic"]}
    return Taxonomy.from_dict({"version": "1.0.0", "nodes": [
        {"id": "a"}, {"id": "a.x"}, {"id": "a.x.p", **L}, {"id": "a.x.q", **L},
        {"id": "a.y", **L}, {"id": "b", **L}]})


# ============================================================================ metrics
class TestAP:
    def test_matches_sklearn_random_ties(self):
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(1)
        for _ in range(500):
            n = int(rng.integers(1, 30))
            y = rng.random(n) < rng.random()
            if not y.any():
                y[0] = True
            s = rng.integers(0, 4, n) / 3.0          # heavy ties, includes exact 0 and 1
            assert M.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)

    def test_all_equal_scores_is_prevalence(self):
        y = np.array([1, 0, 0, 1, 0], bool)
        assert M.average_precision(y, np.full(5, 0.7)) == pytest.approx(0.4)

    def test_single_sample_positive(self):
        assert M.average_precision([True], [0.1]) == 1.0

    def test_scores_outside_unit_interval_rank_only(self):
        y = np.array([1, 0, 1, 0], bool)
        s = np.array([0.9, 0.2, 0.6, 0.1])
        assert M.average_precision(y, s * 10 - 3) == M.average_precision(y, s)

    def test_inf_ties_are_permutation_invariant(self):
        # Two items tied at +inf must form ONE tie group; AP must not depend on input order.
        a = M.average_precision([1, 0, 1], [np.inf, np.inf, 0.3])
        b = M.average_precision([0, 1, 1], [np.inf, np.inf, 0.3])
        assert a == b

    def test_neg_inf_ties_are_permutation_invariant(self):
        a = M.average_precision([1, 1, 0], [0.9, -np.inf, -np.inf])
        b = M.average_precision([1, 0, 1], [0.9, -np.inf, -np.inf])
        assert a == b

    def test_nan_in_observed_rejected_masked_ok(self):
        y = np.array([[1], [0], [1]], bool)
        s = np.array([[0.9], [np.nan], [0.1]])
        with pytest.raises(ValueError):
            M.per_class_ap(y, s)
        ap, r = M.per_class_ap(y, s, np.array([[1], [0], [1]]))
        assert r == [M.SKIP_NO_NEGATIVES] and np.isnan(ap[0])

    def test_int_mask_equals_bool_mask(self):
        rng = np.random.default_rng(0)
        y = rng.random((40, 6)) < 0.4
        s = rng.random((40, 6))
        mb = rng.random((40, 6)) < 0.7
        a1, _ = M.per_class_ap(y, s, mb)
        a2, _ = M.per_class_ap(y.astype(int), s, mb.astype(np.int64))
        a3, _ = M.per_class_ap(y, s, mb.astype(np.uint8) * 7)      # nonzero == True
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(a1, a3)

    def test_masked_entries_do_not_affect_ap(self):
        y = np.array([[1], [0], [1], [0]], bool)
        m = np.array([[1], [1], [1], [0]], bool)
        s1 = np.array([[0.9], [0.5], [0.4], [0.99]])
        s2 = np.array([[0.9], [0.5], [0.4], [0.0]])
        assert M.per_class_ap(y, s1, m)[0][0] == M.per_class_ap(y, s2, m)[0][0]

    def test_all_masked_and_all_skipped_macro(self):
        y = np.array([[1, 0], [1, 0]], bool)
        m = np.array([[0, 1], [0, 1]], bool)
        mac = M.mean_average_precision(y, np.full((2, 2), 0.5), m)
        assert mac.value is None and mac.n_classes == 0 and mac.n_skipped == 2
        assert mac.skipped == {M.SKIP_UNOBSERVED: 1, M.SKIP_NO_POSITIVES: 1}

    def test_macro_empty_cols(self):
        mac = M.macro(np.array([0.5, 0.7]), [None, None], [])
        assert mac.value is None and mac.n_classes == 0 and mac.n_skipped == 0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            M.per_class_ap(np.zeros((2, 2)), np.zeros((2, 3)))
        with pytest.raises(ValueError):
            M.per_class_ap(np.zeros(3), np.zeros(3))


class TestPRF:
    def test_predict_nothing_micro_f1_is_zero(self):
        # a model that predicts no positives while positives exist has F1 = 0, not "undefined"
        y = np.array([[1, 0], [1, 0], [0, 1]], bool)
        prf = M.prf_at_threshold(y, np.full((3, 2), 0.1))
        assert M.micro_prf(prf)["f1"] == 0.0

    def test_predict_nothing_hier_f1_is_zero(self):
        y = np.array([[1]], bool)
        h = M.hierarchical_prf(y, np.array([[0]], bool), np.array([[1]], bool), np.eye(1, dtype=bool))
        assert h["f1"] == 0.0

    def test_threshold_inclusive_and_per_class(self):
        y = np.array([[1, 1], [0, 0], [1, 0]], bool)
        s = np.array([[0.5, 0.3], [0.49, 0.2], [0.6, 0.31]])
        prf = M.prf_at_threshold(y, s, threshold=np.array([0.5, 0.31]))
        assert prf.tp.tolist() == [2, 0] and prf.fp.tolist() == [0, 1] and prf.fn.tolist() == [0, 1]

    def test_skipped_class_fp_counted_in_micro(self):
        y = np.array([[0, 1], [0, 0]], bool)
        s = np.array([[0.9, 0.9], [0.9, 0.1]])
        prf = M.prf_at_threshold(y, s)
        assert prf.reasons[0] == M.SKIP_NO_POSITIVES and np.isnan(prf.f1[0])
        mic = M.micro_prf(prf)
        assert mic["fp"] == 2 and mic["tp"] == 1

    def test_masked_false_alarm_not_counted(self):
        y = np.array([[0], [1]], bool)
        m = np.array([[0], [1]], bool)
        prf = M.prf_at_threshold(y, np.array([[0.9], [0.9]]), m)
        assert int(prf.fp[0]) == 0


class TestHierarchical:
    def test_hand_computed_partial_credit(self):
        t = _mini_tax()
        idx = t.index()
        A = t.ancestor_matrix()
        K = len(t)
        y = np.zeros((2, K), bool)
        p = np.zeros((2, K), bool)
        # item 0: truth a.x.p, pred a.x.q  -> closures {a,a.x,a.x.p} vs {a,a.x,a.x.q}: inter 2
        y[0, idx["a.x.p"]] = True
        p[0, idx["a.x.q"]] = True
        # item 1: truth {a.y, b}, pred {a.x.p} -> {a,a.y,b} vs {a,a.x,a.x.p}: inter 1
        y[1, [idx["a.y"], idx["b"]]] = True
        p[1, idx["a.x.p"]] = True
        h = M.hierarchical_prf(y, p, np.ones_like(y), A)
        assert h["intersection"] == 3 and h["n_pred"] == 6 and h["n_true"] == 6
        assert h["precision"] == pytest.approx(0.5) and h["recall"] == pytest.approx(0.5)
        assert h["f1"] == pytest.approx(0.5)

    def test_mask_restricts_closure(self):
        t = _mini_tax()
        idx = t.index()
        y = np.zeros((1, len(t)), bool)
        p = np.zeros_like(y)
        y[0, idx["a.x.p"]] = True
        p[0, idx["a.x.p"]] = True
        m = np.ones_like(y)
        m[0, idx["a.x.p"]] = False           # leaf unknown: only ancestors count
        h = M.hierarchical_prf(y, p, m, t.ancestor_matrix())
        assert h["n_pred"] == 2 and h["n_true"] == 2 and h["f1"] == 1.0

    def test_empty_everything(self):
        t = _mini_tax()
        z = np.zeros((3, len(t)), bool)
        h = M.hierarchical_prf(z, z, np.ones_like(z), t.ancestor_matrix())
        assert h["precision"] is None and h["recall"] is None and h["f1"] is None

    def test_int_ancestor_matrix_no_overflow(self):
        # closure uses integer matmul; a dense bool matrix must not wrap
        A = np.ones((300, 300), bool)
        x = np.ones((1, 300), bool)
        h = M.hierarchical_prf(x, x, x, np.tril(A))
        assert h["f1"] == 1.0


class TestECE:
    def test_exact_zero_and_one(self):
        e = M.expected_calibration_error([0.0, 1.0, 1.0, 0.0], [0, 1, 1, 0])
        assert e["ece"] == 0.0 and e["bins"][0]["lo"] == 0.0 and e["bins"][-1]["hi"] == 1.0

    def test_empty(self):
        assert M.expected_calibration_error([], [])["ece"] is None

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            M.expected_calibration_error([1.0000001], [1])
        with pytest.raises(ValueError):
            M.expected_calibration_error([-1e-12], [0])

    def test_nan_probability_rejected(self):
        # NaN must not pass the [0,1] check and silently poison / mis-bin
        with pytest.raises(ValueError):
            M.expected_calibration_error([np.nan, 0.9], [1, 1])

    @pytest.mark.parametrize("B", [10, 15, 20])
    def test_bin_edges_land_in_upper_bin_or_close(self, B):
        # p = k/B should go to bin k; float error may push it to k-1. Measure it.
        p = np.array([k / B for k in range(B + 1)])
        e = M.expected_calibration_error(p, np.zeros_like(p, bool), B)
        for b in e["bins"]:
            assert b["lo"] - 1e-9 <= b["conf"] <= b["hi"] + 1e-9

    def test_ece_empty_bins_skipped_and_weighted(self):
        e = M.expected_calibration_error([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 0], 10)
        assert len(e["bins"]) == 2
        assert e["ece"] == pytest.approx(0.5 * 0.05 + 0.5 * abs(0.5 - 0.95))

    def test_multilabel_ece_ignores_masked(self):
        y = np.array([[1, 0], [0, 1]], bool)
        s = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = np.array([[1, 1], [1, 0]], bool)
        s2 = s.copy()
        s2[1, 1] = 0.0                       # masked entry changed
        assert M.multilabel_ece(y, s, m)["ece"] == M.multilabel_ece(y, s2, m)["ece"] == 0.0
        assert M.multilabel_ece(y, s, m)["n"] == 3


# ============================================================================ taxonomy
class TestTaxonomy:
    def test_max_propagate_invariants_random_subsets(self, tax):
        rng = np.random.default_rng(0)
        for seed in range(200):
            k = int(rng.integers(1, len(tax)))
            nodes = list(rng.choice(tax.nodes, size=k, replace=False))
            s = rng.random((5, k))
            out = tax.max_propagate(s, nodes)
            assert tax.violations(out, nodes)[0] == 0
            np.testing.assert_array_equal(tax.max_propagate(out, nodes), out)   # idempotent
            assert (out >= s).all()                                            # only raises
            # nearest present ancestor >= every present descendant (skip missing intermediates)
            pos = {n: i for i, n in enumerate(nodes)}
            for n in nodes:
                for a in tax.ancestors(n):
                    if a in pos:
                        assert (out[:, pos[a]] >= out[:, pos[n]]).all()

    def test_max_propagate_does_not_mutate_input(self, tax):
        s = np.random.default_rng(0).random((3, len(tax)))
        c = s.copy()
        tax.max_propagate(s)
        np.testing.assert_array_equal(s, c)

    def test_close_upward_idempotent(self, tax):
        rng = np.random.default_rng(1)
        for _ in range(100):
            ns = set(rng.choice(tax.nodes, size=int(rng.integers(0, 8)), replace=False))
            c = tax.close_upward(ns)
            assert tax.close_upward(c) == c and set(ns) <= c

    def test_ancestor_matrix_subset(self, tax):
        nodes = ["guitar.electric.distorted", "guitar"]          # intermediate missing
        A = tax.ancestor_matrix(nodes)
        assert A.tolist() == [[True, True], [False, True]]

    @pytest.mark.parametrize("nodes,frag", [
        ([{"id": "a.b", "planned": ["synthetic"]}, {"id": "a"}], "listed after child"),
        ([{"id": "a", "planned": ["synthetic"]}, {"id": "a", "planned": ["synthetic"]}], "duplicate"),
        ([{"id": "x.y", "planned": ["synthetic"]}], "missing"),
        ([{"id": "a"}, {"id": "a.b"}, {"id": "a.b.c"}, {"id": "a.b.c.d", "planned": ["synthetic"]}], "depth"),
        ([{"id": "Guitar", "planned": ["synthetic"]}], "bad id"),
        ([{"id": "a..b", "planned": ["synthetic"]}], "bad id"),
        ([{"id": "a", "planned": ["synthetic"]}, {"id": "a.b", "planned": ["synthetic"]}], "only allowed on leaves"),
        ([{"id": "a"}], "no data source"),
        ([{"id": "a", "data": ["youtube"]}], "unknown data sources"),
        ([], "no nodes"),
    ])
    def test_malformed_rejected(self, nodes, frag):
        with pytest.raises(TaxonomyError, match=frag):
            Taxonomy.from_dict({"version": "1.0.0", "nodes": nodes})

    def test_bad_version(self):
        with pytest.raises(TaxonomyError):
            Taxonomy.from_dict({"version": "1.0", "nodes": [{"id": "a", "planned": ["synthetic"]}]})

    def test_non_dict_node_entry_is_taxonomy_error(self):
        with pytest.raises(TaxonomyError):
            Taxonomy.from_dict({"version": "1.0.0", "nodes": ["guitar"]})

    def test_empty_document_is_taxonomy_error(self):
        with pytest.raises(TaxonomyError):
            Taxonomy.from_dict(None)

    def test_data_as_bare_string_rejected(self):
        # `data: medleydb` (string, not list) must not be silently split into characters
        with pytest.raises(TaxonomyError):
            Taxonomy.from_dict({"version": "1.0.0", "nodes": [{"id": "a", "data": "medleydb"}]})

    def test_packaged_taxonomy_openmic_antichain(self, tax):
        c2n = OM.class_to_node(tax)
        vals = list(c2n.values())
        assert len(set(vals)) == 20
        for a in vals:
            for b in vals:
                if a != b:
                    assert a not in tax.ancestors(b)


# ============================================================================ predictions
def _write_json(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return path


class TestPredictions:
    def _doc(self, tax, items, nodes=None):
        return {"format": "disstruments.predictions/v1", "taxonomy_version": tax.version,
                "nodes": nodes or ["guitar", "bass"], "items": items}

    def test_duplicate_json_item_keys_detected(self, tmp_path, tax):
        raw = ('{"format": "disstruments.predictions/v1", "taxonomy_version": "%s", '
               '"nodes": ["guitar"], "items": {"t1": [0.9], "t1": [0.1]}}' % tax.version)
        p = load_predictions(_write_json(tmp_path / "p.json", raw))
        with pytest.raises(ValueError):
            p.validate(tax)

    def test_nan_and_null_rejected(self, tmp_path, tax):
        for bad in ("NaN", "null", "Infinity"):
            raw = ('{"format": "disstruments.predictions/v1", "taxonomy_version": "%s", '
                   '"nodes": ["guitar"], "items": {"t1": [%s]}}' % (tax.version, bad))
            p = load_predictions(_write_json(tmp_path / f"p_{bad}.json", raw))
            with pytest.raises(ValueError):
                p.validate(tax)

    def test_ragged_rows_rejected(self, tmp_path, tax):
        with pytest.raises(ValueError):
            load_predictions(_write_json(tmp_path / "p.json",
                                         self._doc(tax, {"a": [0.1, 0.2], "b": [0.3]})))

    def test_unknown_and_duplicate_nodes(self, tmp_path, tax):
        for nodes in (["guitar", "kazoo"], ["guitar", "guitar"]):
            p = load_predictions(_write_json(tmp_path / "p.json",
                                             self._doc(tax, {"a": [0.1, 0.2]}, nodes)))
            with pytest.raises(ValueError):
                p.validate(tax)

    def test_out_of_range_scores(self, tmp_path, tax):
        p = load_predictions(_write_json(tmp_path / "p.json", self._doc(tax, {"a": [1.2, 0.2]})))
        with pytest.raises(ValueError, match="probabilities"):
            p.validate(tax)

    def test_major_version_mismatch_rejected_minor_warns(self, tax):
        s = np.zeros((1, 1))
        with pytest.raises(ValueError):
            Predictions(["a"], ["guitar"], s, "9.0.0").validate(tax)
        major = tax.version.split(".")[0]
        assert Predictions(["a"], ["guitar"], s, f"{major}.99.0").validate(tax)

    def test_empty_node_list_rejected(self, tax):
        # a predictions file covering zero nodes scores nothing; must not produce a report
        with pytest.raises(ValueError):
            Predictions(["a"], [], np.zeros((1, 0)), tax.version).validate(tax)

    def test_npz_roundtrip(self, tmp_path, tax):
        s = np.random.default_rng(0).random((3, len(tax)))
        f = save_predictions(tmp_path / "p.npz", ["a", "b", "c"], tax.nodes, s, tax.version, "slakh", {"m": 1})
        p = load_predictions(f)
        np.testing.assert_array_equal(p.scores, s)
        assert p.dataset == "slakh" and p.meta == {"m": 1} and p.nodes == list(tax.nodes)


# ============================================================================ harness
def _rec(item, pos, obs, tax, split="test", artist="x"):
    pos = tax.close_upward(pos)
    return Record("medleydb", item, split, artist, pos, frozenset(obs) | pos, ())


class TestHarness:
    def test_predict_nothing_reports_zero_f1_not_na(self, tax):
        recs = [_rec(f"t{i}", {"guitar.electric.distorted"}, tax.nodes, tax) for i in range(3)]
        recs.append(_rec("t9", {"bass.upright"}, tax.nodes, tax))
        p = Predictions([r.item_id for r in recs], list(tax.nodes), np.full((4, len(tax)), 0.01), tax.version)
        s = evaluate(recs, p, tax)["summary"]
        assert (s["f1_micro_leaf"], s["hier_f1"]) == (0.0, 0.0)

    def test_missing_extra_ids(self, tax):
        recs = [_rec("a", {"guitar"}, tax.nodes, tax), _rec("b", {"bass"}, tax.nodes, tax)]
        p = Predictions(["a", "zzz"], list(tax.nodes), np.full((2, len(tax)), 0.5), tax.version)
        with pytest.raises(ValueError, match="no predictions"):
            evaluate(recs, p, tax)
        r = evaluate(recs, p, tax, allow_missing=True)
        assert r["items"]["n_evaluated"] == 1 and r["items"]["n_missing_predictions"] == 1
        assert r["items"]["n_extra_predictions"] == 1

    def test_extra_ids_only_do_not_change_metrics(self, tax):
        rng = np.random.default_rng(0)
        recs = [_rec(f"t{i}", set(rng.choice(tax.leaves(), 3, replace=False)), tax.nodes, tax)
                for i in range(12)]
        ids = [r.item_id for r in recs]
        s = rng.random((12, len(tax)))
        base = evaluate(recs, Predictions(ids, list(tax.nodes), s, tax.version), tax)["summary"]
        s2 = np.vstack([s, rng.random((3, len(tax)))])
        more = evaluate(recs, Predictions(ids + ["x1", "x2", "x3"], list(tax.nodes), s2, tax.version), tax)["summary"]
        assert base == more

    def test_row_order_invariance(self, tax):
        rng = np.random.default_rng(1)
        recs = [_rec(f"t{i}", set(rng.choice(tax.leaves(), 3, replace=False)), tax.nodes, tax)
                for i in range(15)]
        ids = [r.item_id for r in recs]
        s = rng.random((15, len(tax)))
        perm = rng.permutation(15)
        a = evaluate(recs, Predictions(ids, list(tax.nodes), s, tax.version), tax)["summary"]
        b = evaluate(recs, Predictions([ids[i] for i in perm], list(tax.nodes), s[perm], tax.version), tax)["summary"]
        assert a == pytest.approx(b)

    def test_column_order_invariance(self, tax):
        rng = np.random.default_rng(2)
        recs = [_rec(f"t{i}", set(rng.choice(tax.leaves(), 3, replace=False)), tax.nodes, tax)
                for i in range(15)]
        ids = [r.item_id for r in recs]
        s = rng.random((15, len(tax)))
        perm = rng.permutation(len(tax))
        a = evaluate(recs, Predictions(ids, list(tax.nodes), s, tax.version), tax)["summary"]
        b = evaluate(recs, Predictions(ids, [tax.nodes[i] for i in perm], s[:, perm], tax.version), tax)["summary"]
        assert a == pytest.approx(b)

    def test_complete_scores_no_inf_left(self, tax):
        nodes = ["guitar.electric.distorted", "bass.upright"]
        order, out, synth = complete_scores(nodes, np.array([[0.3, 0.7]]), tax)
        assert np.isfinite(out).all()
        assert set(synth) == {"guitar", "guitar.electric", "bass"}

    def test_duplicate_record_ids_change_metrics(self, tax):
        # Arbiter: resolved by rejecting duplicates (see test_duplicate_record_ids_rejected)
        recs = [_rec("a", {"guitar"}, tax.nodes, tax), _rec("b", {"bass"}, tax.nodes, tax)]
        s = np.random.default_rng(0).random((2, len(tax)))
        p = Predictions(["a", "b"], list(tax.nodes), s, tax.version)
        evaluate(recs, p, tax)
        with pytest.raises(ValueError, match="duplicate"):
            evaluate([recs[0]] + recs, p, tax)

    def test_predict_nothing_internal_consistency(self, tax):
        recs = [_rec(f"t{i}", {"guitar.electric.distorted"}, tax.nodes, tax) for i in range(3)]
        recs.append(_rec("t9", {"bass.upright"}, tax.nodes, tax))
        p = Predictions([r.item_id for r in recs], list(tax.nodes), np.full((4, len(tax)), 0.01), tax.version)
        s = evaluate(recs, p, tax)["summary"]
        assert s["f1_macro_leaf"] == 0.0 and s["hier_recall"] == 0.0
        assert s["f1_micro_leaf"] is not None and s["hier_f1"] is not None

    def test_duplicate_record_ids_rejected(self, tax):
        # the same item twice (e.g. duplicated metadata) would be double-weighted
        recs = [_rec("a", {"guitar"}, tax.nodes, tax), _rec("a", {"guitar"}, tax.nodes, tax),
                _rec("b", {"bass"}, tax.nodes, tax)]
        p = Predictions(["a", "b"], list(tax.nodes), np.full((2, len(tax)), 0.5), tax.version)
        with pytest.raises(ValueError):
            evaluate(recs, p, tax)


# ============================================================================ promotion gate
def _report(summary, **over):
    r = {"config": {"dataset": "medleydb", "split": "test", "threshold": 0.5, "bins": 15,
                    "propagate": True, "unit": "mix"},
         "items": {"labels_hash": "L", "n_evaluated": 10},
         "predictions": {"evaluated_nodes_hash": "N"},
         "provenance": {"taxonomy_version": "1.0.0"},
         "summary": summary}
    for k, v in over.items():
        sec, key = k.split("__")
        r[sec][key] = v
    return r


def _gate(cand, inc, **kw):
    # Arbiter: default primary became map_leaf (critic #7); these tests exercise map_macro
    kw.setdefault("primary", "map_macro")
    return promotion_gate(cand, inc, **kw)


BASE = {"map_macro": 0.50, "map_leaf": 0.40, "f1_macro_leaf": 0.30, "hier_f1": 0.60, "ece": 0.10,
        "ece_level_1": 0.10}


class TestGate:
    def test_exact_tolerance_boundary_allows(self):
        # Arbiter: per-metric tolerance (critic #9): mAP-type may regress 2% of incumbent;
        # 0.40 -> 0.392 is exactly 2% (and 0.4 - 0.392 > 0.008 in float: epsilon needed)
        cand = dict(BASE, map_macro=0.6, map_leaf=0.392)
        assert _gate(_report(cand), _report(BASE))["promote"] is True

    def test_exact_tolerance_boundary_allows_ece(self):
        # Arbiter: ECE-type tolerance is 0.005 absolute (critic #9)
        cand = dict(BASE, map_macro=0.6, ece=0.105)
        assert _gate(_report(cand), _report(BASE))["promote"] is True

    def test_ece_direction(self):
        worse = dict(BASE, map_macro=0.6, ece=0.2)
        better = dict(BASE, map_macro=0.6, ece=0.0)
        assert _gate(_report(worse), _report(BASE))["promote"] is False
        assert _gate(_report(better), _report(BASE))["promote"] is True

    def test_ece_level_secondary_direction(self):
        # ece_level_N is an ECE too: lower is better. Halving it must not block promotion.
        cand = dict(BASE, map_macro=0.6, ece_level_1=0.01)
        res = _gate(_report(cand), _report(BASE),
                             secondaries=("map_leaf", "ece_level_1"))
        assert res["promote"] is True, res["reasons"]

    def test_ece_primary_lower_is_better(self):
        cand = dict(BASE, ece=0.05)
        assert _gate(_report(cand), _report(BASE), primary="ece", secondaries=())["promote"]

    def test_equal_primary_does_not_promote(self):
        assert _gate(_report(BASE), _report(BASE))["promote"] is False

    @pytest.mark.parametrize("field,val", [("provenance__taxonomy_version", "1.1.0"),
                                           ("items__labels_hash", "X"),
                                           ("items__n_evaluated", 9),
                                           ("predictions__evaluated_nodes_hash", "Y"),
                                           ("config__split", "val")])
    def test_incomparable_refused(self, field, val):
        res = _gate(_report(dict(BASE, map_macro=0.9), **{field: val}), _report(BASE))
        assert res["comparable"] is False and res["promote"] is False

    @pytest.mark.parametrize("field,val", [("config__threshold", 0.3), ("config__bins", 5),
                                           ("config__propagate", False)])
    def test_eval_settings_mismatch_refused(self, field, val):
        # threshold changes every F1; bins changes ECE: different eval settings are not comparable
        res = _gate(_report(dict(BASE, map_macro=0.9), **{field: val}), _report(BASE))
        assert res["comparable"] is False

    def test_missing_secondary_on_one_side_blocks(self):
        cand = dict(BASE, map_macro=0.9, hier_f1=None)
        res = _gate(_report(cand), _report(BASE))
        assert res["promote"] is False and res["comparable"] is True

    def test_surprising_primary_undefined_is_incomparable(self):
        res = _gate(_report(dict(BASE, map_macro=None)), _report(BASE))
        assert res["comparable"] is False       # reported as incomparable, not "worse"


# ============================================================================ artist split
class TestArtistSplit:
    def test_disjoint_under_many_seeds(self):
        rng = np.random.default_rng(0)
        groups = {f"artist{i}": int(rng.integers(1, 9)) for i in range(60)}
        for seed in range(200):
            a = artist_disjoint_split(groups, {"train": .7, "val": .15, "test": .15}, seed)
            assert set(a) == set(groups)
            c = Counter()
            for g, s in a.items():
                c[s] += groups[g]
            total = sum(groups.values())
            assert all(c[s] > 0 for s in ("train", "val", "test"))
            assert abs(c["train"] / total - 0.7) < 0.1

    def test_zero_fraction_split_gets_nothing(self):
        groups = {f"a{i}": (i % 5) + 1 for i in range(40)}
        a = artist_disjoint_split(groups, {"train": .85, "val": 0.0, "test": .15})
        assert "val" not in a.values()

    def test_deterministic(self):
        g = {f"a{i}": 1 for i in range(20)}
        f = {"train": .7, "val": .15, "test": .15}
        assert artist_disjoint_split(g, f, 3) == artist_disjoint_split(dict(reversed(g.items())), f, 3)

    def test_bad_fractions(self):
        with pytest.raises(ValueError):
            artist_disjoint_split({"a": 1}, {"train": -0.1, "test": 1.1})
        with pytest.raises(ValueError):
            artist_disjoint_split({"a": 1}, {"train": 0.0})

    @pytest.mark.parametrize("a,b", [("Music Delta", "music  delta"), ("Music Delta", "MUSIC DELTA"),
                                     ("Music Delta Multitracks", "Music Delta"),
                                     ("Sigur Rós", "Sigur Rós"),
                                     ("Guns N’ Roses", "Guns N' Roses"),
                                     ("Ｆｏｏ", "foo")])
    def test_normalize_variants(self, a, b):
        aliases = {"music delta multitracks": "music delta"}
        assert normalize_artist(a, aliases) == normalize_artist(b, aliases)

    def test_surprising_normalize_does_not_strip_the_or_punctuation(self):
        # "The Beatles" vs "Beatles", "AC/DC" vs "ACDC" stay distinct groups unless aliased
        assert normalize_artist("The Beatles") != normalize_artist("Beatles")
        assert normalize_artist("Foo Bar") != normalize_artist("FooBar")


# ============================================================================ MedleyDB
def _mdb_yaml(d: Path, track: str, artist, stems: dict):
    meta = {"stems": {sid: {"instrument": inst, "filename": f"{sid}.wav"} for sid, inst in stems.items()},
            "mix_filename": f"{track}_MIX.wav", "stem_dir": f"{track}_STEMS"}
    if artist is not None:
        meta["artist"] = artist
    (d / f"{track}_METADATA.yaml").write_text(yaml.safe_dump(meta))


class TestMedleyDB:
    def test_instrument_list_vs_string_equivalent(self, tmp_path, tax):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin", "S02": ["cello", "violin"]})
        _mdb_yaml(tmp_path, "B_Two", "B", {"S01": ["violin"], "S02": "cello"})
        idx = load_dataset("medleydb", tmp_path, generated_split=True)
        r = {x.item_id: x for x in idx.records}
        assert r["A_One"].positive == r["B_Two"].positive
        assert r["A_One"].observed == r["B_Two"].observed
        assert "strings.bowed.violin" in r["A_One"].positive and "strings" in r["A_One"].positive

    def test_unknown_label_strict_vs_nonstrict(self, tmp_path, tax):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin", "S02": "kazoo-9000"})
        with pytest.raises(UnknownLabelsError, match="kazoo-9000"):
            load_dataset("medleydb", tmp_path, generated_split=True)
        idx = load_dataset("medleydb", tmp_path, strict=False, generated_split=True)
        r = idx.records[0]
        assert r.unknown_labels == ("kazoo-9000",)
        assert r.observed == r.positive         # everything non-positive is masked
        assert idx.stats["unknown_labels"] == {"kazoo-9000": 1}

    def test_null_instrument_stem(self, tmp_path):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin", "S02": None})
        idx = load_dataset("medleydb", tmp_path, generated_split=True)
        assert "strings.bowed.violin" in idx.records[0].positive

    def test_stem_with_no_label_is_masked(self, tmp_path, tax):
        # Arbiter: changed from "all negative": a stem without an instrument label is unknown
        # content (like MedleyDB's `Unlabeled`): masked at stem AND track level
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin", "S02": None})
        r = load_dataset("medleydb", tmp_path, generated_split=True).records[0]
        st = {s.stem_id: s for s in r.stems}
        assert st["S02"].positive == frozenset() and st["S02"].observed == frozenset()
        assert r.observed == r.positive and "strings.bowed.violin" in r.positive

    def test_positive_always_observed_and_closed(self, tax):
        idx = load_dataset("medleydb", FIX / "medleydb", generated_split=True)
        for r in idx.records:
            assert r.positive <= r.observed
            assert tax.close_upward(r.positive) == r.positive
            for s in r.stems:
                assert s.positive <= s.observed and s.positive <= r.positive

    def test_real_metadata_artist_disjoint_many_seeds(self, mdb_public_root):
        # Arbiter: uses the committed 196-track digest instead of a session temp dir
        for seed in range(25):
            idx = load_dataset("medleydb", mdb_public_root, seed=seed, generated_split=True)
            assert artist_leakage(idx.records) == {}
            assert set(idx.split_counts()) == {"train", "val", "test"}
        assert artist_leakage(load_dataset("medleydb", mdb_public_root).records) == {}   # pin

    def test_alias_variants_same_split(self, tmp_path):
        for i in range(6):
            _mdb_yaml(tmp_path, f"MusicDelta_T{i}", ["Music Delta", "music delta multitracks",
                                                     "MUSIC  DELTA"][i % 3], {"S01": "violin"})
        for i in range(12):
            _mdb_yaml(tmp_path, f"Other{i}_X", f"Other {i}", {"S01": "cello"})
        for seed in range(30):
            idx = load_dataset("medleydb", tmp_path, seed=seed, generated_split=True)
            assert len({r.split for r in idx.records if r.item_id.startswith("MusicDelta")}) == 1
            assert artist_leakage(idx.records) == {}

    def test_missing_artist_fallback_groups_with_named_sibling(self, tmp_path):
        # track A has no `artist` field -> fallback to id prefix "FooBar"; sibling says
        # "Foo Bar". Same artist must not end up in two splits.
        _mdb_yaml(tmp_path, "FooBar_Song1", None, {"S01": "violin"})
        _mdb_yaml(tmp_path, "FooBar_Song2", "Foo Bar", {"S01": "violin"})
        for i in range(10):
            _mdb_yaml(tmp_path, f"Other{i}_X", f"Other {i}", {"S01": "cello"})
        leaked = False
        for seed in range(30):
            idx = load_dataset("medleydb", tmp_path, seed=seed, generated_split=True)
            sp = {r.item_id: r.split for r in idx.records}
            leaked |= sp["FooBar_Song1"] != sp["FooBar_Song2"]
        assert not leaked

    def test_duplicate_metadata_file_in_subdir_rejected(self, tmp_path):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin"})
        _mdb_yaml(tmp_path, "B_Two", "B", {"S01": "cello"})
        (tmp_path / "copy").mkdir()
        shutil.copy(tmp_path / "A_One_METADATA.yaml", tmp_path / "copy")
        with pytest.raises(ValueError):
            idx = load_dataset("medleydb", tmp_path, generated_split=True)
            ids = [r.item_id for r in idx.records]
            assert len(ids) == len(set(ids)), f"duplicate item ids {ids}"
            raise AssertionError("unreachable")      # pragma: no cover

    def test_split_file_missing_and_bad(self, tmp_path):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin"})
        _mdb_yaml(tmp_path, "B_Two", "B", {"S01": "cello"})
        sf = tmp_path / "split.json"
        sf.write_text(json.dumps({"A_One": "train"}))
        with pytest.raises(ValueError):
            load_dataset("medleydb", tmp_path, split_file=sf)
        sf.write_text(json.dumps({"A_One": "train", "B_Two": "holdout"}))
        with pytest.raises(ValueError):
            load_dataset("medleydb", tmp_path, split_file=sf)

    def test_split_file_leak_rejected_by_loader(self, tmp_path):
        # Arbiter: changed from "accepted; only reported by CLI" — a pinned split that
        # leaks an artist now raises
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin"})
        _mdb_yaml(tmp_path, "A_Two", "A", {"S01": "cello"})
        sf = tmp_path / "split.json"
        sf.write_text(json.dumps({"A_One": "train", "A_Two": "test"}))
        with pytest.raises(ValueError, match="not artist-disjoint"):
            load_dataset("medleydb", tmp_path, split_file=sf)

    def test_ignored_sampler_masks_everything(self, tmp_path):
        _mdb_yaml(tmp_path, "A_One", "A", {"S01": "violin", "S02": "sampler"})
        r = load_dataset("medleydb", tmp_path, generated_split=True).records[0]
        assert r.observed == r.positive


# ============================================================================ OpenMIC
def _openmic(root: Path, y_true, y_mask, keys, train, test, *, artists=None, pickled=True,
             part_style="csv", header=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / "openmic" / "openmic-2018" / "class-map.json", root / "class-map.json")
    sk = np.array(keys, dtype=object) if pickled else np.array(keys)
    np.savez(root / "openmic-2018.npz", X=np.zeros((len(keys), 1, 1)), Y_true=np.asarray(y_true, float),
             Y_mask=np.asarray(y_mask, bool), sample_key=sk)
    (root / "partitions").mkdir(exist_ok=True)
    names = {"csv": ("split01_train.csv", "split01_test.csv"), "txt": ("train01.txt", "test01.txt")}[part_style]
    for name, members in zip(names, (train, test)):
        (root / "partitions" / name).write_text(("sample_key\n" if header else "") + "\n".join(members) + "\n")
    if artists is not None:
        with (root / "openmic-2018-metadata.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "artist_id", "artist_name", "sample_key", "start_time"])
            for k in keys:
                w.writerow([k.split("_")[0], artists[k], f"Artist {artists[k]}", k, 0])
    return root


def _cm():
    return json.loads((FIX / "openmic" / "openmic-2018" / "class-map.json").read_text())


class TestOpenMIC:
    def _basic(self, tmp_path, **kw):
        cm = _cm()
        keys = ["000001_0", "000002_0", "000003_0"]
        yt = np.zeros((3, 20))
        ym = np.zeros((3, 20), bool)
        yt[0, cm["violin"]] = 0.5
        ym[0, cm["violin"]] = True
        yt[1, cm["violin"]] = 0.4999
        ym[1, cm["violin"]] = True
        yt[2, cm["guitar"]] = 0.9                   # unmasked-positive-looking but mask False
        ym[2, cm["piano"]] = True                   # piano known negative
        return _openmic(tmp_path / "om", yt, ym, keys, keys[:2], keys[2:], **kw)

    def test_threshold_exact_half_and_masking(self, tmp_path, tax):
        idx = load_dataset("openmic", self._basic(tmp_path))
        r = {x.item_id: x for x in idx.records}
        assert "strings.bowed.violin" in r["000001_0"].positive
        assert "strings.bowed.violin" in r["000002_0"].negative
        assert "strings.bowed.violin" not in r["000001_0"].negative
        # masked guitar value is ignored entirely
        assert not {n for n in r["000003_0"].observed if n.startswith("guitar")}
        assert set(tax.descendants("keys.piano", include_self=True)) <= r["000003_0"].negative
        assert "keys" not in r["000003_0"].observed      # other keys unknown
        assert idx.stats["sample_key_pickled"] is True

    def test_non_pickled_keys(self, tmp_path):
        idx = load_dataset("openmic", self._basic(tmp_path, pickled=False))
        assert idx.stats["sample_key_pickled"] is False and len(idx.records) == 3

    def test_txt_partitions(self, tmp_path):
        idx = load_dataset("openmic", self._basic(tmp_path, part_style="txt"))
        assert idx.split_counts() == {"train": 2, "test": 1}

    def test_partition_header_row_rejected(self, tmp_path):
        # must fail loudly (message is misleading: "sample sample_key in both partitions")
        with pytest.raises(ValueError):
            load_dataset("openmic", self._basic(tmp_path, header=True))

    def test_partition_key_missing_from_npz(self, tmp_path):
        root = self._basic(tmp_path)
        (root / "partitions" / "split01_test.csv").write_text("000003_0\n999999_0\n")
        with pytest.raises(ValueError, match="missing from npz"):
            load_dataset("openmic", root)

    def test_surprising_both_partition_styles_first_wins(self, tmp_path):
        root = self._basic(tmp_path)
        (root / "partitions" / "train01.txt").write_text("000003_0\n")   # conflicting
        (root / "partitions" / "test01.txt").write_text("000001_0\n")
        idx = load_dataset("openmic", root)
        assert idx.split_counts() == {"train": 2, "test": 1}          # csv silently wins

    def test_missing_partition_raises(self, tmp_path):
        root = self._basic(tmp_path)
        (root / "partitions" / "split01_test.csv").unlink()
        with pytest.raises(FileNotFoundError):
            load_dataset("openmic", root)

    def test_key_in_both_partitions(self, tmp_path):
        root = self._basic(tmp_path)
        (root / "partitions" / "split01_test.csv").write_text("000001_0\n000003_0\n")
        with pytest.raises(ValueError, match="both"):
            load_dataset("openmic", root)

    def test_npz_key_not_in_partitions_counted(self, tmp_path):
        root = self._basic(tmp_path)
        (root / "partitions" / "split01_train.csv").write_text("000001_0\n")
        idx = load_dataset("openmic", root)
        assert idx.stats["unassigned_in_npz"] == 1 and len(idx.records) == 2

    def test_val_carveout_artist_disjoint(self, tmp_path):
        cm = _cm()
        keys = [f"{i:06d}_0" for i in range(60)]
        yt = np.zeros((60, 20))
        ym = np.zeros((60, 20), bool)
        ym[:, cm["voice"]] = True
        yt[::2, cm["voice"]] = 1.0
        artists = {k: str(i % 13) for i, k in enumerate(keys)}
        root = _openmic(tmp_path / "om", yt, ym, keys, keys[:50], keys[50:], artists=artists)
        for seed in range(20):
            # Arbiter: this fixture's official test split shares artists with train (i % 13);
            # the (new) official-split recheck would raise, and is not what this test is about
            idx = load_dataset("openmic", root, val_fraction=0.2, seed=seed, check_split_leakage=False)
            by = {}
            for r in idx.records:
                if r.split in ("train", "val"):
                    by.setdefault(r.artist, set()).add(r.split)
            assert all(len(s) == 1 for s in by.values())
            assert idx.split_counts().get("val", 0) > 0

    def test_official_split_artist_leak_rejected(self, tmp_path):
        # Arbiter: changed from "no error" (critic #17 / tester): the loader rechecks the
        # official split's artist-disjointness when metadata.csv is present
        cm = _cm()
        keys = ["000001_0", "000002_0"]
        yt = np.zeros((2, 20))
        ym = np.zeros((2, 20), bool)
        ym[:, cm["voice"]] = True
        root = _openmic(tmp_path / "om", yt, ym, keys, keys[:1], keys[1:],
                        artists={k: "7" for k in keys})
        with pytest.raises(ValueError, match="not artist-disjoint"):
            load_dataset("openmic", root)
        idx = load_dataset("openmic", root, check_split_leakage=False)
        assert artist_leakage(idx.records) == {"fma:7": ["test", "train"]}

    def test_project_scores(self, tax):
        nodes = ["guitar.electric.distorted", "guitar.acoustic", "keys"]
        out = OM.project_scores(np.array([[0.2, 0.7, 0.9]]), nodes)
        col = OM.OPENMIC_CLASSES.index("guitar")
        assert out[0, col] == 0.7
        assert np.isnan(out[0, OM.OPENMIC_CLASSES.index("piano")])   # 'keys' projects nowhere


# ============================================================================ Slakh
def _slakh_stem(inst_class, plugin, drum=False, rendered=True, **kw):
    s = {"audio_rendered": rendered, "inst_class": inst_class, "is_drum": drum,
         "midi_program_name": "x", "program_num": 0, "integrated_loudness": -18.0}
    if plugin is not ...:
        s["plugin_name"] = plugin
    s.update(kw)
    return s


def _slakh_track(root: Path, split_dir: str | None, tid: str, stems: dict, md5="deadbeef"):
    d = (root / split_dir / tid) if split_dir else (root / tid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.yaml").write_text(yaml.safe_dump(
        {"UUID": md5, "lmd_midi_dir": "lmd/x.mid", "stems": stems}))


class TestSlakh:
    def test_fixture_duplicates_never_cross_splits(self):
        for mode in ("redux", "split2"):
            idx = load_dataset("slakh", FIX / "slakh" / "slakh2100_flac_redux", split_mode=mode)
            assert idx.stats["midi_md5_cross_split"] == 0
            by = {}
            for r in idx.records:
                if r.split != "omitted":
                    by.setdefault(r.extra["midi_md5"], set()).add(r.split)
            assert all(len(s) == 1 for s in by.values())

    def test_injected_cross_split_duplicate_raises(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010", {"S00": _slakh_stem("Bass", "harp.nkm")}, md5="same")
        _slakh_track(root, "test", "Track01900", {"S00": _slakh_stem("Bass", "harp.nkm")}, md5="same")
        with pytest.raises(ValueError, match="span splits"):
            load_dataset("slakh", root, split_mode="redux")
        idx = load_dataset("slakh", root, split_mode="orig")
        assert idx.stats["midi_md5_cross_split"] == 1

    def test_surprising_missing_plugin_name_strict_raises_as_plugin_None(self, tmp_path):
        # strict mode never uses the inst_class fallback (every fallback is "surfaced" as an
        # unknown label and strict raises); a MISSING plugin_name key is reported as 'plugin:None'
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010", {"S00": _slakh_stem("Bass", ...)})
        with pytest.raises(UnknownLabelsError, match="plugin:None"):
            load_dataset("slakh", root)
        assert "bass" in load_dataset("slakh", root, strict=False).records[0].positive

    def test_nonstrict_fallback_labels_and_masking(self, tmp_path, tax):
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010", {"S00": _slakh_stem("Bass", "brand_new_patch.nkm")})
        r = load_dataset("slakh", root, strict=False).records[0]
        assert "bass" in r.positive
        assert "plugin:brand_new_patch.nkm" in r.unknown_labels
        # fallback resolved to coarse 'bass' -> bass subtree unknown, other families negative
        assert "bass.electric" not in r.observed and "guitar" in r.negative

    def test_unknown_plugin_and_unknown_inst_class_nonstrict(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010", {"S00": _slakh_stem("Kazoo", "zz.nkm")})
        r = load_dataset("slakh", root, strict=False).records[0]
        assert r.observed == r.positive == frozenset()

    def test_is_drum_with_melodic_patch_follows_patch(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010", {"S00": _slakh_stem("Drums", "pop_kit.nkm", drum=True),
                                                   "S01": _slakh_stem("Bass", "harp.nkm", drum=True)})
        r = load_dataset("slakh", root).records[0]
        assert "drums.acoustic_kit" in r.positive and "percussion.cymbals" not in r.observed

    def test_unrendered_stem_skipped(self, tmp_path):
        root = tmp_path / "s"
        st = _slakh_stem("Bass", "None", rendered=False)
        del st["integrated_loudness"]
        _slakh_track(root, "train", "Track00010", {"S00": st, "S01": _slakh_stem("Organ", "tonewheel_organ_b3.nkm")})
        idx = load_dataset("slakh", root)
        assert idx.stats["n_unrendered_stems"] == 1 and "bass" in idx.records[0].negative

    def test_plugin_path_basename(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, "train", "Track00010",
                     {"S00": _slakh_stem("Organ", "/Users/x/Kontakt/tonewheel_organ_b3.nkm")})
        assert "keys.organ" in load_dataset("slakh", root).records[0].positive

    def test_wrong_dir_layout_raises(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, "test", "Track00010", {"S00": _slakh_stem("Organ", "tonewheel_organ_b3.nkm")})
        with pytest.raises(ValueError, match="layout"):
            load_dataset("slakh", root)

    def test_flat_babyslakh_layout(self, tmp_path):
        root = tmp_path / "s"
        _slakh_track(root, None, "Track00010", {"S00": _slakh_stem("Organ", "tonewheel_organ_b3.nkm",
                                                                    rendered=False)})
        idx = load_dataset("slakh", root)
        assert idx.records[0].split == "train" and "keys.organ" in idx.records[0].positive


# ============================================================================ CLI end-to-end
class TestCLI:
    def _mdb(self, tmp_path):
        d = tmp_path / "mdb"
        shutil.copytree(FIX / "medleydb", d)
        return d

    def test_baseline_run_compare(self, tmp_path, capsys):
        root = self._mdb(tmp_path)
        runs = tmp_path / "runs"
        prior, rnd = tmp_path / "prior.json", tmp_path / "rand.npz"
        common = ["--dataset", "medleydb", "--root", str(root), "--split", "all", "--generated-split"]
        assert cli.main(["baseline", *common, "--kind", "prior", "--fit-split", "all", "--out", str(prior)]) == 0
        assert cli.main(["baseline", *common, "--kind", "random", "--out", str(rnd)]) == 0
        assert cli.main(["run", *common, "--predictions", str(prior), "--runs-dir", str(runs), "--name", "p"]) == 0
        assert cli.main(["run", *common, "--predictions", str(rnd), "--runs-dir", str(runs), "--name", "r"]) == 0
        dirs = sorted(runs.iterdir())
        assert len(dirs) == 2
        reps = {d.name.split("_")[-1]: json.loads((d / "metrics.json").read_text()) for d in dirs}
        for r in reps.values():
            assert (Path(r["provenance"]["predictions_path"]).exists())
            assert r["dataset"]["artist_leakage"]["n_artists"] == 0
        # prior baseline: constant scores per node -> AP == prevalence
        for n, pn in reps["p"]["per_node"].items():
            if pn["ap"] is not None:
                assert pn["ap"] == pytest.approx(pn["n_positive"] / pn["n_observed"])
        rc = cli.main(["compare", str(dirs[0] / "metrics.json"), str(dirs[1] / "metrics.json"),
                       "--n-boot", "20"])
        assert rc in (0, 1)

    def test_same_second_runs_do_not_collide(self, tmp_path):
        root = self._mdb(tmp_path)
        runs = tmp_path / "runs"
        prior = tmp_path / "prior.json"
        common = ["--dataset", "medleydb", "--root", str(root), "--split", "all", "--generated-split"]
        cli.main(["baseline", *common, "--kind", "prior", "--fit-split", "all", "--out", str(prior)])
        for _ in range(3):
            cli.main(["run", *common, "--predictions", str(prior), "--runs-dir", str(runs), "--name", "same"])
        assert len(list(runs.iterdir())) == 3

    def test_config_hash_path_independent(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(), b.mkdir()
        ra, rb = self._mdb(a), self._mdb(b)
        hashes = []
        for root, base in ((ra, a), (rb, b)):
            common = ["--dataset", "medleydb", "--root", str(root), "--split", "all", "--generated-split"]
            cli.main(["baseline", *common, "--kind", "prior", "--fit-split", "all", "--out", str(base / "p.json")])
            cli.main(["run", *common, "--predictions", str(base / "p.json"), "--runs-dir", str(base / "runs")])
            d = next((base / "runs").iterdir())
            hashes.append(json.loads((d / "metrics.json").read_text())["provenance"]["config_hash"])
        assert hashes[0] == hashes[1]

    def test_predictions_for_other_dataset_rejected(self, tmp_path):
        root = self._mdb(tmp_path)
        tax = get_taxonomy()
        p = save_predictions(tmp_path / "p.json", ["x"], tax.nodes, np.zeros((1, len(tax))),
                             tax.version, "slakh")
        with pytest.raises(SystemExit):
            cli.main(["run", "--dataset", "medleydb", "--root", str(root), "--predictions", str(p),
                      "--runs-dir", str(tmp_path / "runs"), "--split", "all", "--generated-split"])

    def test_name_cannot_escape_runs_dir(self, tmp_path):
        root = self._mdb(tmp_path)
        runs = tmp_path / "runs"
        prior = tmp_path / "prior.json"
        common = ["--dataset", "medleydb", "--root", str(root), "--split", "all", "--generated-split"]
        cli.main(["baseline", *common, "--kind", "prior", "--fit-split", "all", "--out", str(prior)])
        # Arbiter: the old version could not fail (the timestamp prefix absorbed one "..");
        # "x/../../../escaped" really did resolve outside runs/. Names are now validated.
        runs.mkdir()
        for bad in ("x/../../../escaped", "../escaped", "a/b", ".."):
            with pytest.raises(SystemExit):
                cli.main(["run", *common, "--predictions", str(prior), "--runs-dir", str(runs),
                          "--name", bad])
        assert not list(tmp_path.rglob("*escaped*")) and not list(tmp_path.parent.glob("*escaped*"))
        assert list(runs.iterdir()) == []
