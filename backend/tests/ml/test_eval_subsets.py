"""Node-subset scoring (golden set) and report-vs-labels scoring."""
import numpy as np
import pytest

from disstruments.ml.datasets.base import Record
from disstruments.ml.eval import endtoend as EE
from disstruments.ml.eval import subsets as SS
from disstruments.ml.eval.harness import EvalArrays, label_items, label_matrices, paired_bootstrap
from disstruments.ml.eval.metrics import average_precision
from disstruments.ml.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


def _rec(tax, i, artist, present, absent_all=True):
    pos = frozenset(tax.close_upward(present))
    return Record(dataset="golden", item_id=f"s{i}", split="test", artist=artist, positive=pos,
                  source_labels=(), observed=frozenset(tax.nodes) if absent_all else pos)


def _arrays(tax, recs, nodes, scores):
    ids, y, m = label_matrices(label_items(recs), tax)
    return EvalArrays(ids, list(nodes), y, m, np.asarray(scores, float), 0.5, 15,
                      [f"artist:{r.artist}" for r in recs])


def test_expected_random_ap_matches_simulation():
    rng = np.random.default_rng(0)
    for n, k in [(25, 3), (12, 6), (7, 1), (5, 5)]:
        y = np.zeros(n, bool)
        y[:k] = True
        sim = np.mean([average_precision(y, rng.random(n)) for _ in range(4000)])
        assert SS.expected_random_ap(n, k) == pytest.approx(sim, abs=0.01)
    assert SS.expected_random_ap(5, 5) == pytest.approx(1.0)
    assert SS.expected_random_ap(25, 3) > 3 / 25                 # above prevalence at small n
    with pytest.raises(ValueError):
        SS.expected_random_ap(5, 0)


def test_eligible_nodes_needs_positives_and_negatives(tax):
    recs = [_rec(tax, i, f"a{i}", ["guitar.electric.clean"] if i < 4 else ["keys.piano"]) for i in range(6)]
    el = SS.eligible_nodes(recs, tax, trained=["guitar", "guitar.electric.clean", "keys", "keys.organ"])
    # guitar: 4 pos / 2 neg -> out; clean: same; keys: 2 pos -> out; organ: 0 pos -> out
    assert el == []
    el = SS.eligible_nodes(recs, tax, trained=["guitar", "keys"], min_pos=2, min_neg=2)
    assert el == ["guitar", "keys"]
    assert SS.eligible_nodes(recs, tax, trained=["guitar", "guitar.electric"], min_pos=2,
                             min_neg=2, levels=[2]) == ["guitar.electric"]


def test_restrict_constant_and_per_node(tax):
    recs = [_rec(tax, i, f"a{i}", ["guitar"] if i % 2 else ["keys.piano"]) for i in range(6)]
    a = _arrays(tax, recs, ["guitar", "keys"], [[0.1, 0.9], [0.8, 0.2], [0.2, 0.7],
                                                [0.9, 0.1], [0.3, 0.6], [0.7, 0.3]])
    r = SS.restrict(a, ["keys"])
    assert r.nodes == ["keys"] and r.scores.shape == (6, 1)
    with pytest.raises(ValueError):
        SS.restrict(a, ["bass"])
    pn = SS.per_node(a, tax)
    assert pn["guitar"]["ap"] == pytest.approx(1.0) and pn["guitar"]["prevalence"] == 0.5
    assert pn["keys"]["random_ap"] == pytest.approx(SS.expected_random_ap(6, 3))
    c = SS.per_node(SS.constant_arrays(a), tax)
    assert c["guitar"]["ap"] == pytest.approx(0.5)               # constant scores: AP = prevalence


def test_bootstrap_ci_uses_paired_bootstrap_resamples(tax):
    rng = np.random.default_rng(1)
    recs = [_rec(tax, i, f"a{i // 2}", ["guitar"] if rng.random() < 0.5 else ["keys"]) for i in range(16)]
    a = _arrays(tax, recs, ["guitar", "keys"], rng.random((16, 2)))
    b = SS.constant_arrays(a)
    ca = SS.bootstrap_ci(a, tax, ["map_macro"], n_boot=200, seed=3, return_draws=True)["map_macro"]
    cb = SS.bootstrap_ci(b, tax, ["map_macro"], n_boot=200, seed=3, return_draws=True)["map_macro"]
    pb = paired_bootstrap(a, b, tax, ["map_macro"], n_boot=200, seed=3)["map_macro"]
    assert ca["n_groups"] == 8 and ca["ci"][0] <= ca["value"] <= ca["ci"][1]
    assert ca["n_valid"] == cb["n_valid"] == pb["n_valid"] == 200
    assert pb["delta"] == pytest.approx(ca["value"] - cb["value"])
    # same resamples, draw for draw: the paired CI is the quantiles of the differences
    d = ca["draws"] - cb["draws"]
    assert pb["ci"] == pytest.approx([np.quantile(d, 0.025), np.quantile(d, 0.975)])


def test_positive_ranks(tax):
    recs = [_rec(tax, i, f"a{i}", ["brass.trumpet"] if i in (1, 3) else ["guitar"]) for i in range(5)]
    a = _arrays(tax, recs, ["brass"], [[0.5], [0.9], [0.95], [0.1], [0.2]])
    got = {d["item_id"]: d["rank"] for d in SS.positive_ranks(a, tax, "brass")}
    assert got == {"s1": 2, "s3": 5}


def test_score_report_entries_families_and_subtypes(tax):
    pos = frozenset(tax.close_upward(["guitar.electric.clean", "voice.sung", "keys.electric_piano"]))
    unknown = set(tax.descendants("guitar.acoustic"))
    obs = frozenset(set(tax.nodes) - unknown)
    report = [
        {"label": "voice", "node": "voice", "confidence": 0.95},
        {"label": "Clean electric guitar", "node": "guitar.electric.clean", "confidence": 0.8, "subtype": True},
        {"label": "piano", "node": "keys.piano", "confidence": 0.7},
        {"label": "drums", "node": "drums", "confidence": 0.4},
        {"label": "Nylon-string guitar", "node": "guitar.acoustic.nylon", "confidence": 0.5, "subtype": True},
        {"label": "bass (subtype uncertain)", "node": "bass", "confidence": 0.35, "subtype_uncertain": True},
        {"label": "tambourine", "node": "percussion.tambourine", "confidence": 0.9, "calibrated": False},
    ]
    e = EE.score_entries(report, pos, obs, tax)
    by = {x["label"]: x for x in e}
    assert by["voice"]["outcome"] == "hit" and by["voice"]["shown_as"] == "confident"
    assert by["Clean electric guitar"]["shown_as"] == "uncertain"          # subtypes always "?"
    assert by["piano"]["outcome"] == "false_alarm" and by["piano"]["family"] == "keys"
    assert by["drums"]["outcome"] == "false_alarm" and by["drums"]["shown_as"] == "uncertain"
    assert by["Nylon-string guitar"]["outcome"] == "unlabelled"
    assert by["tambourine"]["shown_as"] == "uncertain", "uncalibrated labels are always '?'"
    fam = EE.node_outcomes(e, ["keys", "drums", "brass", "guitar"], pos, obs, tax)
    assert fam["keys"] == {"truth": "present", "shown": True, "shown_confident": True}
    assert fam["drums"] == {"truth": "absent", "shown": True, "shown_confident": False}
    assert fam["brass"]["shown"] is False
    assert fam["guitar"]["shown"] and not fam["guitar"]["shown_confident"]  # only via subtypes
    ag = {x["label"]: x["agreement"] for x in EE.subtype_agreement(e, pos, obs)}
    assert ag == {"Clean electric guitar": "agree", "Nylon-string guitar": "unlabelled"}
    assert EE.score_entries([{"label": "x", "node": "not.a.node", "confidence": 0.5}], pos, obs,
                            tax)[0]["outcome"] == "unmapped"
