"""v8 pre-registered adoption checks (ml/eval/adopt.py) on hand-built harness arrays."""
import numpy as np
import pytest

from disstruments.ml.eval.adopt import check_new, check_old, scoreable, trained_from_manifest
from disstruments.ml.eval.harness import EvalArrays
from disstruments.ml.taxonomy import Taxonomy

OLD = ["guitar.electric.distorted", "keys.piano"]
NEW = ["strings.bowed.violin", "brass.trombone"]


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.load()


def _arrays(tax, scores_for, n=120, n_artists=30, seed=0):
    """Labels: each node in OLD+NEW positive on ~30% of items, fully observed. `scores_for`
    maps node -> function(y, rng) giving that node's scores; other nodes score 0.5."""
    rng = np.random.default_rng(seed)
    idx = tax.index()
    nodes = list(tax.nodes)
    y = np.zeros((n, len(nodes)), bool)
    for node in OLD + NEW:
        y[:, idx[node]] = rng.random(n) < 0.3
    y = np.array([[any(y[k, idx[d]] for d in (*tax.descendants(nd), nd)) for nd in nodes]
                  for k in range(n)])                          # ancestors positive too
    s = np.full((n, len(nodes)), 0.5)
    for node, fn in scores_for.items():
        s[:, nodes.index(node)] = fn(y[:, idx[node]], rng)
    groups = [f"artist:{k % n_artists}" for k in range(n)]
    return EvalArrays([f"i{k:03d}" for k in range(n)], nodes, y, np.ones_like(y), s, 0.5, 15, groups)


def good(y, rng):
    return np.clip(0.25 + 0.5 * y + 0.15 * rng.standard_normal(len(y)), 0, 1)


def noise(y, rng):
    return rng.random(len(y))


def test_trained_from_manifest_closes_upward(tax):
    t = trained_from_manifest({"leaf_counts": {"brass.trombone": 5, "keys.piano": 0}}, tax)
    assert t == {"brass", "brass.trombone"}


def test_scoreable_needs_positives_negatives_and_artists(tax):
    a = _arrays(tax, {})
    assert scoreable(a, tax, OLD + NEW) == [n for n in tax.nodes if n in OLD + NEW]
    assert scoreable(a, tax, OLD, min_groups=10_000) == []
    assert scoreable(a, tax, ["voice.sung"]) == []                  # no positives at all


def test_old_check_passes_when_unchanged_and_fails_on_forgetting(tax):
    inc = _arrays(tax, {n: good for n in OLD})
    same = _arrays(tax, {n: good for n in OLD})
    r = check_old(same, inc, tax, OLD, n_boot=60)
    assert r["pass"] and abs(r["subset"]["delta"]) < 1e-12
    forgot = _arrays(tax, {OLD[0]: noise, OLD[1]: good})
    r = check_old(forgot, inc, tax, OLD, n_boot=60)
    assert not r["pass"]
    assert any("harness tolerance" in x for x in r["reasons"])
    assert any(x.startswith(OLD[0]) for x in r["reasons"])


def test_new_check_needs_to_beat_incumbent_and_chance(tax):
    inc = _arrays(tax, {n: noise for n in NEW})
    learned = _arrays(tax, {n: good for n in NEW})
    r = check_new(learned, inc, tax, NEW, n_boot=60)
    assert r["pass"]
    assert all(v["lift"] > 0.5 for v in r["per_node"].values())
    not_learned = _arrays(tax, {n: noise for n in NEW}, seed=0)
    r = check_new(not_learned, inc, tax, NEW, n_boot=60)
    assert not r["pass"]
    assert check_new(learned, inc, tax, [], n_boot=10)["pass"] is False
