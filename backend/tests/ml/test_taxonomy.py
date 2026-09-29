"""Taxonomy structure, consistency helpers, and mapping coverage vs. real source vocabularies."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from disstruments.ml.datasets import medleydb, openmic, slakh
from disstruments.ml.datasets.base import LabelMap, load_mappings
from disstruments.ml.taxonomy import Taxonomy, TaxonomyError, get_taxonomy

VOCAB = Path(__file__).resolve().parents[1] / "fixtures" / "ml" / "vocab"


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


def _flatten(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _flatten(v)
    else:
        yield from x


def test_structure(tax):
    assert tax.version == "3.0.0" and tax.major == 3
    assert len(tax) == 86 and len(tax.leaves()) == 65   # "~60 leaves"
    assert max(tax.level(n) for n in tax.nodes) == 3
    assert set(tax.roots()) == {"guitar", "bass", "keys", "drums", "cymbals", "percussion",
                                "strings", "voice", "brass", "woodwinds"}
    assert len(tax.sha256) == 64


def test_queries(tax):
    n = "guitar.electric.distorted"
    assert tax.ancestors(n) == ("guitar", "guitar.electric")
    assert tax.ancestors(n, include_self=True)[-1] == n
    assert tax.level(n) == 3 and tax.is_leaf(n) and tax.parent(n) == "guitar.electric"
    assert tax.descendants("guitar.acoustic") == ("guitar.acoustic.nylon", "guitar.acoustic.steel")
    assert "guitar.lap_steel" in tax.leaves() and tax.level("guitar.lap_steel") == 2
    assert tax.close_upward(["keys.electric_piano.rhodes", "voice.rap"]) == {
        "keys", "keys.electric_piano", "keys.electric_piano.rhodes", "voice", "voice.rap"}
    with pytest.raises(KeyError):
        tax.level("guitar.banjo")


def test_max_propagate_and_violations(tax):
    nodes = ["guitar", "guitar.electric", "guitar.electric.clean", "guitar.electric.distorted"]
    s = np.array([[0.2, 0.3, 0.1, 0.9],
                  [0.8, 0.5, 0.4, 0.1]])
    assert tax.violations(s, nodes) == (2, 6)       # row0: electric>guitar, distorted>electric
    out = tax.max_propagate(s, nodes)
    np.testing.assert_allclose(out, [[0.9, 0.9, 0.1, 0.9], [0.8, 0.5, 0.4, 0.1]])
    assert tax.violations(out, nodes) == (0, 6)
    # missing intermediate column: leaf propagates to nearest present ancestor
    out2 = tax.max_propagate(np.array([[0.1, 0.7]]), ["guitar", "guitar.electric.clean"])
    np.testing.assert_allclose(out2, [[0.7, 0.7]])


def test_ancestor_matrix(tax):
    nodes = ["guitar", "guitar.electric", "guitar.electric.clean", "bass"]
    a = tax.ancestor_matrix(nodes)
    assert a[2].tolist() == [True, True, True, False]
    assert a[3].tolist() == [False, False, False, True]


@pytest.mark.parametrize("doc,msg", [
    ({"version": "1.0", "nodes": [{"id": "a", "data": ["medleydb"]}]}, "semver"),
    ({"version": "1.0.0", "nodes": [{"id": "a.b", "data": ["slakh"]}]}, "parent a missing"),
    ({"version": "1.0.0", "nodes": [{"id": "a", "data": ["slakh"]}, {"id": "a.b", "data": ["slakh"]}]},
     "only allowed on leaves"),
    ({"version": "1.0.0", "nodes": [{"id": "a"}]}, "rule b"),
    ({"version": "1.0.0", "nodes": [{"id": "a", "data": ["youtube"]}]}, "unknown data sources"),
    ({"version": "1.0.0", "nodes": [{"id": "a"}, {"id": "a.b"}, {"id": "a.b.c"},
                                    {"id": "a.b.c.d", "data": ["slakh"]}]}, "depth 4"),
])
def test_validation_errors(doc, msg):
    with pytest.raises(TaxonomyError, match=msg):
        Taxonomy.from_dict(doc)


def test_leaf_coverage_matches_mappings(tax):
    """`data:` in taxonomy.yaml must equal what the mapping tables actually reach (slakh,
    openmic). For medleydb, `data:` must be reachable AND have real public examples, which
    test_coverage.py audits against the committed 196-track digest; here: subset."""
    derived = {leaf: set() for leaf in tax.leaves()}
    mdb = medleydb.label_map(tax)
    plugins, classes = slakh.label_maps(tax)
    for source, maps in (("medleydb", [mdb]), ("slakh", [plugins, classes])):
        for lm in maps:
            for node in lm.mapped_nodes():
                if node in derived:
                    derived[node].add(source)
    for node in openmic.class_to_node(tax).values():
        if node in derived:
            derived[node].add("openmic")
    for leaf in tax.leaves():
        info = tax.info[leaf]
        assert set(info.data) - {"medleydb"} == derived[leaf] - {"medleydb"}, leaf
        assert set(info.data) <= derived[leaf], leaf
        assert info.data or info.planned, leaf
        # 3.0.0 amendment (arbiter): `golden` (eval-only) may sit next to `data` when the real
        # data has no test positives (checked exactly in test_coverage.py); `synthetic`
        # still only where no real source exists.
        assert not (info.data and "synthetic" in info.planned), f"{leaf}: synthetic only where no real source exists"
        if info.data and info.planned:
            assert info.planned == ("golden",) and info.data == ("medleydb",), leaf


def test_openmic_map_total_and_antichain(tax):
    real = json.loads((VOCAB / "openmic_class-map.json").read_text())
    c2n = openmic.class_to_node(tax)
    assert set(c2n) == set(real) == set(openmic.OPENMIC_CLASSES)
    assert len(set(c2n.values())) == 20                     # exactly one node each, distinct
    nodes = set(c2n.values())
    for n in nodes:
        assert not (set(tax.ancestors(n)) & nodes), f"{n} has a mapped ancestor"
    # projection is total on mapped subtrees and None above them
    assert openmic.openmic_class_of("keys.piano") == "piano"
    assert openmic.openmic_class_of("keys.electric_piano.rhodes") is None     # 3.0.0: EP != piano
    assert openmic.openmic_class_of("woodwinds.saxophone.tenor") == "saxophone"
    assert openmic.openmic_class_of("bass.synth.sub_808") == "bass"
    assert openmic.openmic_class_of("percussion.mallet.marimba") == "mallet_percussion"
    assert openmic.openmic_class_of("keys") is None
    assert openmic.openmic_class_of("strings.bowed.section") is None
    for cls, node in c2n.items():
        assert openmic.openmic_class_of(node) == cls
    # the four families that ARE OpenMIC classes at level 1
    assert {c for c, n in c2n.items() if tax.level(n) == 1} == {"bass", "cymbals", "drums", "guitar", "voice"}


def test_project_scores():
    nodes = ["keys.piano", "keys.electric_piano.rhodes", "guitar.electric.clean", "keys",
             "woodwinds.saxophone", "woodwinds.saxophone.alto"]
    s = np.array([[0.2, 0.7, 0.4, 0.9, 0.3, 0.6]])
    out = openmic.project_scores(s, nodes)
    col = {c: i for i, c in enumerate(openmic.OPENMIC_CLASSES)}
    assert out[0, col["piano"]] == pytest.approx(0.2)                   # rhodes does not count
    assert out[0, col["saxophone"]] == pytest.approx(0.6)               # max over sax subtree
    assert out[0, col["guitar"]] == pytest.approx(0.4)
    assert np.isnan(out[0, col["organ"]])


def test_medleydb_vocab_fully_covered(tax):
    """Every MedleyDB label (their taxonomy + labels seen in real metadata) is mapped or ignored,
    and no mapping key is stale."""
    vocab = set(_flatten(yaml.safe_load((VOCAB / "medleydb_taxonomy.yaml").read_text())))
    vocab |= set(yaml.safe_load((VOCAB / "medleydb_extra_labels.yaml").read_text()))
    lm = medleydb.label_map(tax)
    assert sorted(vocab - lm.labels()) == []
    assert sorted(lm.labels() - vocab) == []


def test_slakh_vocab_fully_covered(tax):
    vocab = json.loads((VOCAB / "slakh_vocab.json").read_text())
    plugins, classes = slakh.label_maps(tax)
    assert len(vocab["patches"]) == 166
    assert sorted(set(vocab["patches"]) - plugins.labels()) == []
    assert sorted(plugins.labels() - set(vocab["patches"])) == []
    assert "Sound effects" in vocab["inst_classes"] and "Sound Effects" in vocab["inst_classes"]
    assert sorted(set(vocab["inst_classes"]) - classes.labels()) == []


def test_mapping_validation(tax):
    with pytest.raises(TaxonomyError, match="unknown node"):
        LabelMap("x", {"foo": "guitar.banjo"}, {}, tax)
    with pytest.raises(TaxonomyError, match="both mapped and ignored"):
        LabelMap("x", {"foo": "guitar"}, {"foo": {"reason": "r"}}, tax)
    with pytest.raises(TaxonomyError, match="reason"):
        LabelMap("x", {}, {"foo": {}}, tax)
    doc, sha = load_mappings(taxonomy=tax)
    assert doc["taxonomy_version"].split(".")[0] == "3" and len(sha) == 64
