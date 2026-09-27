"""Label resolution semantics shared by all loaders (positive / observed / unknown)."""
import pytest

from disstruments.ml.datasets.base import (LabelMap, artist_disjoint_split, normalize_artist,
                                           observed_exhaustive)
from disstruments.ml.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


@pytest.fixture(scope="module")
def lm(tax):
    return LabelMap("t", {
        "acoustic guitar": "guitar.acoustic",
        "clean": "guitar.electric.clean",
        "timpani": {"node": "percussion", "children_unknown": False},
        "kit": {"node": "drums.acoustic_kit", "unknown": ["cymbals"]},
        "two": ["voice.rap", "voice.spoken"],
    }, {"fx": {"reason": "r", "unknown": ["keys.synth"]},
        "room": {"reason": "r"},
        "anything": {"reason": "r", "unknown": ["*"]}}, tax)


def _obs(tax, lm, labels):
    r = lm.resolve(labels)
    return r, observed_exhaustive(tax, r.positive, r.unknown_roots)


def test_coarse_label_masks_descendants_not_negative(tax, lm):
    r, obs = _obs(tax, lm, ["acoustic guitar"])
    assert r.positive == {"guitar", "guitar.acoustic"}
    assert "guitar.acoustic.nylon" not in obs and "guitar.acoustic.steel" not in obs
    assert "guitar.electric.clean" in obs            # known negative
    assert "keys" in obs                             # exhaustive source: known negative


def test_children_known_negative(tax, lm):
    r, obs = _obs(tax, lm, ["timpani"])
    assert r.positive == {"percussion"}
    assert "percussion.shaker" in obs and "percussion.shaker" not in r.positive


def test_unknown_subtree_makes_ancestors_unknown_unless_positive(tax, lm):
    _, obs = _obs(tax, lm, ["fx"])
    assert "keys.synth" not in obs and "keys.synth.pad" not in obs
    assert "keys" not in obs                          # a synth may be present -> keys may be
    assert "keys.piano" in obs                        # sibling stays a known negative
    r, obs2 = _obs(tax, lm, ["kit"])
    # taxonomy 2.0.0: cymbals is its own family, so a kit no longer masks `percussion`
    # (critic #1: that masking made non-kit percussion a drum-kit detector)
    assert "cymbals" not in obs2 and "percussion" in obs2
    assert "drums" in obs2 and "drums.acoustic_kit" in r.positive


def test_positive_wins_over_unknown(tax, lm):
    r, obs = _obs(tax, lm, ["fx", "acoustic guitar", "clean"])
    assert "guitar.electric.clean" in r.positive and "guitar.electric.clean" in obs
    assert "guitar.electric.distorted" in obs


def test_ignore_without_unknown_changes_nothing(tax, lm):
    r, obs = _obs(tax, lm, ["room"])
    assert r.positive == frozenset() and r.ignored_labels == ("room",)
    assert obs == frozenset(tax.nodes)


def test_star_and_unmapped_labels_mask_everything_but_positives(tax, lm):
    for labels in (["anything", "clean"], ["mystery instrument", "clean"]):
        r, obs = _obs(tax, lm, labels)
        assert obs == r.positive == {"guitar", "guitar.electric", "guitar.electric.clean"}
    r = lm.resolve(["mystery instrument"])
    assert r.unknown_labels == ("mystery instrument",)


def test_multi_node_rule(tax, lm):
    r = lm.resolve(["two"])
    assert r.positive == {"voice", "voice.rap", "voice.spoken"}


def test_normalize_artist():
    assert normalize_artist("Declare A String") == normalize_artist("Declare a String")
    assert normalize_artist("Hops ’n Vinyl") == normalize_artist("Hops 'n Vinyl")
    assert normalize_artist("Music Delta Multitracks", {"music delta multitracks": "Music Delta"}) \
        == normalize_artist("Music Delta")


def test_artist_disjoint_split_deterministic_and_balanced():
    groups = {f"a{i}": (i % 5) + 1 for i in range(40)}           # 120 items
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    s1 = artist_disjoint_split(groups, fr, seed=0)
    assert s1 == artist_disjoint_split(dict(reversed(list(groups.items()))), fr, seed=0)
    counts = {k: sum(groups[g] for g, s in s1.items() if s == k) for k in fr}
    assert counts == {"train": 84, "val": 18, "test": 18}
    assert artist_disjoint_split(groups, fr, seed=1) != s1        # seed changes tie-breaks
    with pytest.raises(ValueError):
        artist_disjoint_split(groups, {"train": -1, "test": 1})
