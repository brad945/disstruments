"""Adversarial tests for taxonomy 3.0.0 (tester pass). v2 comparisons run the CURRENT loader
code against the v2 yamls pinned at commit 168e8d3, in a subprocess (so the lru_cached v2
taxonomy/mappings never leak into this session). That is a faithful "v2-equivalent"
computation because apply_unobserved is a no-op when a dataset declares no `unobserved`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from disstruments.ml.datasets import load_dataset, medleydb, openmic, slakh
from disstruments.ml.datasets.base import (apply_unobserved, load_mappings, observed_exhaustive,
                                           unobserved_roots)
from disstruments.ml.eval import cli
from disstruments.ml.eval.audit import missingness
from disstruments.ml.eval.harness import (baseline_predictions, evaluate, label_items,
                                          promotion_gate)
from disstruments.ml.eval.metrics import (SKIP_NO_POSITIVES, SKIP_UNOBSERVED, hierarchical_prf)
from disstruments.ml.eval.predictions import Predictions
from disstruments.ml.taxonomy import TaxonomyError, get_taxonomy

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
ML = BACKEND / "disstruments" / "ml"
FIX = BACKEND / "tests" / "fixtures" / "ml"
SLAKH = FIX / "slakh" / "slakh2100_flac_redux"
OM = FIX / "openmic" / "openmic-2018"
V2_COMMIT = "168e8d3"
SAX = ("soprano", "alto", "tenor", "baritone")
EP = ("rhodes", "wurlitzer", "fm")


@pytest.fixture(scope="module")
def tax():
    return get_taxonomy()


@pytest.fixture(scope="module")
def pub(mdb_public_root):
    return load_dataset("medleydb", mdb_public_root)


def _mdb(root: Path, stems: dict, tid="A_One", artist="A") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    st = {}
    for sid, v in stems.items():
        if isinstance(v, tuple):
            inst, raws = v
            st[sid] = {"instrument": inst,
                       "raw": {f"R{i:02d}": {"instrument": r} for i, r in enumerate(raws)}}
        else:
            st[sid] = {"instrument": v}
    (root / f"{tid}_METADATA.yaml").write_text(yaml.safe_dump(
        {"artist": artist, "genre": "Rock", "has_bleed": "no", "stems": st}))
    return root


def _mdb1(root, strict=True):
    return load_dataset("medleydb", root, generated_split=True, strict=strict).records[0]


def _slakh(root: Path, stems: dict) -> Path:
    d = root / "train" / "Track00010"
    d.mkdir(parents=True)
    st = {sid: {"audio_rendered": True, "inst_class": cls, "is_drum": cls == "Drums",
                "midi_program_name": "x", "program_num": 0, "plugin_name": plugin,
                "integrated_loudness": lufs}
          for sid, (plugin, cls, lufs) in stems.items()}
    (d / "metadata.yaml").write_text(yaml.safe_dump({"UUID": "u", "lmd_midi_dir": "x", "stems": st}))
    return root


def _no_cymbals(rec):
    assert "cymbals" not in rec.positive and "cymbals" not in rec.observed, rec.item_id
    for s in rec.stems:
        assert "cymbals" not in s.positive and "cymbals" not in s.observed, (rec.item_id, s.stem_id)


# ====================================================================== cymbals
CYMBAL_STEM_CASES = {
    "raw_hihat_only": {"S01": "high hat"},
    "raw_cymbal_only": {"S01": "cymbal"},
    "kit_plus_hihat": {"S01": "drum set", "S02": "high hat"},
    "hihat_list_label": {"S01": ["high hat", "cymbal"]},
    "kit_with_raw_hihat": {"S01": ("drum set", ["high hat", "kick drum"])},
    "hihat_with_raw_cymbal": {"S01": ("high hat", ["cymbal"])},
    "orchestral_cymbal_no_kit": {"S01": "cymbal", "S02": "violin section", "S03": "timpani"},
    "drum_machine": {"S01": "drum machine"},
    "kick_only": {"S01": "kick drum"},
    "unlabeled_plus_hihat": {"S01": "Unlabeled", "S02": "high hat"},
}


@pytest.mark.parametrize("case", sorted(CYMBAL_STEM_CASES))
@pytest.mark.parametrize("strict", [True, False])
def test_medleydb_never_cymbals(tmp_path, case, strict):
    r = _mdb1(_mdb(tmp_path, CYMBAL_STEM_CASES[case]), strict=strict)
    _no_cymbals(r)


def test_medleydb_nonstrict_unmapped_label_with_hihat(tmp_path):
    r = _mdb1(_mdb(tmp_path, {"S01": "zither-of-doom", "S02": "high hat"}), strict=False)
    _no_cymbals(r)
    assert r.unknown_labels == ("zither-of-doom",)


def test_medleydb_hihat_only_stem_keeps_nothing_else_observed_wrongly(tmp_path, tax):
    """A cymbal-only stem: stem has no positives. ARBITER AMENDMENT (critic #3 / Alt A):
    `high hat`/`cymbal` are ignore entries with stem_unknown [drums, percussion] (a close
    mic picks up the kit), so drums/percussion are masked on that stem, not known-negative;
    every other node stays known negative. Track level: only cymbals masked."""
    r = _mdb1(_mdb(tmp_path, {"S01": "high hat"}))
    s = r.stems[0]
    assert s.positive == frozenset()
    masked = {"cymbals"} | set(tax.close_downward(["drums", "percussion"]))
    assert s.observed == frozenset(tax.nodes) - masked
    assert r.observed == frozenset(tax.nodes) - {"cymbals"}


def test_medleydb_kit_stem_with_raw_hihat_keeps_negatives(tmp_path, tax):
    """Raw-track `stem_unknown` applies only when the stem label yields no positive: a
    `drum set` stem with a raw hi-hat track keeps drums.electronic / percussion negative."""
    r = _mdb1(_mdb(tmp_path, {"S01": ("drum set", ["high hat", "kick drum"])}))
    s = r.stems[0]
    assert "drums.acoustic_kit" in s.positive
    assert s.observed == frozenset(tax.nodes) - {"cymbals"}
    r2 = _mdb1(_mdb(tmp_path / "b", {"S01": ("fx/processed sound", ["high hat"])}))
    assert not set(tax.close_downward(["drums", "percussion"])) & r2.stems[0].observed


def test_medleydb_orchestral_cymbal_track_without_kit_real(pub):
    """Report says 2 real tracks carry orchestral 'cymbal' without a kit."""
    hits = [r for r in pub.records if "cymbal" in r.source_labels
            and "drums.acoustic_kit" not in r.positive]
    assert hits
    for r in hits:
        _no_cymbals(r)
        assert "drums" in r.negative                       # kit still known-absent


@pytest.mark.parametrize("lufs", [-75.0, -18.0])
@pytest.mark.parametrize("plugin,cls", [("pop_kit.nkm", "Drums"), ("mystery.nkm", "Drums"),
                                         ("street_knowledge_kit.nkm", "Drums"),
                                         ("mystery.nkm", "Percussive"), ("mystery.nkm", "Nope")])
def test_slakh_never_cymbals(tmp_path, plugin, cls, lufs):
    root = _slakh(tmp_path, {"S00": (plugin, cls, lufs), "S01": ("grand_piano.nkm", "Piano", -20.0)})
    r = load_dataset("slakh", root, strict=False).records[0]
    _no_cymbals(r)
    for mode_idx in (load_dataset("slakh", root, strict=False, min_loudness_lufs=None),):
        _no_cymbals(mode_idx.records[0])


def test_slakh_fixture_all_modes_never_cymbals():
    for mode in ("redux", "split2", "orig"):
        idx = load_dataset("slakh", SLAKH, split_mode=mode, include_omitted=True, strict=False)
        for r in idx.records:
            _no_cymbals(r)
        assert idx.config["unobserved"] == ["cymbals"]


def _openmic_copy(tmp_path: Path, edits: dict[str, dict[str, tuple[float, bool]]]) -> Path:
    """Copy the OpenMIC fixture, overriding (Y_true, Y_mask) per (key, class)."""
    dst = tmp_path / "openmic-2018"
    shutil.copytree(OM, dst)
    cmap = json.loads((dst / "class-map.json").read_text())
    with np.load(dst / "openmic-2018.npz", allow_pickle=False) as z:
        arrs = {k: np.array(z[k]) for k in z.files}
    keys = [str(k) for k in arrs["sample_key"]]
    for key, cls_edits in edits.items():
        i = keys.index(key)
        for cls, (yt, ym) in cls_edits.items():
            arrs["Y_true"][i, cmap[cls]] = yt
            arrs["Y_mask"][i, cmap[cls]] = ym
    np.savez(dst / "openmic-2018.npz", **arrs)
    return dst


K1, K2, K3, K4 = "000046_3840", "000046_7680", "000135_483840", "000135_9600"


def test_openmic_cymbals_follow_ytrue_and_mask(tmp_path):
    root = _openmic_copy(tmp_path, {K1: {"cymbals": (0.9, True)}, K2: {"cymbals": (0.1, True)},
                                    K3: {"cymbals": (0.9, False)}, K4: {"cymbals": (0.5, True)}})
    recs = {r.item_id: r for r in load_dataset("openmic", root).records}
    assert "cymbals" in recs[K1].positive
    assert "cymbals" in recs[K2].negative
    assert "cymbals" not in recs[K3].observed
    assert "cymbals" in recs[K4].positive                  # threshold is >= 0.5


def test_eval_skips_cymbals_on_medleydb_stem_unit_and_slakh(pub, tax):
    recs = pub.split("all")
    items = label_items(recs, "stem")
    ids = [i for i, _, _ in items]
    s = baseline_predictions("random", ids, tax, seed=1, unit="stem")
    res = evaluate(recs, Predictions(ids, list(tax.nodes), s, tax.version), tax, unit="stem")
    assert res["per_node"]["cymbals"]["skipped"] == SKIP_UNOBSERVED
    sl = load_dataset("slakh", SLAKH).records
    ids = [i for i, _, _ in label_items(sl)]
    s = baseline_predictions("random", ids, tax, seed=1)
    res = evaluate(sl, Predictions(ids, list(tax.nodes), s, tax.version), tax)
    assert res["per_node"]["cymbals"]["skipped"] == SKIP_UNOBSERVED


# ====================================================================== v2 vs v3 diff
_DUMP = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    v2dir, mdb_root, slakh_root, out = map(Path, sys.argv[1:5])
    import disstruments.ml.taxonomy as T
    import disstruments.ml.datasets.base as B
    import disstruments.ml.eval.cli as C
    T.TAXONOMY_PATH = v2dir / "taxonomy.yaml"
    B.MAPPINGS_PATH = v2dir / "mappings.yaml"
    C.MAPPINGS_PATH = v2dir / "mappings.yaml"
    T.get_taxonomy.cache_clear()
    from disstruments.ml.datasets import load_dataset
    assert T.get_taxonomy().version.startswith("2."), T.get_taxonomy().version
    def dump(idx):
        return {r.item_id: {"pos": sorted(r.positive), "obs": sorted(r.observed),
                            "stems": {s.stem_id: {"pos": sorted(s.positive), "obs": sorted(s.observed)}
                                      for s in r.stems}} for r in idx.records}
    doc = {"medleydb": dump(load_dataset("medleydb", mdb_root)),
           "slakh": dump(load_dataset("slakh", slakh_root, include_omitted=True, strict=False)),
           "nodes": list(T.get_taxonomy().nodes)}
    out.write_text(json.dumps(doc))
""")


@pytest.fixture(scope="module")
def v2dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("v2yaml")
    for f in ("taxonomy.yaml", "mappings.yaml"):
        txt = subprocess.run(["git", "-C", str(REPO), "show", f"{V2_COMMIT}:backend/disstruments/ml/{f}"],
                             check=True, capture_output=True, text=True).stdout
        (d / f).write_text(txt)
    assert yaml.safe_load((d / "taxonomy.yaml").read_text())["version"].startswith("2.")
    return d


def _run_v2(script: str, *args) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, "-c", script, *map(str, args)], cwd=BACKEND,
                       capture_output=True, text=True, env={**os.environ, "DISS_FAKE_ML": "1"})
    assert p.returncode == 0, p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def v2dump(v2dir, mdb_public_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("v2dump") / "v2.json"
    _run_v2(_DUMP, v2dir, mdb_public_root, SLAKH, out)
    return json.loads(out.read_text())


RENAME_V2_TO_V3 = {"keys.piano.acoustic": "keys.piano",
                   **{f"keys.piano.{e}": f"keys.electric_piano.{e}" for e in EP}}
# v2 nodes whose meaning changed and that have no v3 equivalent to compare against
V2_NOT_COMPARABLE = {"keys.piano", "strings.plucked.sitar", "voice.beatbox"}


def _units(dump_ds):
    for tid, r in dump_ds.items():
        yield tid, r
        for sid, s in r["stems"].items():
            yield f"{tid}/{sid}", s


def _v3_units(idx):
    out = {}
    for r in idx.records:
        out[r.item_id] = {"pos": set(r.positive), "obs": set(r.observed)}
        for s in r.stems:
            out[f"{r.item_id}/{s.stem_id}"] = {"pos": set(s.positive), "obs": set(s.observed)}
    return out


def _diff_nodes(v2ds, v3idx, tax, v2nodes):
    v3u = _v3_units(v3idx)
    changed: dict[str, int] = {}
    v2u = dict(_units(v2ds))
    assert set(v2u) == set(v3u)
    for n2 in v2nodes:
        if n2 in V2_NOT_COMPARABLE:
            continue
        n3 = RENAME_V2_TO_V3.get(n2, n2)
        assert n3 in tax, n2
        for uid, u2 in v2u.items():
            u3 = v3u[uid]
            if (n2 in u2["pos"]) != (n3 in u3["pos"]) or (n2 in u2["obs"]) != (n3 in u3["obs"]):
                changed[n3] = changed.get(n3, 0) + 1
    return changed


def test_v2_v3_medleydb_only_cymbals_change(v2dump, pub, tax):
    """ARBITER AMENDMENT (critic #3 / Alt A): tracks: only cymbals changes. Stems: besides
    cymbals, exactly the 9 real stems whose own label is `cymbal` (8) or asserts nothing
    and carries a raw hi-hat/cymbal (1 fx stem) lose drums/percussion observations
    (stem_unknown), and nothing else changes on them (no positive changes at all)."""
    v2_tracks_only = {k: {**v, "stems": {}} for k, v in v2dump["medleydb"].items()}
    v3_tracks_only = SimpleNamespace(records=[SimpleNamespace(item_id=r.item_id, positive=r.positive,
                                                              observed=r.observed, stems=())
                                              for r in pub.records])
    changed = _diff_nodes(v2_tracks_only, v3_tracks_only, tax, v2dump["nodes"])
    assert set(changed) == {"cymbals"}, changed
    v3u = _v3_units(pub)
    by_stem = {r.item_id + "/" + s.stem_id: s for r in pub.records for s in r.stems}
    drumperc = set(tax.close_downward(["drums", "percussion"]))
    hit = set()
    for uid, u2 in _units(v2dump["medleydb"]):
        if "/" not in uid:
            continue
        u3 = v3u[uid]
        for n2 in v2dump["nodes"]:
            if n2 in V2_NOT_COMPARABLE or n2 == "cymbals":
                continue
            n3 = RENAME_V2_TO_V3.get(n2, n2)
            assert (n2 in u2["pos"]) == (n3 in u3["pos"]), (uid, n3)       # positives never change
            if (n2 in u2["obs"]) != (n3 in u3["obs"]):
                assert n2 in u2["obs"] and n3 in drumperc, (uid, n3)       # only masked, only drums/perc
                assert {"high hat", "cymbal"} & set(by_stem[uid].source_labels), uid
                hit.add(uid)
    assert len(hit) == 9, sorted(hit)


def test_v2_v3_slakh_only_cymbals_change(v2dump, tax):
    idx = load_dataset("slakh", SLAKH, include_omitted=True, strict=False)
    changed = _diff_nodes(v2dump["slakh"], idx, tax, v2dump["nodes"])
    assert set(changed) == {"cymbals"}, changed


def test_v2_v3_new_nodes_consistent(v2dump, pub, tax):
    """keys.electric_piano positive <=> 'electric piano' label; v3 saxophone subtree
    positive only where v2 woodwinds.saxophone was positive; sax subtype == label."""
    v2 = v2dump["medleydb"]
    for r in pub.records:
        assert ("keys.electric_piano" in r.positive) == ("electric piano" in r.source_labels), r.item_id
        # v2 keys.piano (internal, incl. EPs) positive <=> v3 piano or e-piano positive
        assert ("keys.piano" in v2[r.item_id]["pos"]) == bool(
            {"keys.piano", "keys.electric_piano"} & r.positive), r.item_id
        for sub in SAX:
            leaf = f"woodwinds.saxophone.{sub}"
            if leaf in r.positive:
                assert f"{sub} saxophone" in r.source_labels, (r.item_id, leaf)
                assert "woodwinds.saxophone" in v2[r.item_id]["pos"]
            elif f"{sub} saxophone" in r.source_labels:
                pytest.fail(f"{r.item_id}: '{sub} saxophone' label but {leaf} not positive")


def test_real_sax_track_subtype_negatives_are_honest(pub):
    """A track with only a tenor sax stem: other subtypes known negative (MedleyDB is
    exhaustive), unless a horn section masks the sax subtree."""
    for r in pub.records:
        sub = {s for s in SAX if f"woodwinds.saxophone.{s}" in r.positive}
        if sub and "horn section" not in r.source_labels:
            for s in set(SAX) - sub:
                assert f"woodwinds.saxophone.{s}" in r.negative, (r.item_id, s)


# ====================================================================== unobserved mechanism
def test_unobserved_validation_bad_entries(tax):
    with pytest.raises(TaxonomyError):
        unobserved_roots({"x": {"unobserved": ["cymbals"]}}, "x", tax)            # non-dict
    with pytest.raises(TaxonomyError):
        unobserved_roots({"x": {"unobserved": "cymbals"}}, "x", tax)              # str
    with pytest.raises(TaxonomyError, match="unknown node"):
        unobserved_roots({"x": {"unobserved": {"*": "everything"}}}, "x", tax)    # ALL sentinel
    with pytest.raises(TaxonomyError, match="needs a reason"):
        unobserved_roots({"x": {"unobserved": {"cymbals": None}}}, "x", tax)
    with pytest.raises(TaxonomyError, match="needs a reason"):
        unobserved_roots({"x": {"unobserved": {"cymbals": "   "}}}, "x", tax)
    assert unobserved_roots({"x": {}}, "x", tax) == frozenset()
    assert unobserved_roots({}, "x", tax) == frozenset()


def test_unobserved_root_that_is_strict_ancestor_of_mapped_nodes_is_rejected(tax):
    """`unobserved: {keys: ...}` with `piano: keys.piano` in the same map would silently strip
    every piano positive the source asserts. Validation should refuse a root that is a
    strict ancestor of a node the dataset's own map targets."""
    doc = {"medleydb": {"unobserved": {"keys": "r"}, "map": {"piano": "keys.piano"}}}
    with pytest.raises(TaxonomyError):
        unobserved_roots(doc, "medleydb", tax)


def test_unobserved_idempotent_and_ancestors_kept(tax):
    """ARBITER AMENDMENT (Alt A): `unobserved` is a pure mask — positives are never stripped
    (validation keeps them out of the subtree; if one got there anyway, positives win)."""
    pos = tax.close_upward(["woodwinds.saxophone.alto", "keys.piano", "cymbals"])
    un = frozenset({"cymbals", "brass"})
    p1, o1 = apply_unobserved(tax, pos, [], un)
    p2, o2 = apply_unobserved(tax, p1, [], un)
    assert (p1, o1) == (p2, o2)
    assert p1 == pos and pos <= o1                                   # positives always win
    assert not set(tax.close_downward(["brass"])) & o1              # non-positive subtree masked
    assert o1 == observed_exhaustive(tax, pos, un)
    # unobserved + ordinary unknown roots compose; ALL still masks all non-positives
    p3, o3 = apply_unobserved(tax, pos, ["*"], un)
    assert o3 == p3 == pos


def test_unobserved_mask_rejects_rules_inside_subtree(tax):
    """Any map/ignore/unknown/stem_unknown target inside an unobserved subtree is an error,
    in every section (incl. openmic `classes` and slakh plugin/inst_class tables)."""
    cases = [
        {"medleydb": {"unobserved": {"cymbals": "r"}, "map": {"hh": "cymbals"}}},
        {"medleydb": {"unobserved": {"cymbals": "r"}, "ignore": {"hh": {"reason": "x", "unknown": ["cymbals"]}}}},
        {"medleydb": {"unobserved": {"keys": "r"}, "ignore": {"hh": {"reason": "x", "stem_unknown": ["keys.organ"]}}}},
        {"medleydb": {"unobserved": {"keys": "r"}, "map": {"x": {"node": "bass", "unknown": ["keys.synth"]}}}},
        {"slakh": {"unobserved": {"keys": "r"}, "plugins": {"p.nkm": "keys.piano"}}},
        {"slakh": {"unobserved": {"keys": "r"}, "inst_class": {"Piano": ["keys", "bass"]}}},
        {"openmic": {"unobserved": {"cymbals": "r"}, "classes": {"cymbals": "cymbals"}}},
    ]
    for doc in cases:
        (name,) = doc
        with pytest.raises(TaxonomyError, match="inside"):
            unobserved_roots(doc, name, tax)
    ok = {"medleydb": {"unobserved": {"cymbals": "r"},
                       "ignore": {"hh": {"reason": "x", "unknown": ["*"], "stem_unknown": ["drums"]}}}}
    assert unobserved_roots(ok, "medleydb", tax) == {"cymbals"}


def test_load_mappings_validates_every_section(tmp_path, tax):
    """load_mappings runs the validation on every dataset section (openmic included), so a
    bad `openmic.unobserved` can never be silently ignored."""
    from disstruments.ml.datasets.base import MAPPINGS_PATH
    doc = yaml.safe_load(MAPPINGS_PATH.read_text())
    doc["openmic"]["unobserved"] = {"cymbals": "r"}
    bad = tmp_path / "m.yaml"
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(TaxonomyError, match="openmic.classes"):
        load_mappings(bad, taxonomy=tax)


def test_load_mappings_returns_private_copy(tax):
    d1, _ = load_mappings(taxonomy=tax)
    d1["medleydb"]["map"]["violin"] = "brass.tuba"
    d1["openmic"]["classes"].clear()
    d2, _ = load_mappings(taxonomy=tax)
    assert d2["medleydb"]["map"]["violin"] == "strings.bowed.violin"
    assert len(d2["openmic"]["classes"]) == 20


def test_unobserved_empty_is_noop(tax):
    pos = tax.close_upward(["guitar.electric.clean"])
    p, o = apply_unobserved(tax, pos, ["bass.synth"], frozenset())
    assert p == pos and o == observed_exhaustive(tax, pos, ["bass.synth"])


def test_deep_unobserved_root_masks_ancestor_unless_positive(tax):
    """Documents current (honest) behaviour: for a non-root unobserved node (guitar.electric)
    the ancestor `guitar` is unknown unless positive -> observed ONLY when positive, i.e.
    label-dependent missingness, contradicting the mappings.yaml:21 claim. The missingness
    audit cannot flag it (flagged requires n_obs > n_pos). Guarded only if the validator
    restricts roots to level 1 — now enforced by `unobserved_roots` (test below), so this
    documents why the level-1 rule exists."""
    un = frozenset({"guitar.electric"})
    recs = []
    for i in range(40):
        lab = ["guitar.electric.clean"] if i < 10 else (["guitar.acoustic.steel"] if i < 20 else ["bass.electric"])
        p, o = apply_unobserved(tax, tax.close_upward(lab), [], un)
        recs.append(SimpleNamespace(positive=p, observed=o))
    n_obs = sum("guitar" in r.observed for r in recs)
    n_pos = sum("guitar" in r.positive for r in recs)
    assert n_obs == n_pos == 20
    assert not missingness(recs, tax)["guitar"]["flagged"]          # audit blind to it


def test_unobserved_root_with_parent_is_rejected(tax):
    """A non-level-1 unobserved root creates label-dependent missingness at its ancestors
    (previous test); the validator should refuse it (or the doc claim must be dropped)."""
    with pytest.raises(TaxonomyError):
        unobserved_roots({"x": {"unobserved": {"guitar.electric": "r"}}}, "x", tax)


def test_hier_f1_via_evaluate_openmic_coarse_sax(tmp_path, tax):
    """End-to-end (propagate + threshold + m_full): OpenMIC clip saxophone=1, all other
    classes masked; predicting alto only must score hier P = R = 1."""
    cm = json.loads((OM / "class-map.json").read_text())
    edits = {K1: {c: (0.0, False) for c in cm} | {"saxophone": (1.0, True)}}
    idx = load_dataset("openmic", _openmic_copy(tmp_path, edits))
    rec = [r for r in idx.records if r.item_id == K1]
    nodes = list(tax.nodes)
    s = np.full((1, len(nodes)), 0.05)
    s[0, nodes.index("woodwinds.saxophone.alto")] = 0.9
    res = evaluate(rec, Predictions([K1], nodes, s, tax.version), tax)
    h = res["summary"]["hier_f1"]
    assert h == pytest.approx(1.0), res.get("hierarchical")


def test_missingness_audit_clean_on_real_medleydb(pub, tax):
    flagged = [n for n, m in missingness(pub.records, tax).items() if m["flagged"]]
    assert flagged == []


# ====================================================================== sax / e-piano
def test_every_medleydb_sax_label_maps_to_leaf(tax):
    lmap = medleydb.label_map(tax)
    sax_labels = sorted(l for l in lmap.labels() if "sax" in l)
    assert sax_labels == sorted(f"{s} saxophone" for s in SAX)
    for l in sax_labels:
        res = lmap.resolve([l])
        assert res.mapped == {f"woodwinds.saxophone.{l.split()[0]}"} and res.unknown_roots == frozenset()


@pytest.mark.parametrize("stem,raw,pos_has,unk_has", [
    ("woodwind section", ["alto saxophone"], "woodwinds.saxophone.alto", None),     # refines
    ("tenor saxophone", ["alto saxophone"], "woodwinds.saxophone.tenor", "woodwinds.saxophone.alto"),
    ("horn section", ["tenor saxophone"], "brass.section", "woodwinds.saxophone.tenor"),
])
def test_medleydb_sax_raw_refinement(tmp_path, stem, raw, pos_has, unk_has):
    r = _mdb1(_mdb(tmp_path, {"S01": (stem, raw)}))
    s = r.stems[0]
    assert pos_has in s.positive
    if unk_has:
        assert unk_has not in s.observed and unk_has not in s.positive


def test_openmic_saxophone_and_piano_semantics(tmp_path, tax):
    root = _openmic_copy(tmp_path, {
        K1: {"saxophone": (0.9, True), "piano": (0.9, True)},
        K2: {"saxophone": (0.1, True), "piano": (0.1, True)},
        K3: {"saxophone": (0.9, False), "piano": (0.9, False)}})
    recs = {r.item_id: r for r in load_dataset("openmic", root).records}
    subs = {f"woodwinds.saxophone.{s}" for s in SAX}
    eps = set(tax.descendants("keys.electric_piano", include_self=True))
    a, b, c = recs[K1], recs[K2], recs[K3]
    assert "woodwinds.saxophone" in a.positive and not subs & a.observed
    assert subs | {"woodwinds.saxophone"} <= b.negative
    assert not (subs | {"woodwinds.saxophone"}) & c.observed
    for r in (a, b, c):                          # OpenMIC piano says nothing about e-pianos
        assert not eps & r.observed, r.item_id
    assert "keys.piano" in a.positive and "keys.piano" in b.negative


def test_openmic_projection_electric_piano_not_piano(tax):
    nodes = list(tax.nodes)
    s = np.zeros((1, len(nodes)))
    s[0, nodes.index("keys.electric_piano.rhodes")] = 0.95
    s[0, nodes.index("woodwinds.saxophone.alto")] = 0.8
    proj = openmic.project_scores(tax.max_propagate(s), nodes)
    classes = list(json.loads((OM / "class-map.json").read_text()))
    assert proj[0, classes.index("piano")] == 0.0
    assert proj[0, classes.index("saxophone")] == pytest.approx(0.8)


def test_openmic_class_map_is_antichain(tax):
    c2n = openmic.class_to_node(tax)
    nodes = set(c2n.values())
    for n in nodes:
        assert not (set(tax.ancestors(n)) & nodes), n


def test_slakh_patch_table_vs_report(tax):
    plugins, classes = slakh.label_maps(tax)
    expect = {"scarbee_a_200.nkm": "keys.electric_piano.wurlitzer",
              "wurly_ep.nkm": "keys.electric_piano.wurlitzer",
              "scarbee_mark_I.nkm": "keys.electric_piano.rhodes",
              "scarbee_pianet.nkm": "keys.electric_piano",
              "scarbee_clavinet_full.nkm": "keys.clavinet"}
    for p, n in expect.items():
        assert plugins.resolve([p]).mapped == {n}, p
    ac = [p for p in plugins.labels() if plugins.resolve([p]).mapped == {"keys.piano"}]
    assert len(ac) == 11
    assert not any("sax" in p for p in plugins.labels()
                   if any(n.startswith("keys") for n in plugins.resolve([p]).mapped))


def test_slakh_unknown_sax_patch_reed_fallback(tmp_path, tax):
    root = _slakh(tmp_path, {"S00": ("soprano_sax_mystery.nkm", "Reed", -20.0)})
    r = load_dataset("slakh", root, strict=False).records[0]
    assert "woodwinds" in r.positive
    assert not set(tax.descendants("woodwinds")) & r.observed


def test_hier_f1_partial_credit_alto_vs_coarse_sax(tax):
    nodes = list(tax.nodes)
    ix = {n: i for i, n in enumerate(nodes)}
    anc = tax.ancestor_matrix()
    subs = [f"woodwinds.saxophone.{s}" for s in SAX]
    # truth: OpenMIC-style saxophone=1, subtypes masked; everything else observed negative
    y = np.zeros((1, len(nodes)), bool)
    for n in tax.close_upward(["woodwinds.saxophone"]):
        y[0, ix[n]] = True
    m = np.ones_like(y)
    for n in subs:
        m[0, ix[n]] = False
    pred = np.zeros_like(y)
    pred[0, ix["woodwinds.saxophone.alto"]] = True
    r = hierarchical_prf(y, pred, m, anc)
    assert r["precision"] == 1.0 and r["recall"] == 1.0, r
    # alto predicted, truth tenor (MedleyDB, subtypes observed): partial credit
    y2 = np.zeros_like(y)
    for n in tax.close_upward(["woodwinds.saxophone.tenor"]):
        y2[0, ix[n]] = True
    r2 = hierarchical_prf(y2, pred, np.ones_like(y), anc)
    assert r2["intersection"] == 2 and r2["precision"] == pytest.approx(2 / 3)
    assert r2["recall"] == pytest.approx(2 / 3)


def test_rare_sax_leaves_skip_cleanly(pub, tax):
    test = pub.split("test")
    ids = [i for i, _, _ in label_items(test)]
    s = baseline_predictions("prior", ids, tax, train_records=pub.split("train"))
    res = evaluate(test, Predictions(ids, list(tax.nodes), s, tax.version), tax)
    for sub in ("soprano", "alto", "baritone"):
        leaf = f"woodwinds.saxophone.{sub}"
        assert res["per_node"][leaf]["skipped"] in (SKIP_NO_POSITIVES, SKIP_UNOBSERVED), leaf


# ====================================================================== collapse + grep
REMOVED = ("keys.piano.acoustic", "keys.piano.rhodes", "keys.piano.wurlitzer", "keys.piano.fm",
           "strings.plucked.sitar", "voice.beatbox")


def test_removed_ids_absent_from_taxonomy(tax):
    for n in REMOVED:
        assert n not in tax


def test_no_removed_ids_in_ml_code_or_data():
    hits = []
    for p in ML.rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix not in {".py", ".yaml", ".json"}:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if any(n in line for n in REMOVED) and not line.lstrip().startswith("#"):
                hits.append(f"{p.relative_to(ML)}:{i}: {line.strip()}")
    assert hits == [], hits


def test_split_file_still_artist_disjoint_under_v3(pub):
    from disstruments.ml.datasets.base import artist_leakage
    assert artist_leakage(pub.records) == {}


@pytest.mark.parametrize("label,parent", [("sitar", "strings.plucked"), ("beatboxing", "voice")])
def test_collapse_stem_level_and_with_sibling(tmp_path, tax, label, parent):
    sib = "banjo" if parent == "strings.plucked" else "male singer"
    r = _mdb1(_mdb(tmp_path, {"S01": label, "S02": sib}))
    st = {s.stem_id: s for s in r.stems}
    s1 = st["S01"]
    assert parent in s1.positive and set(tax.children(parent)) <= s1.observed - s1.positive
    assert set(tax.children(parent)) & r.positive               # sibling leaf still positive


# ====================================================================== gate
_V2_RUN = textwrap.dedent("""
    import sys
    from pathlib import Path
    v2dir, mdb_root, runs = map(Path, sys.argv[1:4])
    import disstruments.ml.taxonomy as T
    import disstruments.ml.datasets.base as B
    import disstruments.ml.eval.cli as C
    T.TAXONOMY_PATH = v2dir / "taxonomy.yaml"
    B.MAPPINGS_PATH = v2dir / "mappings.yaml"
    C.MAPPINGS_PATH = v2dir / "mappings.yaml"
    T.get_taxonomy.cache_clear()
    preds = runs / "prior_v2.json"
    assert C.main(["baseline", "--kind", "prior", "--dataset", "medleydb", "--root", str(mdb_root),
                   "--split", "val", "--out", str(preds)]) == 0
    assert C.main(["run", "--dataset", "medleydb", "--root", str(mdb_root), "--split", "val",
                   "--predictions", str(preds), "--runs-dir", str(runs), "--name", "v2"]) == 0
""")


@pytest.fixture(scope="module")
def gate_runs(v2dir, mdb_public_root, tmp_path_factory):
    runs = tmp_path_factory.mktemp("gate")
    _run_v2(_V2_RUN, v2dir, mdb_public_root, runs)
    preds = runs / "prior_v3.json"
    assert cli.main(["baseline", "--kind", "prior", "--dataset", "medleydb", "--root", str(mdb_public_root),
                     "--split", "val", "--out", str(preds)]) == 0
    assert cli.main(["run", "--dataset", "medleydb", "--root", str(mdb_public_root), "--split", "val",
                     "--predictions", str(preds), "--runs-dir", str(runs), "--name", "v3"]) == 0
    get = lambda n: json.loads(next(runs.glob(f"*_{n}/metrics.json")).read_text())  # noqa: E731
    return get("v2"), get("v3"), runs


def test_gate_refuses_v2_vs_v3(gate_runs):
    v2, v3, _ = gate_runs
    assert v2["provenance"]["taxonomy_version"].startswith("2.")
    assert v3["provenance"]["taxonomy_version"] == "3.0.0"
    for a, b in ((v3, v2), (v2, v3)):
        g = promotion_gate(a, b)
        assert g["comparable"] is False and g["promote"] is False
        assert any("taxonomy_version" in x for x in g["reasons"]), g["reasons"]


def test_cli_compare_v2_vs_v3_exit_nonzero(gate_runs, capsys):
    _, _, runs = gate_runs
    c = next(runs.glob("*_v3"))
    i = next(runs.glob("*_v2"))
    assert cli.main(["compare", str(c), str(i)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["comparable"] is False


def test_v2_predictions_rejected_by_v3_harness(gate_runs, mdb_public_root):
    _, _, runs = gate_runs
    with pytest.raises((ValueError, SystemExit), match="taxonomy|nodes not in"):
        cli.main(["run", "--dataset", "medleydb", "--root", str(mdb_public_root), "--split", "val",
                  "--predictions", str(runs / "prior_v2.json"), "--runs-dir", str(runs / "x")])


def test_session_taxonomy_not_leaked(tax):
    assert get_taxonomy().version == "3.0.0"
    doc, _ = load_mappings()
    assert doc["taxonomy_version"] == "3.0.0"


# ====================================================================== arbiter additions
@pytest.mark.parametrize("dataset", ["medleydb", "slakh"])
def test_unobserved_mask_invariance_real_data(dataset, mdb_public_root, monkeypatch, tax):
    """Property (critic #9): loading with the dataset's `unobserved` vs with it forced empty
    differs ONLY in observation of nodes inside the unobserved subtrees — never in any
    positive, never outside the subtree — on every record and stem of the real 196-track
    MedleyDB digest and the Slakh fixture (all split modes)."""
    mod = {"medleydb": medleydb, "slakh": slakh}[dataset]
    doc, _ = load_mappings(taxonomy=tax)
    masked = set(tax.close_downward(unobserved_roots(doc, dataset, tax)))
    assert masked == {"cymbals"} | set(tax.descendants("cymbals"))

    def load():
        if dataset == "medleydb":
            return [load_dataset("medleydb", mdb_public_root)]
        return [load_dataset("slakh", SLAKH, split_mode=m, include_omitted=True, strict=False)
                for m in ("redux", "split2", "orig")]

    on = load()
    monkeypatch.setattr(mod, "unobserved_roots", lambda *a, **k: frozenset())
    off = load()
    n_units = 0
    for a, b in zip(on, off):
        assert [r.item_id for r in a.records] == [r.item_id for r in b.records]
        for ra, rb in zip(a.records, b.records):
            units = [(ra, rb)] + list(zip(ra.stems, rb.stems))
            for ua, ub in units:
                n_units += 1
                assert ua.positive == ub.positive
                assert not ua.positive & masked
                assert (ua.observed ^ ub.observed) <= masked
                assert not ua.observed & masked
    assert n_units > (1000 if dataset == "medleydb" else 10)


def test_n_nonrefining_raw_labels_exact(pub):
    """Critic #4: raw hi-hat/cymbal tracks under a kit/drum-machine/aux-percussion/fx stem
    are ignore entries now, so they no longer count as contradictions: 64 -> 40 (exactly
    the 24 such (stem, raw label) pairs in the real digest)."""
    assert pub.stats["n_nonrefining_raw_labels"] == 40


def test_split_pin_hash_unchanged_and_gaps_disclosed(pub):
    """Critic #6: the pin gained disclosure metadata only; split_hash (item, split) is unchanged."""
    from disstruments.ml.datasets.base import split_hash
    doc = json.loads(medleydb.PINNED_SPLIT_PATH.read_text())
    assert doc["split_hash"] == "fba910c7f963460a" == split_hash(pub.records)
    assert doc["stratified_on_taxonomy"] == "2.0.0" and "keys.electric_piano" in doc["known_gaps"]
    count = lambda n, sp: sum(n in r.positive for r in pub.split(sp))  # noqa: E731
    assert [count("keys.electric_piano", s) for s in ("train", "val", "test")] == [9, 1, 0]
    assert count("woodwinds.saxophone.tenor", "test") == 2
    for sub in ("soprano", "alto", "baritone"):
        assert count(f"woodwinds.saxophone.{sub}", "test") == 0


def test_golden_planned_exactly_where_no_test_source(pub, tax):
    """Leaves whose only real source is MedleyDB and that have 0 positives in the pinned
    test split carry `planned: [golden]` (and vice versa)."""
    no_test = {l for l in tax.leaves() if tax.info[l].data == ("medleydb",)
               and not any(l in r.positive for r in pub.split("test"))}
    marked = {l for l in tax.leaves() if tax.info[l].data and "golden" in tax.info[l].planned}
    assert no_test == marked == {"woodwinds.saxophone.soprano", "woodwinds.harmonica", "voice.spoken"}


def test_openmic_piano_incl_ep_diagnostic_not_gated(tax):
    """Non-gated diagnostic: AP of max(keys.piano, keys.electric_piano) vs OpenMIC piano.
    Lives in the openmic20 block only (not summary -> not a secondary / bootstrap metric)."""
    from disstruments.ml.eval.harness import DEFAULT_SECONDARIES, comparability_key
    idx = load_dataset("openmic", OM)
    recs = idx.records
    ids = [i for i, _, _ in label_items(recs)]
    nodes = list(tax.nodes)
    s = np.full((len(ids), len(nodes)), 0.1)
    piano = [("keys.piano" in r.positive) for r in recs]
    assert any(piano) and not all(p for r, p in zip(recs, piano) if "keys.piano" in r.observed)
    for k, p in enumerate(piano):                  # perfect on piano only via the e-piano head
        s[k, nodes.index("keys.electric_piano")] = 0.9 if p else 0.05
    res = evaluate(recs, Predictions(ids, nodes, s, tax.version), tax)
    d = res["openmic20"]["piano_incl_ep"]
    assert d["ap"] == pytest.approx(1.0) and d["skipped"] is None
    assert res["openmic20"]["per_class_ap"]["piano"] < 1.0      # acoustic-only view penalizes it
    assert not any("piano_incl" in k for k in res["summary"]) and not any(
        "piano_incl" in k for k in DEFAULT_SECONDARIES)
    assert "openmic20" not in comparability_key(res)
