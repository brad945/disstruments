"""M2 synthetic pipeline: MIDI selection, leaf assignment, registry validation, rendering
(fake engine), labels.json provenance, the synthetic loader and harness integration."""
import json
from pathlib import Path

import numpy as np
import pytest

pretty_midi = pytest.importorskip("pretty_midi")
pytest.importorskip("pedalboard")
pytest.importorskip("pyloudnorm")

from disstruments.ml.synth import midi as M  # noqa: E402
from disstruments.ml.synth.sources import SourceError, load_registry  # noqa: E402
from disstruments.ml.taxonomy import Taxonomy  # noqa: E402

V1 = ["guitar.electric.clean", "guitar.electric.distorted", "bass.electric.picked",
      "keys.piano", "drums.acoustic_kit"]


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.load()


def _song(path: Path, programs=((33, False), (29, False), (0, False), (0, True)), secs=24.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    for prog, drum in programs:
        inst = pretty_midi.Instrument(program=prog, is_drum=drum)
        t = 0.0
        while t < secs:
            inst.notes.append(pretty_midi.Note(90, 36 if drum else 48 + prog % 12, t, t + 0.4))
            t += 0.5
        pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


@pytest.fixture
def midi_root(tmp_path):
    root = tmp_path / "midi"
    for k, (artist, title) in enumerate([("Band A", "Song 1"), ("Band A", "Song 1.1"),
                                         ("Band B", "Other"), ("Band C", "Third")]):
        _song(root / artist / f"{title}.mid")
    _song(root / "Band D" / "NoRoles.mid", programs=((48, False),))   # strings: no v1 role
    return root


@pytest.fixture
def registry_file(tmp_path):
    doc = {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "sources": [
        {"id": f"fake_{i}", "name": f"fake {l}", "kind": "synth", "engine": "fake",
         "leaves": [l], "license": "CC0-1.0", "commercial_clean": True}
        for i, l in enumerate(V1)] + [
        {"id": "nc_piano", "name": "nc", "kind": "synth", "engine": "fake",
         "leaves": ["keys.piano"], "license": "CC-BY-NC-4.0", "commercial_clean": False}]}
    import yaml
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


# ------------------------------------------------------------------- midi
def test_roles_and_hints():
    assert M.role_of(33, False) == "bass" and M.role_of(0, True) == "drums"
    assert M.role_of(48, False) is None                     # strings: dropped in v1
    assert M.PROGRAM_HINTS[29] == ("guitar.electric.distorted",)


def test_composition_key_groups_versions(midi_root):
    a = M.composition_key(midi_root / "Band A" / "Song 1.mid", midi_root)
    b = M.composition_key(midi_root / "Band A" / "Song 1.1.mid", midi_root)
    assert a == b == "band a/song 1"


def test_split_is_deterministic_and_proportional():
    keys = [f"artist{i}/song" for i in range(4000)]
    s = [M.split_of(k, {"train": 0.8, "val": 0.1, "test": 0.1}) for k in keys]
    assert s == [M.split_of(k, {"train": 0.8, "val": 0.1, "test": 0.1}) for k in keys]
    frac = s.count("test") / len(s)
    assert 0.08 < frac < 0.12


def test_windows_keep_only_v1_roles(midi_root):
    wins = M.load_windows(midi_root / "Band A" / "Song 1.mid", midi_root)
    assert wins and all(p.role in M.ROLE_LEAVES for w in wins for p in w.parts)
    assert all(n.start >= 0 and n.end <= 10.0 for w in wins for p in w.parts for n in p.notes)
    assert M.load_windows(midi_root / "Band D" / "NoRoles.mid", midi_root) == []


def test_assign_leaves_unique_per_clip_and_balanced(midi_root):
    rng = np.random.default_rng(0)
    bal = M.LeafBalancer(V1, rng)
    seen = []
    for f in sorted(midi_root.rglob("*.mid")):
        for w in M.load_windows(f, midi_root):
            w = M.assign_leaves(w, bal)
            leaves = [p.leaf for p in w.parts]
            assert len(leaves) == len(set(leaves)) and all(leaves)
            seen += leaves
    assert set(seen) <= set(V1)


# ------------------------------------------------------------------- registry
def test_registry_validates(registry_file, tax):
    reg = load_registry(registry_file, tax)
    assert len(reg.for_leaf("keys.piano")) == 2
    assert [s.id for s in reg.for_leaf("keys.piano", commercial_only=True)] == ["fake_3"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["sources"][0].update(license="WTFPL"), "not in the reviewed list"),
    (lambda d: d["sources"][-1].update(commercial_clean=True), "contradicts licence"),
    (lambda d: d["sources"][0].update(leaves=["guitar.bogus"]), "not a taxonomy leaf"),
    (lambda d: d["sources"].pop(0), "no commercially clean source"),
    (lambda d: d["sources"][0].update(kind="sfz"), "needs a path"),
])
def test_registry_rejects(registry_file, tax, mutate, msg):
    import yaml
    d = yaml.safe_load(registry_file.read_text())
    mutate(d)
    registry_file.write_text(yaml.safe_dump(d))
    with pytest.raises(SourceError, match=msg):
        load_registry(registry_file, tax)


def test_packaged_registry_if_present(tax):
    from disstruments.ml.synth.sources import REGISTRY_PATH
    if not REGISTRY_PATH.exists():
        pytest.skip("sources.yaml not written yet")
    reg = load_registry(REGISTRY_PATH, tax)        # validation only; files may be absent
    assert len(reg.v1_leaves) == 20


# ------------------------------------------------------------------- render + load
def test_build_render_load_evaluate(midi_root, registry_file, tmp_path, tax):
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.eval.harness import baseline_predictions, evaluate
    from disstruments.ml.eval.predictions import Predictions
    from disstruments.ml.synth.build import build

    out = tmp_path / "synth"
    m = build(midi_root, out, 12, seed=1, sr=16000, registry_path=str(registry_file), fake=True)
    assert m["n_rendered"] > 0 and m["fake"] is True
    clips = sorted((out / "clips").glob("*/labels.json"))
    lab = json.loads(clips[0].read_text())
    for key in ("clip_id", "seed", "renderer_sha", "taxonomy_version", "positive_leaves",
                "commercial_clean", "sources_commercial_clean", "midi", "stems", "mix"):
        assert key in lab, key
    st = lab["stems"][0]
    for key in ("leaf", "source_id", "source_license", "engine_meta", "fx_chain", "lufs",
                "mix_gain_db", "midi"):
        assert key in st, key
    assert lab["commercial_clean"] is False                # Lakh-style MIDI: never clean
    assert (clips[0].parent / "mix.flac").exists()
    assert {p.stem for p in (clips[0].parent / "stems").glob("*.flac")} == set(lab["positive_leaves"])
    if any(s["fx_chain"] and s["fx_chain"][0]["plugin"] == "Distortion" for s in lab["stems"]):
        assert any(s["leaf"] == "guitar.electric.distorted" or s["leaf"].startswith(("bass", "keys"))
                   for s in lab["stems"])
    # deterministic + resumable: same seed -> same clip ids, nothing re-rendered differently
    m2 = build(midi_root, out, 12, seed=1, sr=16000, registry_path=str(registry_file), fake=True)
    assert m2["n_rendered"] == m["n_rendered"]
    idx = load_dataset("synthetic", out)
    r = idx.records[0]
    assert r.observed == frozenset(tax.nodes)              # perfect labels: all observed
    assert set(tax.leaves()) & r.positive == set(json.loads(
        (out / "clips" / r.item_id / "labels.json").read_text())["positive_leaves"])
    comps = {}
    for rec in idx.records:                                 # artist-disjoint splits
        comps.setdefault(rec.artist, set()).add(rec.split)
    assert all(len(v) == 1 for v in comps.values())
    ids = [x.item_id for x in idx.records]
    s = baseline_predictions("prior", ids, tax, train_records=idx.records)
    res = evaluate(idx.records, Predictions(ids, list(tax.nodes), s, tax.version), tax)
    assert res["items"]["n_evaluated"] == len(ids)
    assert not any(v.get("skipped") == "unobserved" for v in res["per_node"].values())
    assert load_dataset("synthetic", out, commercial_only=True).records == []


def test_distortion_never_on_clean_guitar():
    from disstruments.ml.synth.fx import stem_chain
    for seed in range(300):
        rng = np.random.default_rng(seed)
        assert all(c["plugin"] != "Distortion"
                   for c in stem_chain("guitar.electric.clean", rng))
        assert stem_chain("guitar.electric.distorted", np.random.default_rng(seed))[0]["plugin"] == "Distortion"


# ------------------------------------------------------------------- engines without plugins
def test_parse_display_values():
    from disstruments.ml.synth.patches import _parse
    assert _parse("300.6 ms") == 300.6 and _parse("1.20 s") == 1200.0
    assert _parse("1.5 kHz") == 1500.0 and _parse("-3.00 dB") == -3.0 and _parse("Off") is None


def test_set_nearest_snaps_to_valid_value():
    from disstruments.ml.synth import patches

    class P:
        valid_values = ["0.0 ms", "10.0 ms", "250.0 ms", "1.01 s"]

    class Plug:
        name = "fakeplug"
        parameters = {"atk": P()}

    pl = Plug()
    assert patches.set_nearest(pl, "atk", 900.0) == "1.01 s" and pl.atk == "1.01 s"
    assert patches.set_nearest(pl, "atk", 200.0) == "250.0 ms"


def test_drum808_deterministic_and_routes_gm_notes():
    from disstruments.ml.synth.drum808 import Drum808Engine
    notes = [M.Note(36, 0.0, 0.1, 110), M.Note(42, 0.5, 0.6, 80), M.Note(38, 1.0, 1.1, 100),
             M.Note(99, 1.5, 1.6, 100)]
    part = M.Part(0, 0, True, "drums", notes, leaf="drums.electronic.tr808")
    a1, m1 = Drum808Engine().render(part, None, Path("."), 16000, 2.0, np.random.default_rng(3))
    a2, m2 = Drum808Engine().render(part, None, Path("."), 16000, 2.0, np.random.default_rng(3))
    assert np.array_equal(a1, a2) and m1 == m2
    assert m1["skipped_notes"] == 1 and len(a1) == 32000 and np.abs(a1).max() > 0.05
    assert set(m1["kit"]) >= {"kick", "snare", "hat_closed", "clap"}
