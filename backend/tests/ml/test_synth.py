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


def test_fit_octaves_prefers_in_range_and_small_shift():
    from disstruments.ml.synth.engines import fit_octaves
    assert fit_octaves([60, 62, 64], 44, 76) == 0
    assert fit_octaves([30, 32, 35], 44, 76) == 24
    assert fit_octaves([90, 95], 26, 46) == -48


def test_stem_fraction_saves_subset_but_labels_complete(midi_root, registry_file, tmp_path):
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.synth.build import build
    out = tmp_path / "s"
    build(midi_root, out, 12, seed=2, sr=16000, registry_path=str(registry_file), fake=True,
          stem_fraction=0.0, windows_per_file=4)
    labs = [json.loads(p.read_text()) for p in (out / "clips").glob("*/labels.json")]
    assert labs and all(not l["stems_saved"] and l["stems"] for l in labs)
    assert not any((out / "clips").glob("*/stems/*.flac"))
    idx = load_dataset("synthetic", out)
    assert all(s.audio is None for r in idx.records for s in r.stems)


def test_fx_v2_identity_safe_and_applies():
    from disstruments.ml.synth.fx import apply_chain, master_chain_v2, stem_chain_v2, synth_room_ir
    for seed in range(200):
        ch = stem_chain_v2("guitar.electric.clean", np.random.default_rng(seed))
        drives = [c["params"]["drive_db"] for c in ch if c["plugin"] == "Distortion"]
        assert all(d <= 4.0 for d in drives)            # clean guitar: at most amp warmth
    ir = synth_room_ir(16000, rt60_s=0.8, predelay_ms=10, n_early=4, brightness=0.5, seed=1)
    assert np.abs(ir).max() == 1.0 and len(ir) > 8000
    rng = np.random.default_rng(5)
    x = (np.random.default_rng(0).standard_normal((1, 16000)) * 0.1).astype(np.float32)
    y = apply_chain(x, 16000, stem_chain_v2("guitar.electric.distorted", rng) + master_chain_v2(rng))
    assert y.shape[0] == 1 and np.isfinite(y).all() and np.abs(y).max() > 0


def test_fit_octaves_keeps_register_when_mostly_in_range():
    from disstruments.ml.synth.engines import fit_octaves
    assert fit_octaves([38, 40, 43, 45, 47], 26, 46) == 0       # 80% fit: no register shift
    assert fit_octaves([50, 52, 55, 57], 26, 46) == -12


def test_leave_source_out_build(midi_root, tmp_path):
    import yaml
    from disstruments.ml.synth.build import build
    doc = {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "sources": [
        {"id": f"fake_{i}", "name": "x", "kind": "synth", "engine": "fake", "leaves": [l],
         "license": "CC0-1.0", "commercial_clean": True} for i, l in enumerate(V1)] + [
        {"id": "held_piano", "name": "h", "kind": "synth", "engine": "fake", "leaves": ["keys.piano"],
         "license": "CC0-1.0", "commercial_clean": True}]}
    reg = tmp_path / "reg.yaml"
    reg.write_text(yaml.safe_dump(doc))
    out = tmp_path / "lso"
    # fake=True routes every source to the fake engine but keeps source ids/holdout logic
    m = build(midi_root, out, 40, seed=3, sr=16000, registry_path=str(reg), fake=True,
              holdout=["held_piano:keys.piano"], windows_per_file=6)
    labs = [json.loads(p.read_text()) for p in (out / "clips").glob("*/labels.json")]
    used = lambda l: {(s["source_id"], s["leaf"]) for s in l["stems"]}  # noqa: E731
    train_like = [l for l in labs if l["split"] != "test_lso"]
    lso = [l for l in labs if l["split"] == "test_lso"]
    assert not any(("held_piano", "keys.piano") in used(l) for l in train_like)
    assert lso and m["holdout"] == ["held_piano:keys.piano"]
    piano_lso = [l for l in lso if "keys.piano" in l["positive_leaves"]]
    assert piano_lso and all(("held_piano", "keys.piano") in used(l) for l in piano_lso)
    assert all(l["source_role"] == "lso" for l in lso)
    n_test = sum(l["split"] == "test" for l in labs)
    assert len(lso) == n_test                                   # every test window twice


def test_label_as_marks_subtree_unknown(midi_root, tmp_path, tax):
    import yaml
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.synth.build import build
    doc = {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "sources": [
        {"id": f"fake_{i}", "name": "x", "kind": "synth", "engine": "fake", "leaves": [l],
         "license": "CC0-1.0", "commercial_clean": True, "weight": 0.001 if l == "bass.electric.picked" else 1.0}
        for i, l in enumerate(V1)] + [
        {"id": "coarse_bass", "name": "c", "kind": "synth", "engine": "fake",
         "leaves": ["bass.electric.picked"], "label_as": "bass.electric",
         "license": "CC0-1.0", "commercial_clean": True}]}
    reg = tmp_path / "reg.yaml"
    reg.write_text(yaml.safe_dump(doc))
    out = tmp_path / "la"
    build(midi_root, out, 20, seed=4, sr=16000, registry_path=str(reg), fake=True, windows_per_file=6)
    idx = load_dataset("synthetic", out)
    hit = [r for r in idx.records if "coarse_bass" in r.source_labels]
    assert hit
    for r in hit:
        assert "bass.electric" in r.positive and "bass.electric.picked" not in r.positive
        assert "bass.electric.picked" not in r.observed and "bass.electric.fingered" not in r.observed
        assert "bass" in r.observed and "guitar" in r.observed
    bad = {"version": 1, "root": "x", "v1_leaves": V1, "sources": doc["sources"][:-1] + [
        dict(doc["sources"][-1], label_as="guitar")]}
    reg.write_text(yaml.safe_dump(bad))
    with pytest.raises(SourceError, match="label_as"):
        load_registry(reg, tax)


# ------------------------------------------------------------------- drive continuum (M4 L2b)
def test_nonlinearity_and_drive_label():
    from disstruments.ml.synth.fx import (DRIVE_CLEAN_MAX, DRIVE_DIST_MIN, drive_label,
                                          drive_stage, nonlinearity)
    t = np.arange(16000) / 16000
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert nonlinearity(x, 3.0 * x) < 1e-6                       # pure gain is not distortion
    assert nonlinearity(x, np.tanh(40 * x)) > DRIVE_DIST_MIN      # hard clipping is
    assert drive_label(0.0) == "guitar.electric.clean"
    assert drive_label(DRIVE_DIST_MIN) == "guitar.electric.distorted"
    assert drive_label((DRIVE_CLEAN_MAX + DRIVE_DIST_MIN) / 2) is None
    seen = set()
    for seed in range(40):                                       # label follows measurement
        y, meta = drive_stage(x, 16000, np.random.default_rng(seed))
        assert meta["label"] == drive_label(meta["nonlinearity"])
        seen.add(meta["label"])
    assert seen == {"guitar.electric.clean", "guitar.electric.distorted", None}


def test_drive_profile_labels_by_measurement(midi_root, tmp_path, tax):
    import yaml
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.synth.build import build
    doc = {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "sources": [
        {"id": f"fake_{i}", "name": "x", "kind": "synth", "engine": "fake", "leaves": [l],
         "license": "CC0-1.0", "commercial_clean": True} for i, l in enumerate(V1)]}
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(doc))
    clean_src = {f"fake_{i}" for i, l in enumerate(V1) if l == "guitar.electric.clean"}
    out = tmp_path / "drv"
    build(midi_root, out, 40, seed=2, sr=16000, registry_path=str(p), fake=True,
          windows_per_file=8, fx_profile="drive")
    labs = [json.loads(f.read_text()) for f in (out / "clips").glob("*/labels.json")]
    driven = [(lab, s) for lab in labs for s in lab["stems"] if s.get("drive")]
    assert driven, "no drive-path stems rendered"
    for lab, s in driven:
        assert s["source_id"] in clean_src                       # drive path uses clean sources
        assert not any(c["plugin"] == "Distortion" for c in s["fx_chain"])
        want = s["drive"]["label"] or "guitar.electric"
        assert s["leaf"] == want and s["coarse"] == (s["drive"]["label"] is None)
        if s["coarse"]:
            assert "guitar.electric" in lab["unknown_below"]
    idx = {r.item_id: r for r in load_dataset("synthetic", out).records}
    for lab, s in driven:
        r = idx[lab["clip_id"]]
        assert "guitar.electric" in r.positive
        if s["coarse"] and not ({"guitar.electric.clean", "guitar.electric.distorted"}
                                & set(lab["positive_leaves"])):
            assert "guitar.electric.clean" not in r.observed
            assert "guitar.electric.distorted" not in r.observed
    # the default profile never takes the drive path
    out1 = tmp_path / "v1"
    build(midi_root, out1, 12, seed=2, sr=16000, registry_path=str(p), fake=True)
    assert not any(s.get("drive") for f in (out1 / "clips").glob("*/labels.json")
                   for s in json.loads(f.read_text())["stems"])


def test_drive2_only_converts_distorted_parts(midi_root, tmp_path):
    import yaml
    from disstruments.ml.synth.build import build
    doc = {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "sources": [
        {"id": f"fake_{i}", "name": "x", "kind": "synth", "engine": "fake", "leaves": [l],
         "license": "CC0-1.0", "commercial_clean": True} for i, l in enumerate(V1)]}
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(doc))
    out = tmp_path / "d2"
    build(midi_root, out, 40, seed=2, sr=16000, registry_path=str(p), fake=True,
          windows_per_file=8, fx_profile="drive2")
    stems = [s for f in (out / "clips").glob("*/labels.json") for s in json.loads(f.read_text())["stems"]]
    driven = [s for s in stems if s.get("drive")]
    assert driven and all(s["part_leaf"] == "guitar.electric.distorted" for s in driven)
    if any(s["part_leaf"] == "guitar.electric.clean" for s in stems):
        assert all(not s.get("drive") for s in stems if s["part_leaf"] == "guitar.electric.clean")


# ------------------------------------------------------------------- note collections (M4 L3c)
def _note_dir(root: Path, inst: str, pitches=range(40, 61, 2), vels=(50, 100), sr=16000):
    import soundfile as sf
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(sr) / sr
    for p in pitches:
        for v in vels:
            f = 440 * 2 ** ((p - 69) / 12)
            sf.write(root / f"{inst}-{p:03d}-{v:03d}.wav", (v / 127 * 0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32), sr)


def test_expand_labels_weights_and_coarse(tmp_path, tax):
    import yaml
    labels = {"labels": {"g1": "guitar.acoustic.nylon", "g2": "guitar.acoustic.nylon",
                         "g3": "guitar.acoustic", "g4": "skip"}}
    (tmp_path / "lab.json").write_text(json.dumps(labels))
    doc = {"version": 1, "root": str(tmp_path), "v1_leaves": [], "sources": [
        {"id": "ns", "name": "n", "kind": "notes", "path": "notes", "license": "CC-BY-4.0",
         "commercial_clean": True, "native_sr": 16000, "weight": 1.0,
         "leaves": ["guitar.acoustic.nylon", "guitar.acoustic.steel"],
         "expand_labels": str(tmp_path / "lab.json")}]}
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(doc))
    reg = load_registry(p, tax)
    by = {s.id: s for s in reg.sources}
    assert set(by) == {"ns:g1", "ns:g2", "ns:g3"}                       # skip dropped
    assert by["ns:g1"].leaves == ("guitar.acoustic.nylon",) and by["ns:g1"].label_as is None
    assert by["ns:g3"].leaves == ("guitar.acoustic.nylon", "guitar.acoustic.steel")
    assert by["ns:g3"].label_as == "guitar.acoustic"                      # unsure -> coarse
    assert abs(sum(s.weight for s in reg.for_leaf("guitar.acoustic.nylon")) - 1.0) < 1e-9
    assert by["ns:g1"].instrument == "g1" and by["ns:g1"].native_sr == 16000
    labels["labels"]["g5"] = "bass.upright"                                # outside the entry
    (tmp_path / "lab.json").write_text(json.dumps(labels))
    with pytest.raises(SourceError, match="covers none"):
        load_registry(p, tax)


def test_notes_engine_pitch_and_range(tmp_path):
    from disstruments.ml.synth.engines import NotesEngine
    from disstruments.ml.synth.midi import Note, Part
    from disstruments.ml.synth.sources import Source
    _note_dir(tmp_path / "notes", "inst")
    src = Source(id="x", name="x", kind="notes", engine="notes", leaves=("keys.piano",),
                 license="CC0-1.0", commercial_clean=True, path="notes", instrument="inst")
    sr = 24000
    part = Part(track_index=0, program=0, is_drum=False, gm_name="x", role="keys", leaf="keys.piano",
                notes=[Note(pitch=45, velocity=100, start=0.1, end=0.6)])   # 45 not recorded: 44/46
    y, meta = NotesEngine().render(part, src, tmp_path, sr, 1.0, np.random.default_rng(0))
    seg = y[int(0.15 * sr):int(0.55 * sr)]
    spec = np.abs(np.fft.rfft(seg))
    f_peak = np.argmax(spec) * sr / len(seg)
    assert abs(f_peak - 440 * 2 ** ((45 - 69) / 12)) < 5                  # resampled to exact pitch
    assert np.abs(y[int(0.75 * sr):]).max() < 1e-6                        # cut after release
    assert meta["skipped_out_of_range"] == 0


def test_bandlimit_applies_only_when_asked(midi_root, registry_file, tmp_path):
    from disstruments.ml.synth.build import build
    out = tmp_path / "bl"
    build(midi_root, out, 8, seed=3, sr=16000, registry_path=str(registry_file), fake=True, bandlimit_p=1.0)
    stems = [s for f in (out / "clips").glob("*/labels.json") for s in json.loads(f.read_text())["stems"]]
    assert stems and all(s["fx_chain"][0]["plugin"] == "LowpassFilter"
                         and 7000 <= s["fx_chain"][0]["params"]["cutoff_frequency_hz"] <= 8000 for s in stems)
    out0 = tmp_path / "bl0"
    build(midi_root, out0, 8, seed=3, sr=16000, registry_path=str(registry_file), fake=True)
    assert json.loads((out0 / "manifest.json").read_text())["bandlimit_p"] == 0.0
