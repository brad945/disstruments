"""M5 leaf-model serving: windowing, feature parity with training, routing rules, ONNX parity."""
import numpy as np
import pytest

from disstruments.pipeline import leaf_tagging as LT
from disstruments.pipeline.tagging import _probs_to_intervals

NAMES = {"bass": "Bass", "bass.electric": "Electric bass", "bass.upright": "Upright bass",
         "guitar.acoustic": "Acoustic guitar", "guitar.electric": "Electric guitar",
         "guitar.acoustic.nylon": "Nylon-string acoustic guitar",
         "guitar.acoustic.steel": "Steel-string acoustic guitar",
         "drums.electronic": "Drum machine", "drums.electronic.tr808": "TR-808"}
ROUTING = {"presence": ["bass"], "descend_margin": 0.2,
           "subtypes": {"guitar": ["guitar.acoustic.nylon", "guitar.acoustic.steel", "guitar.electric"],
                        "bass": ["bass.upright", "bass.electric"],
                        "drums": ["drums.electronic.tr808"]}}


def _merge(panns, scores, routing=ROUTING, trained=None):
    """`scores`: {node: per-window probs or a single song-level prob}."""
    nodes = list(scores)
    probs = np.array([np.atleast_1d(v) for v in scores.values()], float).T
    return LT.merge(panns, nodes, probs, 10.0, routing, name_of=NAMES.get,
                    panns_node={"guitar": "guitar", "bass": "bass", "voice": "voice", "drums": "drums"},
                    show=0.3, intervals=_probs_to_intervals, trained=trained)


def _panns(label, conf, stem="mix"):
    return {"label": label, "confidence": conf, "stem": stem, "activations": [[0.0, 10.0]]}


GUITAR = {"guitar": 0.9, "guitar.acoustic": 0.9, "guitar.electric": 0.1,
          "guitar.acoustic.nylon": 0.9, "guitar.acoustic.steel": 0.2}


def test_windows_match_eval_layout():
    sr = 100
    y = np.arange(2550, dtype=float)                  # 25.5 s at 100 Hz
    ws = LT.windows(y, sr, 10.0)
    assert [len(w) for w in ws] == [1000, 1000, 1000]
    assert ws[1][0] == 1000                          # non-overlapping, from t = 0
    assert ws[2][549] == 2549 and ws[2][550:].sum() == 0   # tail zero-padded like the cache
    assert len(LT.windows(y[:2050], sr, 10.0)) == 2  # 0.5 s tail dropped
    assert len(LT.windows(y[:50], sr, 10.0)) == 1    # but a short song keeps its only window


def test_serving_features_equal_training_features(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    from disstruments.ml.models import scratch as S
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(int(S.SR * S.SECONDS))).astype(np.float32)
    path = tmp_path / "w.wav"
    sf.write(path, y, S.SR)
    np.testing.assert_array_equal(S.logmel(path), S.logmel_array(sf.read(path, dtype="float32")[0]))


def test_owned_presence_replaces_incumbent_and_one_capped_subtype_per_family():
    scores = {"bass": 0.8, "bass.electric": 0.7, "bass.upright": 0.1, **GUITAR}
    # PANNs heard the guitar bleeding into the bass stem: our entry is still mix-level
    out = _merge([_panns("bass", 0.95, "bass"), _panns("guitar", 0.5, "bass"), _panns("voice", 0.9)],
                 scores)
    by = {t["label"]: t for t in out}
    assert set(by) == {"Electric bass", "Nylon-string acoustic guitar", "voice"}
    assert by["Electric bass"]["model"] == "disstruments-cnn" and by["Electric bass"]["confidence"] == 0.7
    nylon = by["Nylon-string acoustic guitar"]
    assert nylon["confidence"] == 0.5, "a subtype can't be more certain than its family"
    assert nylon["subtype"] and nylon["calibrated"] is False
    assert nylon["stem"] == "mix", "window scores come from the mix, not PANNs' stem"
    assert by["voice"]["model"] == "panns-sed" and by["voice"]["node"] == "voice"


def test_no_clear_winner_falls_back_to_family():
    out = _merge([_panns("guitar", 0.8, "other")],
                 {"guitar": 0.6, "guitar.acoustic": 0.6, "guitar.electric": 0.5})
    assert [t["label"] for t in out] == ["guitar (subtype uncertain)"]
    assert out[0]["subtype_uncertain"] and out[0]["stem"] == "other"


def test_descent_stops_at_the_level_it_can_tell_apart():
    out = _merge([_panns("guitar", 0.8)], {**GUITAR, "guitar.acoustic.nylon": 0.6,
                                           "guitar.acoustic.steel": 0.5})
    assert [t["label"] for t in out] == ["Acoustic guitar"]


def test_unlisted_winner_blocks_descent_but_still_competes():
    routing = {**ROUTING, "subtypes": {"guitar": ["guitar.acoustic.steel"]}}
    out = _merge([_panns("guitar", 0.8)], {**GUITAR, "guitar.acoustic.steel": 0.3}, routing)
    assert [t["label"] for t in out] == ["Acoustic guitar"], "never name steel when nylon wins"


def test_single_trained_child_is_never_named():
    out = _merge([_panns("drums", 0.9)], {"drums": 0.9, "drums.acoustic_kit": 0.1,
                                          "drums.electronic": 0.9, "drums.electronic.tr808": 0.9})
    assert [t["label"] for t in out] == ["Drum machine"]


def test_untrained_nodes_do_not_compete():
    scores = {**GUITAR, "guitar.lap_steel": 0.85}
    assert _merge([_panns("guitar", 0.8)], scores)[0]["label"] == "guitar (subtype uncertain)"
    out = _merge([_panns("guitar", 0.8)], scores, trained=set(GUITAR))
    assert out[0]["label"] == "Nylon-string acoustic guitar"


def test_owned_presence_below_threshold_drops_the_instrument():
    assert _merge([_panns("bass", 0.9)], {"bass": 0.1, "bass.electric": 0.1}) == []


def test_song_confidence_is_p95_over_windows():
    col = [0.1] * 18 + [0.5, 0.9]                    # p95 = 0.52: not the max, not the mean
    out = _merge([], {"bass": col, "bass.electric": [0.0] * 20})
    assert [t["label"] for t in out] == ["Bass (subtype uncertain)"]
    assert out[0]["confidence"] == pytest.approx(0.52, abs=1e-3)


def test_onnx_matches_torch():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    torch = pytest.importorskip("torch")
    from disstruments.ml.models import export as EX
    from disstruments.ml.models import scratch as S
    torch.manual_seed(0)
    model = S.make_model(7, S.ScratchConfig(channels=(4, 8, 8, 8)))
    model.eval()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = EX.to_onnx(model, Path(d) / "m.onnx")
        X = np.random.default_rng(0).random((3, S.N_MELS, S.FRAMES)).astype(np.float32)
        assert EX.parity(model, p, X) < 1e-5


def test_served_parents_are_max_propagated_like_the_harness():
    nodes = ["bass", "bass.electric", "bass.electric.fingered", "guitar"]
    out = LT._finalize(np.array([[0.1, 0.2, 0.7, 0.4]]), nodes)
    assert out.tolist() == [[0.7, 0.7, 0.7, 0.4]]


def test_gain_augmentation_equals_recomputing_features_at_that_level():
    torch = pytest.importorskip("torch")
    pytest.importorskip("librosa")
    from disstruments.ml.models import scratch as S
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(S.SR * 2)).astype(np.float32)
    x = torch.from_numpy(S.logmel_array(y).astype(np.float32))[None]
    out = S._gain(x, np.random.default_rng(1), 12.0)
    g_db = np.random.default_rng(1).uniform(-12, 12, size=(1, 1, 1)).item()   # same draw
    ref = S.logmel_array(y * 10 ** (g_db / 20)).astype(np.float32)
    assert np.abs(out[0].numpy() - ref).max() < 0.02          # float16 cache precision
