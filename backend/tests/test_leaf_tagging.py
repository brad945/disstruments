"""M5 leaf-model serving: windowing, feature parity with training, routing rules, ONNX parity."""
import numpy as np
import pytest

from disstruments.pipeline import leaf_tagging as LT
from disstruments.pipeline.tagging import _probs_to_intervals

NAMES = {"bass": "Bass", "bass.electric": "Electric bass", "bass.upright": "Upright bass",
         "guitar.acoustic.nylon": "Nylon-string acoustic guitar",
         "guitar.acoustic.steel": "Steel-string acoustic guitar"}
ROUTING = {"presence": ["bass"],
           "subtypes": {"guitar": ["guitar.acoustic.nylon", "guitar.acoustic.steel"],
                        "bass": ["bass.upright", "bass.electric"]}}


def _merge(panns, nodes, probs):
    return LT.merge(panns, nodes, np.array(probs, float), 10.0, ROUTING, name_of=NAMES.get,
                    panns_node={"guitar": "guitar", "bass": "bass", "voice": "voice"},
                    show=0.3, intervals=_probs_to_intervals)


def _panns(label, conf, stem="mix"):
    return {"label": label, "confidence": conf, "stem": stem, "activations": [[0.0, 10.0]]}


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


def test_owned_presence_replaces_incumbent_and_subtypes_are_capped():
    nodes = ["bass", "bass.electric", "bass.upright", "guitar.acoustic.nylon", "guitar.acoustic.steel"]
    out = _merge([_panns("bass", 0.95, "bass"), _panns("guitar", 0.5), _panns("voice", 0.9)],
                 nodes, [[0.8, 0.7, 0.1, 0.9, 0.2]])
    by = {t["label"]: t for t in out}
    assert "bass" not in by and by["Electric bass"]["model"] == "disstruments-cnn"
    assert by["Electric bass"]["confidence"] == 0.7
    nylon = by["Nylon-string acoustic guitar"]
    assert nylon["confidence"] == 0.5, "a subtype can't be more certain than its family"
    assert nylon["subtype"] and nylon["calibrated"] is False and nylon["stem"] == "mix"
    assert "Steel-string acoustic guitar" not in by                 # below show threshold
    assert by["voice"]["model"] == "panns-sed" and by["voice"]["node"] == "voice"


def test_family_without_confident_subtype_falls_back():
    out = _merge([_panns("guitar", 0.8, "other")], ["guitar.acoustic.nylon", "guitar.acoustic.steel"],
                 [[0.1, 0.2]])
    assert [t["label"] for t in out] == ["guitar (subtype uncertain)"]
    assert out[0]["subtype_uncertain"] and out[0]["stem"] == "other"


def test_owned_presence_below_threshold_drops_the_instrument():
    out = _merge([_panns("bass", 0.9)], ["bass", "bass.electric"], [[0.1, 0.1]])
    assert out == []


def test_song_confidence_is_p95_over_windows():
    col = [0.1] * 18 + [0.5, 0.9]                    # p95 = 0.52: not the max, not the mean
    out = _merge([], ["bass", "bass.electric"], [[c, 0.0] for c in col])
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
