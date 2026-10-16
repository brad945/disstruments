"""Fix 2 after golden-1: AudioSet classes mapped to new labels (docs/debriefs/fixes-after-golden1.md).

New labels have no real calibration data, so they must reach the report flagged
`calibrated: false` (always "?"), carry their taxonomy node, and never alter the input of an
already calibrated label."""
import json

import numpy as np

from disstruments.config import settings
from disstruments.ml.datasets.openmic import OPENMIC_CLASSES, class_to_node
from disstruments.ml.taxonomy import get_taxonomy
from disstruments.pipeline import tagging


def test_every_new_label_names_a_taxonomy_node_and_none_shadows_openmic():
    tax = get_taxonomy()
    assert set(tagging.EXTRA_AUDIOSET_MAP.values()) == set(tagging.EXTRA_LABEL_NODES)
    for label, node in tagging.EXTRA_LABEL_NODES.items():
        assert node in tax, (label, node)
        assert label not in OPENMIC_CLASSES, "an OpenMIC label would inherit its Platt fit"
    assert tagging.label_nodes() == {**class_to_node(tax), **tagging.EXTRA_LABEL_NODES}


def test_new_classes_never_feed_a_calibrated_label():
    base = {k: v for k, v in tagging.AUDIOSET_MAP.items() if k not in tagging.EXTRA_AUDIOSET_MAP}
    assert not set(tagging.EXTRA_AUDIOSET_MAP) & set(base)
    assert set(base.values()) <= set(OPENMIC_CLASSES)


def test_tag_flags_calibration_per_label_and_sets_nodes(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"classes": {"voice": {"a": 1.0, "b": 3.0}}}))
    monkeypatch.setattr(tagging, "CALIBRATION_PATH", path)
    monkeypatch.setattr(tagging, "_calib", None)
    monkeypatch.setattr(settings, "panns_calibrated", True)
    monkeypatch.setattr(settings, "fake_ml", False)
    frames = np.full(100, 0.5)
    monkeypatch.setattr(tagging, "panns_framewise",
                        lambda y: ({"voice": frames, "tambourine": frames}, 0.1))
    import librosa
    monkeypatch.setattr(librosa, "load", lambda *a, **k: (np.zeros(32000, np.float32), 32000))
    out = {t["label"]: t for t in tagging.tag(tmp_path / "x.wav")}
    assert out["voice"]["calibrated"] is True and out["voice"]["node"] == "voice"
    assert out["tambourine"]["calibrated"] is False, "no Platt fit: must be shown as '?'"
    assert out["tambourine"]["node"] == "percussion.tambourine"
    assert out["tambourine"]["confidence"] == 0.5, "uncalibrated labels keep their raw score"


def test_version_names_the_label_map():
    v = tagging.version()
    assert v.endswith(f"+map@{tagging.MAP_SHA[:8]}")
