"""M5: per-class Platt scaling for the incumbent PANNs tagger."""
import json

import numpy as np
import pytest

from disstruments.config import settings
from disstruments.ml.models.calibrate import apply_platt, fit_platt
from disstruments.pipeline import tagging


def test_platt_recovers_scale_and_bias_and_keeps_ranking():
    rng = np.random.default_rng(0)
    z = rng.normal(-2.5, 1.5, 20000)                       # raw scores sit far below 0.3
    p_raw = 1 / (1 + np.exp(-z))
    y = rng.random(z.size) < 1 / (1 + np.exp(-(1.5 * z + 3.0)))
    f = fit_platt(p_raw[:, None], y[:, None].astype(float), np.ones((z.size, 1)))[0]
    assert f["fitted"] and f["a"] == pytest.approx(1.5, abs=0.1) and f["b"] == pytest.approx(3.0, abs=0.15)
    assert f["nll_after"] < f["nll_before"]
    p = apply_platt(p_raw, f["a"], f["b"])
    assert (np.argsort(p) == np.argsort(p_raw)).all(), "monotone: AP unchanged"
    assert (p[y] >= 0.3).mean() > (p_raw[y] >= 0.3).mean()


def test_platt_identity_without_both_classes_and_respects_mask():
    s = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([[1, 1], [1, 0], [1, 1]], float)
    m = np.array([[1, 1], [1, 0], [1, 1]], float)          # column 1: the only negative is masked
    fits = fit_platt(s, y, m)
    assert [f["fitted"] for f in fits] == [False, False]
    assert (fits[0]["a"], fits[0]["b"]) == (1.0, 0.0) and fits[1]["n_obs"] == 2


def test_tagging_applies_calibration_and_reports_version(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"classes": {"voice": {"a": 1.0, "b": 3.0}}}))
    monkeypatch.setattr(tagging, "CALIBRATION_PATH", path)
    monkeypatch.setattr(tagging, "_calib", None)
    monkeypatch.setattr(settings, "panns_calibrated", True)
    monkeypatch.setattr(settings, "fake_ml", False)
    raw = {"voice": np.array([0.1, 0.2]), "drums": np.array([0.1, 0.2])}
    out = tagging.calibrate(raw)
    assert (out["voice"] > 0.6).all() and (out["drums"] == raw["drums"]).all()  # unknown class: as-is
    assert tagging.version().startswith("cnn14+platt@")
    monkeypatch.setattr(settings, "panns_calibrated", False)
    assert tagging.calibrate(raw) is raw and tagging.version() == "cnn14"


def test_stems_only_vouch_for_their_own_instrument():
    def t(label, conf, stem):
        return {"label": label, "confidence": conf, "stem": stem, "activations": []}
    mix = [t("voice", 0.5, "mix"), t("guitar", 0.7, "mix")]
    stems = [t("voice", 0.9, "vocals"), t("ukulele", 0.87, "vocals"), t("cello", 0.6, "other"),
             t("guitar", 0.95, "other"), t("synthesizer", 0.9, "bass"), t("drums", 0.97, "drums")]
    out = {x["label"]: x for x in tagging.combine(mix, stems)}
    assert set(out) == {"voice", "guitar", "drums"}
    assert out["voice"]["stem"] == "vocals" and out["voice"]["confidence"] == 0.9
    assert out["guitar"]["stem"] == "mix", "the residual stem never vouches"
