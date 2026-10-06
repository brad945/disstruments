"""PANNs device selection: honour settings.device, fall back to CPU on failure."""
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _FakeModel:
    def __init__(self, fail_on):
        self.fail_on, self.device = fail_on, "cpu"

    def to(self, device):
        if device == self.fail_on:
            raise RuntimeError(f"op not supported on {device}")
        self.device = device
        return self


def _install_fake_panns(monkeypatch, fail_on):
    class FakeSED:
        def __init__(self, checkpoint_path=None, device="cpu"):
            self.model, self.device = _FakeModel(fail_on), "cpu"

        def inference(self, audio):
            return np.zeros((1, 1, 527), dtype=np.float32)

    mod = types.ModuleType("panns_inference")
    mod.SoundEventDetection = FakeSED
    monkeypatch.setitem(sys.modules, "panns_inference", mod)


@pytest.fixture
def tagging(monkeypatch):
    from disstruments.config import settings
    from disstruments.pipeline import tagging as t
    monkeypatch.setattr(t, "_sed", None)
    monkeypatch.setattr(settings, "device", "mps")
    yield t
    t._sed, t._sed_device = None, "cpu"


def test_moves_model_to_configured_device(monkeypatch, tagging):
    _install_fake_panns(monkeypatch, fail_on=None)
    sed = tagging._get_sed()
    assert tagging._sed_device == "mps" and sed.device == "mps" and sed.model.device == "mps"


def test_falls_back_to_cpu_when_device_fails(monkeypatch, tagging):
    _install_fake_panns(monkeypatch, fail_on="mps")
    sed = tagging._get_sed()
    assert tagging._sed_device == "cpu" and sed.device == "cpu" and sed.model.device == "cpu"


def test_cpu_setting_never_moves(monkeypatch, tagging):
    from disstruments.config import settings
    monkeypatch.setattr(settings, "device", "cpu")
    _install_fake_panns(monkeypatch, fail_on="cpu-should-not-be-called")
    tagging._get_sed()
    assert tagging._sed_device == "cpu"
