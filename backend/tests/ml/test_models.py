"""M2 S5/S6: embedding cache (with a fake backbone) and the linear probe."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from disstruments.ml.models.embed import Backbone, build_cache, load_cache  # noqa: E402
from disstruments.ml.models.probe import LinearProbe, ProbeConfig  # noqa: E402


def _fake_backbone():
    rng = np.random.default_rng(0)
    proj = rng.standard_normal((3, 64, 16)).astype(np.float32)

    def embed(batch):
        out = []
        for y in batch:
            spec = np.abs(np.fft.rfft(y[:2048]))[:64].astype(np.float32)
            out.append(np.einsum("f,lfd->ld", spec, proj))
        return np.stack(out)
    return Backbone("fake", "none", 16000, "CC0-1.0", True, embed)


def test_cache_roundtrip_and_resume(tmp_path):
    bb = _fake_backbone()
    items = [(f"c{i}", tmp_path / f"{i}.wav") for i in range(10)]
    loader = lambda p, sr: np.sin(np.arange(16000) * (int(p.stem) + 1) / 50).astype(np.float32)  # noqa: E731
    out = build_cache(items, bb, tmp_path / "fake" / "x_mix.npz", batch_size=3, loader=loader)
    ids, emb, meta = load_cache(out)
    assert ids == [i for i, _ in items] and emb.shape == (10, 3, 16) and emb.dtype == np.float16
    assert meta["license"] == "CC0-1.0" and meta["n"] == 10


def test_probe_learns_separable_labels():
    rng = np.random.default_rng(1)
    n, L, D, K = 600, 3, 8, 4
    X = rng.standard_normal((n, L, D)).astype(np.float32)
    Y = np.zeros((n, K), bool)
    Y[:, 0] = X[:, 2, 0] > 0                      # signal lives in layer 2
    Y[:, 1] = X[:, 2, 1] + X[:, 2, 2] > 0.5
    Y[:, 2] = True                                # never negative
    M = np.ones_like(Y)
    M[: n // 2, 3] = False                        # half unobserved: must not count
    Y[:, 3] = X[:, 2, 3] > 0
    tr, va = slice(0, 400), slice(400, None)
    p = LinearProbe(L, D, K, ProbeConfig(epochs=80, patience=20)).fit(
        X[tr], Y[tr], M[tr], X[va], Y[va], M[va])
    s = p.predict(X[va])
    acc = ((s[:, 0] > 0.5) == Y[va, 0]).mean()
    assert acc > 0.9
    w = p.layer_weights()
    assert w[2] == max(w)                         # learned the informative layer
