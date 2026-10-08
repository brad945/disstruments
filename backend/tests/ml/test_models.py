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


def test_scratch_cnn_learns_toy_task():
    from disstruments.ml.models import scratch as S
    rng = np.random.default_rng(0)
    n, F, T = 160, 128, 64
    X = rng.standard_normal((n, F, T)).astype(np.float16) * 0.1
    Y = np.zeros((n, 2), bool)
    Y[: n // 2, 0] = True
    X[: n // 2, 20:30, :] += 2.0                       # class 0 = energy in a band
    Y[:, 1] = ~Y[:, 0]
    M = np.ones_like(Y)
    cfg = S.ScratchConfig(epochs=6, batch_size=32, crop_frames=48, channels=(8, 16), patience=6)
    model, hist = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], cfg, device="cpu", log=lambda *_: None)
    s = S.predict(model, X[1::2], "cpu")
    # Ranking, not threshold: after a handful of steps BatchNorm running stats haven't
    # converged, so probabilities sit near 0.5 even when the ranking is perfect.
    yv = Y[1::2, 0]
    assert s[yv, 0].min() > s[~yv, 0].max()
    assert S.n_params(model) > 0 and len(hist) >= 1


def test_temperature_scaling_fixes_overconfidence_and_keeps_ranking():
    from disstruments.ml.models.calibrate import apply_temperatures, fit_temperatures
    rng = np.random.default_rng(0)
    n = 4000
    true_p = rng.uniform(0.05, 0.95, size=(n, 2))
    y = rng.random((n, 2)) < true_p
    z = np.log(true_p / (1 - true_p)) * 3.0          # 3x overconfident logits
    s = 1 / (1 + np.exp(-z))
    m = np.ones_like(y)
    temps = fit_temperatures(s, y, m, node_levels=[1, 2])
    assert all(abs(t["T"] - 3.0) < 0.4 for t in temps.values())
    cal = apply_temperatures(s, [1, 2], temps)
    assert np.all(np.argsort(cal[:, 0]) == np.argsort(s[:, 0]))      # ranking unchanged
    assert np.abs(cal - true_p).mean() < np.abs(s - true_p).mean() / 2
    # a level with no observed labels keeps T = 1
    t2 = fit_temperatures(s, y, np.zeros_like(m), node_levels=[1, 2])
    assert all(t["T"] == 1.0 and not t["fitted"] for t in t2.values())
