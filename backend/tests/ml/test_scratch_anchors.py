"""Anchor plan Phase 2 anchors in scratch.train (docs/debriefs/anchor-plan.md): hierarchy-gated
distillation, synthetic replay, LP-FT head-first phase."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from disstruments.ml.models import scratch as S  # noqa: E402
from disstruments.ml.taxonomy import Taxonomy  # noqa: E402


class _Tree:
    """Tiny taxonomy stand-in: a -> a.x, a.y ; b -> b.z (nodes in pre-order)."""
    nodes = ("a", "a.x", "a.y", "b", "b.z")

    def __len__(self):
        return 5

    def index(self):
        return {n: i for i, n in enumerate(self.nodes)}

    def descendants(self, n):
        return tuple(m for m in self.nodes if m.startswith(n + "."))


def test_descendant_matrix_is_strict_and_transitive():
    D = S.descendant_matrix(_Tree())
    assert D[0, 1] and D[0, 2] and D[3, 4]
    assert not D.diagonal().any() and not D[1, 0] and not D[0, 4]
    real = S.descendant_matrix(Taxonomy.load())
    assert real.shape == (86, 86) and real.sum() > 86 and not real.diagonal().any()


def test_distill_gate_only_unobserved_descendants_of_observed_positives():
    D = S.descendant_matrix(_Tree())
    #            a      a.x    a.y    b      b.z
    Y = np.array([[1, 0, 0, 0, 0],      # a positive, children unobserved -> gate a.x, a.y
                  [1, 1, 0, 0, 0],      # a.x observed positive: not distilled; a.y gated
                  [0, 0, 0, 0, 0],      # b observed negative (propagated to b.z): nothing
                  [1, 0, 0, 0, 0]], bool)   # a positive but NOT observed (should not happen) -> nothing
    M = np.array([[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0]], bool)
    g = S.distill_gate(Y, M, D)
    assert g.tolist() == [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert not (g & M).any()
    # per-node quantile: with q=0.5 only clips at/above the median teacher prob keep the gate
    p = np.zeros((4, 5)); p[:, 2] = [0.1, 0.9, 0.0, 0.0]
    gq = S.distill_gate(Y, M, D, teacher_p=p, quantile=0.5)
    assert gq[:, 2].tolist() == [0, 1, 0, 0] and gq[:, 1].tolist() == [1, 0, 0, 0]


def test_distill_loss_zero_at_teacher_and_positive_away_from_it():
    z = torch.randn(6, 5)
    g = torch.ones(6, 5)
    assert float(S.distill_loss(z, z.clone(), g, 2.0)) > 0          # BCE vs soft target is entropy, not 0 ...
    far = S.distill_loss(z + 3, z, g, 2.0)
    assert float(far) > float(S.distill_loss(z, z.clone(), g, 2.0))   # ... but grows with distance
    assert float(S.distill_loss(z, z, torch.zeros(6, 5), 2.0)) == 0.0  # empty gate -> 0, no NaN
    zz = z.clone().requires_grad_(True)
    S.distill_loss(zz, z, g, 2.0).backward()
    assert torch.allclose(zz.grad, torch.zeros_like(zz), atol=1e-6)   # gradient vanishes at the teacher


def _toy(n=96, F=128, T=64, K=4, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, F, T)) * 0.1).astype(np.float16)
    Y = np.zeros((n, K), bool)
    Y[: n // 2, 0] = True
    X[: n // 2, 20:30, :] += 2.0
    Y[:, 1] = ~Y[:, 0]
    M = np.ones_like(Y)
    return X, Y, M


def _cfg(**kw):
    base = dict(epochs=2, batch_size=16, crop_frames=48, channels=(8, 16), patience=4, lr=2e-3)
    base.update(kw)
    return S.ScratchConfig(**base)


def test_distillation_pulls_gated_outputs_toward_the_teacher():
    X, Y, M = _toy()
    base, _ = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], _cfg(), device="cpu", log=lambda *_: None)
    init = {k: t.clone() for k, t in base.state_dict().items()}
    # teacher = the init model; gate columns 2-3 (unobserved) on every clip; mask them out of BCE
    Mk = M.copy(); Mk[:, 2:] = False
    tl = S.predict(base, X[::2], "cpu", logits=True)
    G = np.zeros_like(Y[::2]); G[:, 2:] = True
    cfg = _cfg(seed=3, epochs=3, lr=1e-2)
    free, _ = S.train(X[::2], Y[::2], Mk[::2], X[1::2], Y[1::2], M[1::2], cfg, device="cpu",
                      log=lambda *_: None, init_state=init)
    diag = {}
    held, hist = S.train(X[::2], Y[::2], Mk[::2], X[1::2], Y[1::2], M[1::2],
                         S.ScratchConfig(**{**cfg.__dict__, "distill_lambda": 10.0}), device="cpu",
                         log=lambda *_: None, init_state=init, distill=(tl, G), diag=diag)
    t = tl[:, 2:]
    d_free = np.abs(S.predict(free, X[::2], "cpu", logits=True)[:, 2:] - t).mean()
    d_held = np.abs(S.predict(held, X[::2], "cpu", logits=True)[:, 2:] - t).mean()
    assert d_held < d_free
    assert diag["distill"]["lambda"] == 10.0 and diag["distill"]["gated_entries_per_clip"] == 2.0
    assert all("distill_loss" in h for h in hist)


def test_replay_appends_synthetic_clips_with_their_own_labels():
    X, Y, M = _toy()
    Xr, Yr, Mr = _toy(n=40, seed=1)
    Yr[:] = False; Yr[:, 3] = True                      # replay clips carry a label the real set never has
    Mr[:] = True
    cfg = _cfg(replay_frac=0.25)
    tt = lambda a: torch.from_numpy(a.astype(np.float32))  # noqa: E731
    b = S._Batcher(X, tt(Y), tt(M), cfg, np.random.default_rng(0), replay=(Xr, tt(Yr), tt(Mr)), n_replay=4)
    xb, yb, mb, tb, gb = b.make(np.arange(16))
    assert xb.shape == (20, 128, cfg.crop_frames) and yb.shape == (20, 4)
    assert torch.equal(yb[:16], tt(Y)[:16]) and torch.equal(mb[:16], tt(M)[:16])   # real rows first, untouched
    assert yb[16:, 3].all() and not yb[16:, :3].any() and mb[16:].all()            # replay rows: their own labels
    assert not gb[16:].any() and not tb[16:].any()                                   # never distilled
    # end to end: the replay-only label is learned only when replay is on (direction, small toy)
    diag = {}
    m, _ = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], _cfg(replay_frac=0.25, epochs=4, lr=1e-2),
                   device="cpu", log=lambda *_: None, replay=(Xr, Yr, Mr), diag=diag)
    assert diag["replay_per_batch"] == 4
    d0 = {}
    m0, _ = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], _cfg(epochs=4, lr=1e-2),
                    device="cpu", log=lambda *_: None, diag=d0)
    assert d0["replay_per_batch"] == 0
    assert S.predict(m, Xr, "cpu")[:, 3].mean() > S.predict(m0, Xr, "cpu")[:, 3].mean()


def test_lp_ft_trains_only_the_head_first_then_everything():
    X, Y, M = _toy()
    base, _, = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], _cfg(), device="cpu", log=lambda *_: None)
    init = {k: t.clone() for k, t in base.state_dict().items()}
    diag = {}
    m, hist = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], _cfg(seed=2, lp_epochs=2, lp_lr=1e-2),
                      device="cpu", log=lambda *_: None, init_state=init, diag=diag)
    lp = diag["lp"]
    assert lp["epochs"] == 2 and len(lp["hist"]) == 2
    ch = lp["weight_change_after_lp"]
    assert ch["head"] > 0 and all(ch[b] == pytest.approx(0.0) for b in ch if b != "head")
    # BN running stats untouched during LP (trunk frozen)
    assert diag["epochs_run"] == len(hist) == 2
    final = diag["relative_weight_change"]
    assert final["block1"] > 0                           # the main phase then moved the trunk
