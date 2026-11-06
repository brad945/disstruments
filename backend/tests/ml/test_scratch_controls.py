"""Anchor plan Phase 1 recipe controls in scratch.train (docs/debriefs/anchor-plan.md)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from disstruments.ml.models import scratch as S  # noqa: E402


def _toy(n=96, F=128, T=64, K=4):
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((n, F, T)) * 0.1).astype(np.float16)
    Y = np.zeros((n, K), bool)
    Y[: n // 2, 0] = True
    X[: n // 2, 20:30, :] += 2.0
    Y[:, 1] = ~Y[:, 0]
    Y[:, 2] = rng.random(n) < 0.5
    M = np.ones_like(Y)
    return X, Y, M


def _run(cfg, init_state=None):
    X, Y, M = _toy()
    diag = {}
    model, hist = S.train(X[::2], Y[::2], M[::2], X[1::2], Y[1::2], M[1::2], cfg, device="cpu",
                          log=lambda *_: None, init_state=init_state, diag=diag)
    return model, hist, diag


def _cfg(**kw):
    base = dict(epochs=2, batch_size=16, crop_frames=48, channels=(8, 16), patience=4, lr=2e-3)
    base.update(kw)
    return S.ScratchConfig(**base)


def test_flat_decay_factor_shape():
    assert S.flat_decay_factor(0, 100, 0.7) == 1.0
    assert S.flat_decay_factor(69, 100, 0.7) == 1.0
    assert S.flat_decay_factor(85, 100, 0.7) == pytest.approx(0.5, abs=0.04)
    assert S.flat_decay_factor(100, 100, 0.7) == 0.0


def test_block_of_groups_state_dict_keys():
    assert S.block_of("body.0.weight") == "block1" and S.block_of("body.6.weight") == "block1"
    assert S.block_of("body.7.weight") == "block2" and S.block_of("body.21.weight") == "block4"
    assert S.block_of("head.weight") == "head" and S.block_of("bn0.running_mean") == "bn0"


def test_freeze_bn_stats_keeps_running_statistics():
    base, _, _ = _run(_cfg())
    init = {k: t.clone() for k, t in base.state_dict().items()}
    moved, _, _ = _run(_cfg(seed=3), init_state=init)
    frozen, _, d = _run(_cfg(seed=3, freeze_bn_stats=True), init_state=init)
    rk = next(k for k in init if k.endswith("running_mean") and k.startswith("body"))
    assert not torch.allclose(moved.state_dict()[rk], init[rk])      # normal training updates them
    assert torch.allclose(frozen.state_dict()[rk], init[rk])         # frozen keeps them
    assert d["freeze_bn_stats"] is True


def test_frozen_rows_and_frozen_blocks_do_not_move():
    base, _, _ = _run(_cfg())
    init = {k: t.clone() for k, t in base.state_dict().items()}
    m, _, d = _run(_cfg(seed=5, frozen_rows=(1, 3), freeze_blocks=(2,)), init_state=init)
    sd = m.state_dict()
    assert torch.allclose(sd["head.weight"][[1, 3]], init["head.weight"][[1, 3]])
    assert torch.allclose(sd["head.bias"][[1, 3]], init["head.bias"][[1, 3]])
    assert not torch.allclose(sd["head.weight"][0], init["head.weight"][0])
    for k in init:
        if S.block_of(k) == "block2" and torch.is_floating_point(init[k]):
            assert torch.allclose(sd[k], init[k]), k                  # whole block 2 frozen, BN stats included
    assert not torch.allclose(sd["body.0.weight"], init["body.0.weight"])
    assert d["relative_weight_change"]["block2"] == pytest.approx(0.0)
    assert d["relative_weight_change"]["block1"] > 0
    assert d["n_trainable"] < S.n_params(m) and d["frozen_rows"] == 2 and d["freeze_blocks"] == [2]


def test_head_lr_group_flat_decay_schedule_and_diagnostics():
    m, hist, d = _run(_cfg(head_lr=1e-2, schedule="flat-decay"))
    assert d["lr_groups"] == [2e-3, 1e-2] and d["schedule"] == "flat-decay"
    assert set(d["rgn_epoch1"]) >= {"block1", "block2", "head"}
    assert all(v > 0 for v in d["rgn_epoch1"].values())
    assert 0 <= d["best_epoch"] < len(hist) and d["epochs_run"] == len(hist)
    assert d["train_loss_at_best"] == hist[d["best_epoch"]]["train_loss"]


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        _run(_cfg(schedule="cosine"))


def test_old_checkpoint_cfg_without_new_fields_still_loads(tmp_path):
    m, hist, _ = _run(_cfg())
    cfg = _cfg()
    S.save(m, tmp_path / "m.pt", cfg, hist, {})
    ck = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    for k in ("schedule", "head_lr", "freeze_bn_stats", "frozen_rows", "freeze_blocks", "flat_fraction"):
        ck["cfg"].pop(k)                                             # a pre-anchor checkpoint
    torch.save(ck, tmp_path / "old.pt")
    loaded = S.load_model(tmp_path / "old.pt", 4, "cpu")
    assert S.n_params(loaded) == S.n_params(m)
