"""Anchor plan Phase 0 tooling (docs/debriefs/anchor-plan.md): interpolation, CKA, features."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from disstruments.ml.models import anchor as A  # noqa: E402
from disstruments.ml.models import scratch as S  # noqa: E402


def _two_models(n_out=6):
    cfg = S.ScratchConfig(channels=(4, 8))
    torch.manual_seed(0)
    m1 = S.make_model(n_out, cfg)
    torch.manual_seed(1)
    m3 = S.make_model(n_out, cfg)
    with torch.no_grad():                       # give BN running stats something to interpolate
        for m in (m1, m3):
            m.train()
            m(torch.randn(8, S.N_MELS, 48))
            m.eval()
    return m1, m3, cfg


def test_interpolate_state_endpoints_midpoint_and_kept_rows():
    m1, m3, _ = _two_models()
    s1, s3 = m1.state_dict(), m3.state_dict()
    keep = [1, 4]
    a0 = A.interpolate_state(s1, s3, 0.0, keep_rows=keep)
    a1 = A.interpolate_state(s1, s3, 1.0, keep_rows=keep)
    ah = A.interpolate_state(s1, s3, 0.5, keep_rows=keep)
    for k in s1:
        if not torch.is_floating_point(s1[k]):
            assert torch.equal(a0[k], s3[k])                        # int buffers come from c3
            continue
        if k.startswith("head."):
            assert torch.allclose(a0[k][keep], s3[k][keep])          # kept rows always c3
            other = [i for i in range(s1[k].shape[0]) if i not in keep]
            assert torch.allclose(a0[k][other], s1[k][other])
        else:
            assert torch.allclose(a0[k], s1[k])
            assert torch.allclose(ah[k], 0.5 * (s1[k] + s3[k]))
        assert torch.allclose(a1[k], s3[k])
    # BN running stats are interpolated too
    rk = next(k for k in s1 if k.endswith("running_mean") and k.startswith("body"))
    assert not torch.allclose(ah[rk], s1[rk]) and not torch.allclose(ah[rk], s3[rk])


def test_linear_cka_identity_rotation_and_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16))
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    assert A.linear_cka(x, x) == pytest.approx(1.0)
    assert A.linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)        # invariant to rotation and scale
    assert A.linear_cka(x, rng.standard_normal((200, 16))) < 0.3


def test_block_features_reproduce_forward_logits():
    m1, _, _ = _two_models(n_out=5)
    X = (np.random.default_rng(0).standard_normal((10, S.N_MELS, 48)) * 0.1).astype(np.float16)
    pooled, per_block = A.block_features(m1, X, "cpu", batch=4)
    assert pooled.shape == (10, 2 * 8) and [b.shape[1] for b in per_block] == [4, 8]
    with torch.no_grad():
        ref = torch.sigmoid(m1(torch.from_numpy(X.astype(np.float32)))).numpy()
    assert np.allclose(A.head_scores(m1, pooled, "cpu"), ref, atol=1e-5)


def test_subset_map_skips_unscoreable_columns():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]], bool)
    s = np.array([[0.9, 0.2, 0.5], [0.1, 0.3, 0.5], [0.8, 0.1, 0.5], [0.2, 0.4, 0.5]])
    m = np.ones_like(y)
    assert A.subset_map(y, s, m, [0]) == pytest.approx(1.0)
    assert A.subset_map(y, s, m, [0, 1, 2]) == pytest.approx(1.0)   # cols 1 and 2 have no pos/neg: skipped
    assert A.subset_map(y, s, m, [1]) is None


def test_relative_weight_change_groups_by_block():
    m1, m3, _ = _two_models()
    rel = A.relative_weight_change(m1.state_dict(), m3.state_dict())
    assert set(rel) >= {"bn0", "block1", "block2", "head"}
    assert all(rel[g] > 0 for g in ("block1", "block2", "head"))    # bn0 affine init is identical (1, 0)
    same = A.relative_weight_change(m1.state_dict(), m1.state_dict())
    assert all(v == pytest.approx(0.0) for v in same.values())


def test_swap_bn_stats_moves_only_running_statistics():
    m1, m3, _ = _two_models()
    s1, s3 = m1.state_dict(), m3.state_dict()
    sw = A.swap_bn_stats(s3, s1)
    assert set(sw) == set(s3)
    for k in s3:
        src = s1 if k.endswith(A.BN_STAT_SUFFIXES) else s3
        assert torch.equal(sw[k], src[k]), k
    assert any(k.endswith("running_mean") and not torch.equal(s1[k], s3[k]) for k in s1)  # the swap is non-trivial
