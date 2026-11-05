"""Anchor plan, Phase 0 (docs/debriefs/anchor-plan.md): zero-cost diagnostics on checkpoints
that already exist. No training; selection reads only the OpenMIC validation carve-out and
the synthetic v1 validation split, never a test set.

- `wise`:  weight interpolation theta(alpha) = (1-alpha) theta_c1 + alpha theta_c3 between the
           synthetic-only model and its real-data fine-tune (Wortsman et al. 2022). The
           untrained-11 head rows (OpenMIC classes with no synthetic positives) stay at c3,
           because they were random in c1. A concave curve means both models sit in one
           low-loss basin; a dip means the fine-tune left it.
- `refit`: freeze the c3 trunk and refit only the leaf head rows on synthetic train. If leaf
           skill returns, the subtype information survived in the features ("suppressed,
           not erased", Ramasesh et al. 2021); if not, the trunk itself drifted.
- `cka`:   per-block linear CKA between c1 and c3 on the same clips, pooled-feature L2
           change, and per-block relative weight change: which layers moved.

    python -m disstruments.ml.models.anchor wise --c1 c1.pt --c3 c3.pt --out preds/anchor/p0/wise_s0 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..datasets import load_dataset
from ..datasets.openmic import class_to_node
from ..eval import metrics as M
from ..eval.harness import label_items, label_matrices
from ..taxonomy import Taxonomy

OPENMIC_VAL_FRACTION = 0.15          # the triangle's carve-out: same clips in every arm


# ----------------------------------------------------------------------------- pure parts

def interpolate_state(s1: dict, s3: dict, alpha: float, keep_rows: Sequence[int] = (),
                      head_keys: Sequence[str] = ("head.weight", "head.bias")) -> dict:
    """theta(alpha) over every float tensor (conv, BN affine and BN running stats included).
    Integer buffers (BN `num_batches_tracked`) are taken from s3. Rows `keep_rows` of the
    head tensors are taken from s3 whatever alpha is."""
    import torch
    out = {}
    for k, t1 in s1.items():
        t3 = s3[k]
        if not torch.is_floating_point(t1):
            out[k] = t3.clone()
            continue
        t = (1.0 - alpha) * t1 + alpha * t3
        if k in head_keys and len(keep_rows):
            t = t.clone()
            t[list(keep_rows)] = t3[list(keep_rows)]
        out[k] = t
    return out


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centred kernel alignment between two feature matrices on the same N items
    (Kornblith et al. 2019): 1 = identical up to rotation/scale, ~0 = unrelated."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    num = np.linalg.norm(x.T @ y, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den) if den > 0 else float("nan")


def subset_map(y: np.ndarray, s: np.ndarray, m: np.ndarray, cols: Sequence[int]) -> float | None:
    """Mean AP over the given node columns, skipping unscoreable ones (as the harness does)."""
    ap, reasons = M.per_class_ap(y, s, m)
    return M.macro(ap, reasons, np.asarray(cols, dtype=int)).value


BN_STAT_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def swap_bn_stats(dst: dict, src: dict) -> dict:
    """`dst` with its BatchNorm running statistics replaced by `src`'s (weights untouched).
    Evaluation-only diagnostic: separates "BN statistics adapted to the new data" from
    "weights travelled" as causes of a score change."""
    out = dict(dst)
    out.update({k: src[k] for k in src if k.endswith(BN_STAT_SUFFIXES)})
    return out


def node_sets(tax: Taxonomy) -> dict[str, list[int]]:
    """Column indices in `tax.nodes` for the plan's node lists."""
    from ..eval.triangle import v1_trained
    trained = v1_trained(tax)
    om = list(class_to_node(tax).values())
    ix = tax.index()
    return {"trained9": [ix[n] for n in om if n in trained],
            "untrained11": [ix[n] for n in om if n not in trained],
            "all20": [ix[n] for n in om],
            "leaf": [ix[n] for n in sorted(trained) if tax.is_leaf(n)]}


from .scratch import relative_weight_change  # noqa: E402  (per-block ||d theta|| / ||theta||)


# ----------------------------------------------------------------------- model features

def block_features(model, X, dev: str, batch: int = 64):
    """-> (pooled 512-d features (N, 2C), [per-block spatially pooled features (N, C_k)]).
    `pooled @ head` reproduces `model(x)`; the per-block vectors feed CKA."""
    import torch
    model.eval()
    blocks = [m for m in model.body]
    n_blocks = len(blocks) // 7
    pooled, per_block = [], [[] for _ in range(n_blocks)]
    with torch.no_grad():
        for k in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[k:k + batch]).astype(np.float32)).to(dev)
            h = model.bn0(xb[:, None])
            for b in range(n_blocks):
                for mod in blocks[7 * b: 7 * b + 7]:
                    h = mod(h)
                per_block[b].append(h.mean(dim=(2, 3)).cpu().numpy())
            h = h.mean(dim=2)
            pooled.append(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1).cpu().numpy())
    return np.concatenate(pooled), [np.concatenate(p) for p in per_block]


def head_scores(model, feats: np.ndarray, dev: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.sigmoid(model.head(torch.from_numpy(feats).to(dev))).cpu().numpy()


# -------------------------------------------------------------------------- data access

def _rows(cache_ids, X, ids):
    from .__main__ import _Rows
    pos = {i: k for k, i in enumerate(cache_ids)}
    return _Rows(X, [pos[i] for i in ids])


def openmic_val(cache: Path, root: Path, tax: Taxonomy):
    from . import scratch as S
    idx = load_dataset("openmic", root, val_fraction=OPENMIC_VAL_FRACTION, seed=0,
                       check_split_leakage=False)
    ids, y, m = label_matrices(label_items(idx.split("val"), "mix"), tax)
    cids, X = S.load_mel_cache(cache / "mels" / "openmic_train_mix.npz")
    return _rows(cids, X, ids), y, m


def synthetic_split(cache: Path, root: Path, tax: Taxonomy, split: str):
    from . import scratch as S
    idx = load_dataset("synthetic", root)
    ids, y, m = label_matrices(label_items(idx.split(split), "mix"), tax)
    cids, X = S.load_mel_cache(cache / "mels" / f"synthetic_{split}_mix.npz")
    return _rows(cids, X, ids), y, m


def _load(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state"], ck["cfg"]


def _model_from(state: dict, cfg: dict, n_out: int, dev: str):
    from . import scratch as S
    c = dict(cfg)
    c["channels"] = tuple(c["channels"])
    model = S.make_model(n_out, S.ScratchConfig(**c))
    model.load_state_dict(state)
    return model.to(dev).eval()


def _score(model, X, y, m, tax: Taxonomy, dev: str, sets: dict[str, list[int]], which: Sequence[str]):
    from . import scratch as S
    s = tax.max_propagate(S.predict(model, X, dev))        # harness convention
    return {k: subset_map(y, s, m, sets[k]) for k in which}


# -------------------------------------------------------------------------------- CLI

def cmd_wise(a) -> int:
    from .embed import _device
    dev = _device()
    tax = Taxonomy.load()
    sets = node_sets(tax)
    s1, cfg = _load(a.c1)
    s3, _ = _load(a.c3)
    Xo, yo, mo = openmic_val(a.cache, a.openmic_root, tax)
    Xs, ys, ms = synthetic_split(a.cache, a.train_root, tax, "val")
    Xo, Xs = Xo.materialize(), Xs.materialize()
    rows = []
    for alpha in a.alphas:
        t0 = time.time()
        st = interpolate_state(s1, s3, alpha, keep_rows=sets["untrained11"])
        model = _model_from(st, cfg, len(tax), dev)
        r = {"alpha": alpha}
        r.update({f"openmic_val_{k}": v for k, v in
                  _score(model, Xo, yo, mo, tax, dev, sets, ("trained9", "all20", "untrained11")).items()})
        r["synthetic_val_leaf"] = _score(model, Xs, ys, ms, tax, dev, sets, ("leaf",))["leaf"]
        r["sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"alpha {alpha:.2f}  om trained9 {r['openmic_val_trained9']:.4f}  all20 {r['openmic_val_all20']:.4f}"
              f"  untrained11 {r['openmic_val_untrained11']:.4f}  syn leaf {r['synthetic_val_leaf']:.4f}", flush=True)
    _write(a.out, "wise", {"c1": str(a.c1), "c3": str(a.c3), "keep_rows": "untrained11",
                           "n_openmic_val": len(yo), "n_synthetic_val": len(ys), "rows": rows})
    return 0


def cmd_refit(a) -> int:
    """Leaf-head refit on frozen trunks: c3 (the question) and c1 (the ceiling of the
    procedure itself). Reports synthetic-val leaf AP for original heads and refit heads."""
    from .embed import _device
    from .probe import LinearProbe, ProbeConfig
    dev = _device()
    tax = Taxonomy.load()
    sets = node_sets(tax)
    leaf = np.asarray(sets["leaf"])
    Xtr, ytr, mtr = synthetic_split(a.cache, a.train_root, tax, "train")
    Xva, yva, mva = synthetic_split(a.cache, a.train_root, tax, "val")
    out = {"n_train": len(ytr), "n_val": len(yva), "models": {}}
    for name, path in (("c1", a.c1), ("c3", a.c3)):
        st, cfg = _load(path)
        model = _model_from(st, cfg, len(tax), dev)
        ftr, _ = block_features(model, Xtr, dev)
        fva, _ = block_features(model, Xva, dev)
        orig = tax.max_propagate(head_scores(model, fva, dev))
        orig_ap = subset_map(yva, orig, mva, leaf)
        probe = LinearProbe(1, ftr.shape[1], len(leaf), ProbeConfig(epochs=200, patience=15, seed=a.seed))
        probe.fit(ftr[:, None, :], ytr[:, leaf], mtr[:, leaf], fva[:, None, :], yva[:, leaf], mva[:, leaf])
        s = orig.copy()
        s[:, leaf] = probe.predict(fva[:, None, :])
        refit_ap = subset_map(yva, tax.max_propagate(s), mva, leaf)
        out["models"][name] = {"path": str(path), "synthetic_val_leaf_original": orig_ap,
                               "synthetic_val_leaf_refit": refit_ap, "probe_epochs": len(probe.history)}
        print(f"{name}: synthetic-val leaf AP original {orig_ap:.4f} -> refit head {refit_ap:.4f}", flush=True)
    _write(a.out, "refit", out)
    return 0


def cmd_cka(a) -> int:
    from .embed import _device
    dev = _device()
    tax = Taxonomy.load()
    s1, cfg = _load(a.c1)
    s3, _ = _load(a.c3)
    m1, m3 = _model_from(s1, cfg, len(tax), dev), _model_from(s3, cfg, len(tax), dev)
    rng = np.random.default_rng(0)
    out = {"c1": str(a.c1), "c3": str(a.c3), "n_clips": a.n_clips,
           "relative_weight_change": relative_weight_change(s1, s3), "features": {}}
    for name, (X, *_) in (("synthetic_val", synthetic_split(a.cache, a.train_root, tax, "val")),
                          ("openmic_val", openmic_val(a.cache, a.openmic_root, tax))):
        pick = np.sort(rng.permutation(len(X))[: a.n_clips])
        Xs = np.asarray([X[int(i)] for i in pick])
        p1, b1 = block_features(m1, Xs, dev)
        p3, b3 = block_features(m3, Xs, dev)
        rel = float(np.linalg.norm(p3 - p1, axis=1).mean() / (np.linalg.norm(p1, axis=1).mean() + 1e-12))
        out["features"][name] = {"cka_per_block": [linear_cka(x, y) for x, y in zip(b1, b3)],
                                 "cka_pooled": linear_cka(p1, p3), "pooled_relative_l2_change": rel}
        print(f"{name}: CKA per block {[round(v, 3) for v in out['features'][name]['cka_per_block']]}"
              f"  pooled {out['features'][name]['cka_pooled']:.3f}  rel L2 change {rel:.3f}", flush=True)
    print("relative weight change:", {k: round(v, 4) for k, v in out["relative_weight_change"].items()})
    _write(a.out, "cka", out)
    return 0


def cmd_score(a) -> int:
    """Selection metrics for an existing checkpoint (e.g. the triangle's 2e-4 fine-tunes):
    OpenMIC-val trained-9 / all-20 / untrained-11 and synthetic-val leaf AP."""
    from .embed import _device
    dev = _device()
    tax = Taxonomy.load()
    sets = node_sets(tax)
    st, cfg = _load(a.model)
    model = _model_from(st, cfg, len(tax), dev)
    Xo, yo, mo = openmic_val(a.cache, a.openmic_root, tax)
    Xs, ys, ms = synthetic_split(a.cache, a.train_root, tax, "val")
    vm = {f"openmic_val_{k}": v for k, v in
          _score(model, Xo.materialize(), yo, mo, tax, dev, sets, ("trained9", "all20", "untrained11")).items()}
    vm["synthetic_val_leaf"] = _score(model, Xs.materialize(), ys, ms, tax, dev, sets, ("leaf",))["leaf"]
    print("val metrics:", {k: round(v, 4) for k, v in vm.items() if v is not None})
    _write(a.out, "score", {"model": str(a.model), "val_metrics": vm})
    return 0


def cmd_bnswap(a) -> int:
    """For each fine-tuned checkpoint: score as is, with c1's BN statistics swapped in, and c1
    with the checkpoint's BN statistics swapped in."""
    from .embed import _device
    dev = _device()
    tax = Taxonomy.load()
    sets = node_sets(tax)
    s1, cfg1 = _load(a.c1)
    Xo, yo, mo = openmic_val(a.cache, a.openmic_root, tax)
    Xs, ys, ms = synthetic_split(a.cache, a.train_root, tax, "val")
    Xo, Xs = Xo.materialize(), Xs.materialize()

    def score(st, cfg):
        m = _model_from(st, cfg, len(tax), dev)
        r = {f"openmic_val_{k}": v for k, v in
             _score(m, Xo, yo, mo, tax, dev, sets, ("trained9", "all20", "untrained11")).items()}
        r["synthetic_val_leaf"] = _score(m, Xs, ys, ms, tax, dev, sets, ("leaf",))["leaf"]
        return r

    rows = []
    for path in a.models:
        s3, cfg3 = _load(path)
        rows.append({"model": str(path), "as_is": score(s3, cfg3),
                     "model_with_c1_bn": score(swap_bn_stats(s3, s1), cfg3),
                     "c1_with_model_bn": score(swap_bn_stats(s1, s3), cfg1)})
        print(path.parent.name, {k: round(v["synthetic_val_leaf"], 3) for k, v in rows[-1].items() if k != "model"})
    _write(a.out, "bnswap", {"c1": str(a.c1), "rows": rows})
    return 0


def _write(out: Path, name: str, doc: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {out / f'{name}.json'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.models.anchor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--c1", type=Path, required=True, help="synthetic-only checkpoint (.pt)")
        sp.add_argument("--c3", type=Path, required=True, help="its real-data fine-tune (.pt)")
        sp.add_argument("--cache", type=Path, required=True, help="mel cache root (has mels/)")
        sp.add_argument("--train-root", type=Path, required=True, help="synthetic v1 root")
        sp.add_argument("--openmic-root", type=Path, required=True)
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--seed", type=int, default=0)

    w = sub.add_parser("wise", help="weight-interpolation curve c1 -> c3")
    common(w)
    w.add_argument("--alphas", type=float, nargs="+", default=[round(0.1 * i, 1) for i in range(11)])
    w.set_defaults(fn=cmd_wise)
    r = sub.add_parser("refit", help="leaf-head refit on the frozen c3 (and c1) trunk")
    common(r)
    r.set_defaults(fn=cmd_refit)
    c = sub.add_parser("cka", help="per-block CKA and weight change between c1 and c3")
    common(c)
    c.add_argument("--n-clips", type=int, default=500)
    c.set_defaults(fn=cmd_cka)
    s = sub.add_parser("score", help="selection metrics for one existing checkpoint")
    s.add_argument("--model", type=Path, required=True)
    for flag, kw in (("--cache", {}), ("--train-root", {}), ("--openmic-root", {}), ("--out", {})):
        s.add_argument(flag, type=Path, required=True, **kw)
    s.set_defaults(fn=cmd_score)
    b = sub.add_parser("bnswap", help="BN-statistics swap diagnostic between c1 and fine-tuned checkpoints")
    b.add_argument("--c1", type=Path, required=True)
    b.add_argument("--models", type=Path, nargs="+", required=True)
    for flag in ("--cache", "--train-root", "--openmic-root", "--out"):
        b.add_argument(flag, type=Path, required=True)
    b.set_defaults(fn=cmd_bnswap)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
