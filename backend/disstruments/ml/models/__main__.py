"""CLI: python -m disstruments.ml.models {embed,probe}.

    # S5: cache embeddings (per backbone, dataset, unit)
    python -m disstruments.ml.models embed --backbone clap --dataset synthetic \
        --root ~/datasets/disstruments-synth/v1 --cache ~/datasets/disstruments-synth/emb
    # S6: train on synthetic train (early stop on synthetic val), predict eval sets
    python -m disstruments.ml.models probe --backbone clap --train-root ~/datasets/disstruments-synth/v1 \
        --cache ~/datasets/disstruments-synth/emb --eval synthetic:test openmic:test --out preds/
    # S7: score each predictions file with the M1 harness
    python -m disstruments.ml.eval run --dataset openmic --root ... --split test --predictions preds/clap_openmic_test.npz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from ..datasets import load_dataset
from ..eval.harness import label_items, label_matrices
from ..eval.predictions import save_predictions
from ..taxonomy import Taxonomy
from .embed import build_cache, cache_path, load_backbone, load_cache
from .probe import LinearProbe, ProbeConfig


def _audio_items(records, unit: str) -> list[tuple[str, Path]]:
    out = []
    for r in records:
        if unit == "mix":
            if "mix" in r.audio:
                out.append((r.item_id, r.audio["mix"]))
        else:
            out += [(f"{r.item_id}/{s.stem_id}", s.audio) for s in r.stems if s.audio]
    return out


def _matrix(ids: list[str], cache_ids: list[str], emb: np.ndarray) -> np.ndarray:
    pos = {i: k for k, i in enumerate(cache_ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise ValueError(f"{len(missing)} items missing from the embedding cache, e.g. {missing[:3]}")
    return emb[[pos[i] for i in ids]].astype(np.float32)


def cmd_embed(a) -> int:
    opts = {"audio_root": a.audio_root} if a.audio_root else {}
    if a.dataset == "openmic" and a.no_split_leakage_check:
        opts["check_split_leakage"] = False
    idx = load_dataset(a.dataset, a.root, **opts)
    items = _audio_items(idx.split(a.split), a.unit)
    if not items:
        raise SystemExit(f"{a.dataset}: no audio found (unit={a.unit})")
    bb = load_backbone(a.backbone)
    out = build_cache(items, bb, cache_path(a.cache, a.backbone, a.dataset, a.unit),
                      batch_size=a.batch_size, meta={"dataset": a.dataset, "unit": a.unit,
                                                     "loader_config": idx.config})
    print(f"wrote {out} ({len(items)} items)")
    return 0


def cmd_probe(a) -> int:
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    train_idx = load_dataset("synthetic", a.train_root, commercial_only=a.commercial_only)
    units = [u.strip() for u in a.train_units.split(",") if u.strip()]
    caches = {u: load_cache(cache_path(a.cache, a.backbone, "synthetic", u)) for u in units}
    cmeta = caches[units[0]][2]

    def xy(records):
        """Training rows from every requested unit (D6: mixes and isolated stems)."""
        parts = []
        for u in units:
            cids, emb, _ = caches[u]
            ids, y, m = label_matrices(label_items(records, u), tax)
            parts.append((_matrix(ids, cids, emb), y, m))
        return (None, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))

    _, X, Y, Mk = xy(train_idx.split("train"))
    _, Xv, Yv, Mv = xy(train_idx.split("val"))
    cfg = ProbeConfig(seed=a.seed)
    probe = LinearProbe(X.shape[1], X.shape[2], len(nodes), cfg).fit(X, Y, Mk, Xv, Yv, Mv)
    train_hash = hashlib.sha256("\n".join(sorted(r.item_id for r in train_idx.split("train")))
                                .encode()).hexdigest()[:16]
    tag = "" if units == ["mix"] else "_" + "+".join(units)
    meta = {"model": f"{a.backbone}+linear_probe", "backbone": cmeta["hf_id"],
            "train_units": units,
            "backbone_license": cmeta["license"],
            "commercial_clean": bool(cmeta["commercial_clean"] and a.commercial_only),
            "train": {"dataset": "synthetic", "root_name": Path(a.train_root).name,
                      "n_train": len(X), "n_val": len(Xv), "items_sha": train_hash,
                      "commercial_only": a.commercial_only},
            "probe": {"layer_weights": probe.layer_weights(), "epochs_run": len(probe.history),
                      "best_val_map": max(h["val_map"] for h in probe.history)}}
    a.out.mkdir(parents=True, exist_ok=True)
    probe.save(a.out / f"{a.backbone}{tag}_probe.pt", extra=meta)
    for spec in a.eval:
        ds, split = spec.split(":")
        root = a.train_root if ds == "synthetic" else getattr(a, f"{ds}_root")
        if root is None:
            print(f"skip {spec}: --{ds}-root not given")
            continue
        idx = train_idx if ds == "synthetic" else load_dataset(
            ds, root, **({"audio_root": a.medleydb_audio_root} if ds == "medleydb" else
                         {"check_split_leakage": False} if ds == "openmic" else {}))
        recs = idx.split(split)
        ids = [i for i, _, _ in label_items(recs, a.unit)]
        eids, eemb, _ = load_cache(cache_path(a.cache, a.backbone, ds, a.unit))
        have = set(eids)
        ids = [i for i in ids if i in have]
        scores = probe.predict(_matrix(ids, eids, eemb))
        p = save_predictions(a.out / f"{a.backbone}{tag}_{ds}_{split}_{a.unit}.npz", ids, nodes, scores,
                             tax.version, dataset=ds, meta=meta)
        print(f"wrote {p} ({len(ids)} items; best synthetic val mAP {meta['probe']['best_val_map']:.3f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.models")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--backbone", required=True, choices=("mert95m", "clap"))
    e.add_argument("--dataset", required=True)
    e.add_argument("--root", required=True, type=Path)
    e.add_argument("--audio-root", type=Path)
    e.add_argument("--unit", default="mix", choices=("mix", "stem"))
    e.add_argument("--cache", required=True, type=Path)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--split", default="all", help="only embed this split (default: all)")
    e.add_argument("--no-split-leakage-check", action="store_true",
                   help="openmic: official split01 has 1 artist (fma:15155) in train and test")
    e.set_defaults(fn=cmd_embed)
    q = sub.add_parser("probe")
    q.add_argument("--backbone", required=True, choices=("mert95m", "clap"))
    q.add_argument("--train-root", required=True, type=Path)
    q.add_argument("--cache", required=True, type=Path)
    q.add_argument("--unit", default="mix", choices=("mix", "stem"), help="evaluation unit")
    q.add_argument("--train-units", default="mix", help="comma list: mix,stem (D6)")
    q.add_argument("--eval", nargs="*", default=["synthetic:test"],
                   help="dataset:split pairs, e.g. synthetic:test openmic:test medleydb:test")
    q.add_argument("--openmic-root", type=Path)
    q.add_argument("--medleydb-root", type=Path)
    q.add_argument("--medleydb-audio-root", type=Path)
    q.add_argument("--commercial-only", action="store_true")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", required=True, type=Path)
    q.set_defaults(fn=cmd_probe)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
