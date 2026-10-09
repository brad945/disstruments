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
    cfg = ProbeConfig(seed=a.seed, hidden=a.hidden, lr=a.lr if a.lr else (1e-3 if a.hidden else 1e-2))
    probe = LinearProbe(X.shape[1], X.shape[2], len(nodes), cfg).fit(X, Y, Mk, Xv, Yv, Mv)
    train_hash = hashlib.sha256("\n".join(sorted(r.item_id for r in train_idx.split("train")))
                                .encode()).hexdigest()[:16]
    tag = ("" if units == ["mix"] else "_" + "+".join(units)) + (f"_mlp{a.hidden}" if a.hidden else "")
    meta = {"model": f"{a.backbone}+" + (f"mlp{a.hidden}" if a.hidden else "linear_probe"),
            "backbone": cmeta["hf_id"],
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


def cmd_mels(a) -> int:
    from .scratch import build_mel_cache
    opts = {"check_split_leakage": False} if a.dataset == "openmic" else {}
    idx = load_dataset(a.dataset, a.root, **opts)
    items = _audio_items(idx.split(a.split), a.unit)
    out = a.cache / "mels" / f"{a.prefix or a.dataset}_{a.split}_{a.unit}.npz"
    build_mel_cache(items, out, workers=a.workers)
    print(f"wrote {out} ({len(items)} items)")
    return 0


COMMERCIAL_BLOCKERS = ("noncommercial", "no derivative", "noderivatives", "noderivs")


def openmic_licences(root: Path) -> dict[str, str]:
    """sample_key -> FMA licence title (per-clip licences: most OpenMIC clips are NC/ND)."""
    import csv
    with open(Path(root) / "openmic-2018-metadata.csv", newline="") as f:
        return {r["sample_key"]: r["license_title"] for r in csv.DictReader(f)}


def commercial_ok(title: str) -> bool:
    t = title.lower().replace("-", " ").replace("_", " ")
    t2 = t.replace(" ", "")
    return bool(title) and not any(b.replace(" ", "") in t2 for b in COMMERCIAL_BLOCKERS)


class _Concat:
    """Row view over several (X, rows) sources, e.g. synthetic + real co-training data."""

    def __init__(self, parts):
        self.parts = parts
        self.index = [(p, r) for p, view in enumerate(parts) for r in range(len(view))]
        self.shape = (len(self.index),) + tuple(parts[0].shape[1:])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            p, r = self.index[k[0]]
            return self.parts[p][(r,) + k[1:]]
        p, r = self.index[k]
        return self.parts[p][r]


class _Rows:
    """Lazy row view over a (possibly memmapped) mel cache: X[i] reads one clip from disk."""

    def __init__(self, X, rows):
        self.X, self.rows = X, np.asarray(rows)
        self.shape = (len(self.rows),) + tuple(X.shape[1:])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return self.X[(self.rows[k[0]],) + k[1:]]
        return self.X[self.rows[k]]

    def materialize(self):
        return np.asarray(self.X[np.sort(self.rows)])[np.argsort(np.argsort(self.rows))]


def cmd_scratch(a) -> int:
    """Rung R5 (A2): train the from-scratch CNN on synthetic train/val, predict eval sets."""
    from . import scratch as S
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    idx = load_dataset("synthetic", a.train_root, commercial_only=a.commercial_only)

    def xy(split):
        ids, y, m = label_matrices(label_items(idx.split(split), "mix"), tax)
        cids, X = S.load_mel_cache(a.cache / "mels" / f"{a.mel_prefix}_{split}_mix.npz")
        pos = {i: k for k, i in enumerate(cids)}
        rows = np.array([pos[i] for i in ids])
        if split == "train" and a.train_limit:            # data-scaling curve: nested subsets
            keep = np.random.default_rng(0).permutation(len(rows))[: a.train_limit]
            rows, y, m = rows[np.sort(keep)], y[np.sort(keep)], m[np.sort(keep)]
        return _Rows(X, rows), y, m

    X, Y, Mk = xy("train")
    Xv, Yv, Mv = xy("val")
    Xv = Xv.materialize()                                  # val is small: keep in RAM
    cot = {}
    if a.cotrain_openmic:
        # M4 L1: real coarse labels (masked, partial) added to the synthetic training set.
        oidx = load_dataset("openmic", a.openmic_root, check_split_leakage=False)
        lic = openmic_licences(a.openmic_root)
        recs = oidx.split("train")
        if a.cotrain_licence == "commercial":
            recs = [r for r in recs if commercial_ok(lic.get(r.item_id, ""))]
        oids, oy, om = label_matrices(label_items(recs, "mix"), tax)
        ocids, OX = S.load_mel_cache(a.cache / "mels" / "openmic_train_mix.npz")
        opos = {i: k for k, i in enumerate(ocids)}
        keep = [k for k, i in enumerate(oids) if i in opos]
        orow = _Rows(OX, [opos[oids[k]] for k in keep])
        X = _Concat([X, orow])
        Y, Mk = np.concatenate([Y, oy[keep]]), np.concatenate([Mk, om[keep]])
        cot = {"dataset": "openmic", "split": "train", "licence": a.cotrain_licence,
               "n": len(keep)}
        print(f"co-training with {len(keep)} OpenMIC train clips (licence={a.cotrain_licence})")
    cfg = S.ScratchConfig(seed=a.seed, epochs=a.epochs)
    model, hist = S.train(X, Y, Mk, Xv, Yv, Mv, cfg)
    meta = {"model": "scratch_cnn", "n_params": S.n_params(model), "backbone": None,
            "backbone_license": "none (random init)", "commercial_clean": a.commercial_only,
            "train": {"dataset": "synthetic", "n_train": len(X), "n_val": len(Xv),
                      "commercial_only": a.commercial_only},
            "best_val_map": max(h["val_map"] for h in hist), "epochs_run": len(hist),
            "cotrain": cot}
    a.out.mkdir(parents=True, exist_ok=True)
    tag = (f"_{a.mel_prefix}" + (f"_n{a.train_limit}" if a.train_limit else "")
           + (f"_cot{a.cotrain_licence}" if a.cotrain_openmic else ""))
    S.save(model, a.out / f"scratch_cnn{tag}.pt", cfg, hist, meta)
    from .embed import _device
    for spec in a.eval:
        ds, split = spec.split(":")
        f = a.cache / "mels" / f"{ds}_{split}_mix.npz"
        if not (f.exists() or f.with_suffix(".npy").exists()):
            print(f"skip {spec}: no mel cache {f.name}")
            continue
        ids, Xe = S.load_mel_cache(f)
        ids = [i for i in ids if i]
        scores = S.predict(model, Xe, _device())
        p = save_predictions(a.out / f"scratch{tag}_{ds}_{split}_mix.npz", ids, nodes, scores,
                             tax.version, dataset=ds, meta=meta)
        print(f"wrote {p} ({len(ids)} items; best val mAP {meta['best_val_map']:.3f}, "
              f"{meta['n_params']/1e6:.2f}M params)")
    return 0


def cmd_calibrate(a) -> int:
    """C1: fit per-level temperatures on one predictions file (+ its labels), apply them
    to another predictions file."""
    import json as _json

    from ..eval.predictions import load_predictions
    from .calibrate import apply_temperatures, fit_temperatures
    tax = Taxonomy.load()
    opts = {"check_split_leakage": False} if a.fit_dataset == "openmic" else {}
    fit_idx = load_dataset(a.fit_dataset, a.fit_root, **opts)
    fp = load_predictions(a.fit_predictions)
    items = [it for it in label_items(fit_idx.split(a.fit_split), "mix") if it[0] in set(fp.item_ids)]
    ids, y, m = label_matrices(items, tax)
    pos = {i: k for k, i in enumerate(fp.item_ids)}
    col = {n: k for k, n in enumerate(fp.nodes)}
    order = [col[n] for n in tax.nodes]
    S = fp.scores[[pos[i] for i in ids]][:, order]
    # Per-level temperatures do not commute with the harness's max-propagation (parent =
    # max(children)), so they can change rankings (seen in M3: mAP dropped). A single
    # global temperature is monotone for every node and commutes with the max.
    levels = [0 if a.global_temperature else tax.level(n) for n in tax.nodes]
    trained = None
    if a.trained_only:
        # Only nodes the model was trained on carry meaningful scores; fitting on the
        # others drags temperatures to the bound (seen in M3: CLAP level-1 T -> 20).
        from ..synth.sources import load_registry
        reg = load_registry(taxonomy=tax)
        trained = set(tax.close_upward(list(reg.v1_leaves) + ["cymbals"]))
        m = m & np.array([n in trained for n in tax.nodes])[None, :]
    temps = fit_temperatures(S, y, m, levels)
    ap = load_predictions(a.apply)
    acol = [{n: k for k, n in enumerate(ap.nodes)}[n] for n in tax.nodes]
    cal = apply_temperatures(ap.scores[:, acol], levels, temps)
    if trained is not None:                       # untrained nodes keep raw scores
        keep = np.array([n not in trained for n in tax.nodes])
        cal[:, keep] = ap.scores[:, acol][:, keep]
    meta = dict(ap.meta)
    meta["calibration"] = {"method": "temperature_global" if a.global_temperature else "temperature_per_level",
                           "trained_only": a.trained_only,
                           "fit_dataset": a.fit_dataset,
                           "fit_split": a.fit_split, "n_fit": len(ids),
                           "temperatures": {str(k): v for k, v in temps.items()}}
    p = save_predictions(a.out, ap.item_ids, list(tax.nodes), cal, tax.version,
                         dataset=ap.dataset, meta=meta)
    print(f"wrote {p}; T per level: " + _json.dumps({k: round(v["T"], 3) for k, v in temps.items()}))
    return 0


def cmd_lora(a) -> int:
    """Rung R3: LoRA + layer-weighted head, trained on synthetic audio end to end."""
    from . import lora as L
    from .embed import _device
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    idx = load_dataset("synthetic", a.train_root, commercial_only=a.commercial_only)

    def xy(split, limit=None):
        recs = idx.split(split)[:limit] if limit else idx.split(split)
        ids, y, m = label_matrices(label_items(recs, "mix"), tax)
        by = {r.item_id: r.audio["mix"] for r in recs}
        return [by[i] for i in ids], y, m

    P, Y, Mk = xy("train", a.limit)
    Pv, Yv, Mv = xy("val", a.limit and max(8, a.limit // 4))
    cfg = L.LoraConfigM3(seed=a.seed, epochs=a.epochs, rank=a.rank)
    dev = a.device or _device()
    bb = L.Backbone(a.backbone, dev)
    a.out.mkdir(parents=True, exist_ok=True)
    head, hist, n_lora = L.train(bb, P, Y, Mk, Pv, Yv, Mv, cfg, ckpt=a.out / f"{a.backbone}_lora_r{a.rank}.pt")
    meta = {"model": f"{a.backbone}+lora_r{a.rank}", "n_lora_params": n_lora,
            "backbone_license": "CC-BY-NC-4.0" if a.backbone == "mert95m" else "Apache-2.0",
            "commercial_clean": a.backbone == "clap" and a.commercial_only,
            "best_val_map": max(h["val_map"] for h in hist), "history": hist,
            "train": {"dataset": "synthetic", "n_train": len(P), "n_val": len(Pv)}}
    for spec in a.eval:
        ds, split = spec.split(":")
        root = a.train_root if ds == "synthetic" else getattr(a, f"{ds}_root")
        if root is None:
            continue
        eidx = idx if ds == "synthetic" else load_dataset(
            ds, root, **({"check_split_leakage": False} if ds == "openmic" else {}))
        recs = [r for r in eidx.split(split) if "mix" in r.audio][: a.limit or None]
        scores = L.predict(bb, head, [r.audio["mix"] for r in recs])
        p = save_predictions(a.out / f"{a.backbone}_lora_r{a.rank}_{ds}_{split}_mix.npz",
                             [r.item_id for r in recs], nodes, scores, tax.version, dataset=ds, meta=meta)
        print(f"wrote {p} ({len(recs)} items)")
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
    q.add_argument("--hidden", type=int, default=0, help="MLP head width (0 = linear probe)")
    q.add_argument("--lr", type=float, default=0.0)
    q.add_argument("--out", required=True, type=Path)
    q.set_defaults(fn=cmd_probe)
    mm = sub.add_parser("mels", help="cache log-mel spectrograms for the from-scratch CNN")
    mm.add_argument("--dataset", required=True)
    mm.add_argument("--root", required=True, type=Path)
    mm.add_argument("--split", default="all")
    mm.add_argument("--unit", default="mix", choices=("mix", "stem"))
    mm.add_argument("--cache", required=True, type=Path)
    mm.add_argument("--workers", type=int, default=6)
    mm.add_argument("--prefix", help="cache name prefix (default: dataset name)")
    mm.set_defaults(fn=cmd_mels)
    sc = sub.add_parser("scratch", help="M3 R5: train the from-scratch CNN (A2)")
    sc.add_argument("--train-root", required=True, type=Path)
    sc.add_argument("--cache", required=True, type=Path)
    sc.add_argument("--eval", nargs="*", default=["synthetic:test"])
    sc.add_argument("--epochs", type=int, default=40)
    sc.add_argument("--mel-prefix", default="synthetic", help="mel cache prefix, e.g. synthetic_v2")
    sc.add_argument("--train-limit", type=int, help="use a nested random subset of N train clips")
    sc.add_argument("--cotrain-openmic", action="store_true", help="M4 L1: add OpenMIC train (real, coarse)")
    sc.add_argument("--cotrain-licence", default="all", choices=("all", "commercial"))
    sc.add_argument("--openmic-root", type=Path)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--commercial-only", action="store_true")
    sc.add_argument("--out", required=True, type=Path)
    sc.set_defaults(fn=cmd_scratch)
    lr = sub.add_parser("lora", help="M3 R3: LoRA fine-tune + head on synthetic audio")
    lr.add_argument("--backbone", required=True, choices=("mert95m", "clap"))
    lr.add_argument("--train-root", required=True, type=Path)
    lr.add_argument("--eval", nargs="*", default=["synthetic:test"])
    lr.add_argument("--openmic-root", type=Path)
    lr.add_argument("--epochs", type=int, default=3)
    lr.add_argument("--rank", type=int, default=8)
    lr.add_argument("--seed", type=int, default=0)
    lr.add_argument("--limit", type=int, help="smoke test: use only N train clips")
    lr.add_argument("--device")
    lr.add_argument("--commercial-only", action="store_true")
    lr.add_argument("--out", required=True, type=Path)
    lr.set_defaults(fn=cmd_lora)
    cb = sub.add_parser("calibrate", help="M3 C1: per-level temperature scaling")
    cb.add_argument("--fit-predictions", required=True, type=Path)
    cb.add_argument("--fit-dataset", required=True)
    cb.add_argument("--fit-root", required=True, type=Path)
    cb.add_argument("--fit-split", required=True)
    cb.add_argument("--apply", required=True, type=Path)
    cb.add_argument("--out", required=True, type=Path)
    cb.add_argument("--global-temperature", action="store_true",
                    help="one temperature for all levels (commutes with max-propagation)")
    cb.add_argument("--trained-only", action="store_true",
                    help="fit/apply only on nodes covered by the v1 training leaves")
    cb.set_defaults(fn=cmd_calibrate)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
