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
    if a.dataset == "moisesdb":
        opts["window_start"] = a.moisesdb_window
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
                         {"check_split_leakage": False} if ds == "openmic" else
                         {"window_start": a.moisesdb_window} if ds == "moisesdb" else {}))
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
    opts = ({"check_split_leakage": False} if a.dataset == "openmic" else
            {"window_start": a.moisesdb_window} if a.dataset == "moisesdb" else {})
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
    cfg = S.ScratchConfig(seed=a.seed, epochs=a.epochs, gain_db=a.gain_db)
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
                             tax.version, dataset="synthetic" if ds.startswith("synthetic") else ds,
                             meta=meta)
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


def cmd_scratch_predict(a) -> int:
    """Score an existing from-scratch model on other mel caches (e.g. leave-production-out:
    a v1-production model on v2-production renders)."""
    import json as _json

    from . import scratch as S
    from .embed import _device
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    dev = _device()
    model = S.load_model(a.model, len(nodes), dev)
    meta = _json.loads(a.model.with_suffix(".json").read_text())
    meta = {k: meta[k] for k in ("n_params", "cfg") if k in meta} | {"model": "scratch_cnn",
                                                                      "model_file": a.model.name}
    for spec in a.eval:
        prefix, split = spec.split(":")
        ids, Xe = S.load_mel_cache(a.cache / "mels" / f"{prefix}_{split}_mix.npz")
        scores = S.predict(model, Xe, dev)
        ds = ("moisesdb" if prefix.startswith("moisesdb") else
              prefix if prefix in ("openmic", "golden") or prefix.startswith("idmt") else "synthetic")
        out = a.out / f"{a.model.stem}__{prefix}_{split}_mix.npz"
        save_predictions(out, ids, nodes, scores, tax.version, dataset=ds, meta=meta)
        print(f"wrote {out} ({len(ids)} items)")
    return 0


def cmd_panns_predict(a) -> int:
    """M5: score the app's incumbent tagger (PANNs SED, exactly as `pipeline.tagging` uses it:
    framewise -> 95th percentile per OpenMIC-20 label) so a model swap is judged against it
    on the same clips. Labels map to taxonomy nodes via the OpenMIC projection."""
    import time

    import librosa

    from ...pipeline import tagging
    from ..datasets.openmic import class_to_node
    tax = Taxonomy.load()
    c2n = class_to_node(tax)
    labels = sorted(c2n)
    opts = ({"check_split_leakage": False} if a.dataset == "openmic" else   # same as `mels`
            {"window_start": a.moisesdb_window} if a.dataset == "moisesdb" else {})
    items = _audio_items(load_dataset(a.dataset, a.root, **opts).split(a.split), "mix")
    scores = np.zeros((len(items), len(labels)), np.float32)
    t0 = time.time()
    for k, (_, path) in enumerate(items):
        y, _ = librosa.load(path, sr=32000, mono=True)
        per, _ = tagging.panns_framewise(y)
        scores[k] = [float(np.percentile(per[l], 95)) if l in per else 0.0 for l in labels]
    ds = "moisesdb" if a.dataset == "moisesdb" else a.dataset
    meta = {"model": "panns-sed", "version": "cnn14", "aggregate": "p95 framewise",
            "latency_s": round((time.time() - t0) / max(1, len(items)), 3)}
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"panns__{a.prefix or a.dataset}_{a.split}_mix.npz"
    save_predictions(out, [i for i, _ in items], [c2n[l] for l in labels], scores, tax.version,
                     dataset=ds, meta=meta)
    print(f"wrote {out} ({len(items)} items)")
    return 0


def cmd_panns_calibrate(a) -> int:
    """M5: per-class Platt scaling for the incumbent PANNs tagger, fit on its OpenMIC-train
    scores (`panns-predict --split train`; real, never trained on). Writes the JSON the app
    applies framewise in `pipeline.tagging`."""
    import datetime as dt
    import json as _json

    from ..datasets.openmic import class_to_node
    from ..eval.predictions import load_predictions
    from .calibrate import fit_platt
    tax = Taxonomy.load()
    node_of = class_to_node(tax)
    label_of = {n: l for l, n in node_of.items()}
    pred = load_predictions(a.predictions)
    opts = {"check_split_leakage": False}
    items = label_items(load_dataset("openmic", a.root, **opts).split("train"), "mix")
    ids, y, m = label_matrices(items, tax)
    pos = {i: k for k, i in enumerate(pred.item_ids)}
    keep = [k for k, i in enumerate(ids) if i in pos]
    rows = [pos[ids[k]] for k in keep]
    ix = tax.index()
    cols = [ix[n] for n in pred.nodes]
    fits = fit_platt(pred.scores[rows], y[keep][:, cols], m[keep][:, cols])
    doc = {"format": "disstruments.panns_calibration/v1",
           "model": "panns-sed cnn14 (Cnn14_DecisionLevelMax)",
           "map": "p' = sigmoid(a * logit(p) + b), applied to framewise probs",
           "fit": {"dataset": "openmic", "split": "train", "n_clips": len(keep),
                   "statistic": "p95 over frames per 10 s clip (as the app aggregates)",
                   "predictions": a.predictions.name,
                   "created": dt.date.today().isoformat()},
           "classes": {label_of[n]: {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in f.items()} for n, f in zip(pred.nodes, fits)}}
    a.out.write_text(_json.dumps(doc, indent=1) + "\n")
    for lab, f in doc["classes"].items():
        print(f"{lab:18s} a={f['a']:6.3f} b={f['b']:+7.3f} n_pos={f['n_pos']:5d} "
              + (f"nll {f['nll_before']:.3f} -> {f['nll_after']:.3f}" if f["fitted"] else "(identity)"))
    print(f"wrote {a.out} ({len(keep)} clips)")
    return 0


def cmd_export_onnx(a) -> int:
    """M5: checkpoint -> ONNX + sidecar (see `export.py`), with a global temperature fit on
    real held-out clips and an ONNX-vs-torch parity check on real mels."""
    import json as _json
    import shutil

    from . import export as EX
    from . import scratch as S
    from .embed import _device
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    manifest = _json.loads(a.manifest.read_text())
    ckpt_meta = _json.loads(a.model.with_suffix(".json").read_text())
    trained = EX.trained_nodes(manifest, tax)
    tmp = a.out.with_name(a.out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model = S.load_model(a.model, len(nodes), "cpu")
    onnx_path = EX.to_onnx(model, tmp / EX.ONNX_NAME)

    # temperature: OpenMIC train (real, never trained on), trained nodes only. A licence-clean
    # model is calibrated on the commercially licensed clips only (most OpenMIC clips are
    # NC/ND); the all-clips T is recorded next to it for comparison.
    clean = bool(ckpt_meta.get("commercial_clean", False))
    opts = {"check_split_leakage": False}
    recs = load_dataset("openmic", a.calib_root, **opts).split("train")
    lic = openmic_licences(a.calib_root)
    cids, Xc = S.load_mel_cache(a.calib_cache / "mels" / "openmic_train_mix.npz")
    pos = {i: k for k, i in enumerate(cids)}
    items = [it for it in label_items(recs, "mix") if it[0] in pos]
    ids, y, m = label_matrices(items, tax)
    rows = np.array([pos[i] for i in ids])
    Xr = np.asarray(Xc[np.sort(rows)])[np.argsort(np.argsort(rows))]
    scores = S.predict(S.load_model(a.model, len(nodes), _device()), Xr, _device())
    scores = tax.max_propagate(scores, nodes)       # fit T on what is served (and evaluated)
    ok = np.array([commercial_ok(lic.get(i, "")) for i in ids])
    t_all = EX.fit_global_temperature(scores, y, m, nodes, set(trained))
    if clean:
        temp = EX.fit_global_temperature(scores[ok], y[ok], m[ok], nodes, set(trained))
        temp |= {"fit_dataset": "openmic", "fit_split": "train", "fit_licence": "commercial",
                 "n_fit": int(ok.sum()), "T_all_clips": t_all["T"]}
    else:
        temp = t_all | {"fit_dataset": "openmic", "fit_split": "train", "fit_licence": "all",
                        "n_fit": len(ids)}
    diff = EX.parity(model, onnx_path, Xr[:64].astype(np.float32))
    if diff > 1e-4:
        raise SystemExit(f"ONNX parity failed: max |dp| = {diff:.2e}")
    doc = EX.sidecar(tax=tax, onnx_path=onnx_path, ckpt_meta=ckpt_meta, manifest=manifest,
                     trained=trained, temperature=temp, parity_max=diff, source=str(a.model))
    (tmp / EX.SIDECAR_NAME).write_text(_json.dumps(doc, indent=1))
    shutil.rmtree(a.out, ignore_errors=True)
    tmp.rename(a.out)
    print(f"wrote {a.out}: sha256 {doc['sha256'][:12]}, T={temp['T']:.3f} (n={temp['n_fit']}, {temp['fit_licence']} clips), "
          f"onnx parity {diff:.1e}")
    return 0


def cmd_shortcut(a) -> int:
    """M4 rigor: non-timbre shortcut baseline (see ml/eval/shortcut.py)."""
    from ..eval import shortcut as SC
    tax = Taxonomy.load()
    nodes = list(tax.nodes)
    idx = load_dataset("synthetic", a.root)
    cache = a.cache / "shortcut" / f"{a.root.name}.npz"
    if cache.exists():
        z = np.load(cache); fid, F = z["ids"].tolist(), z["X"]
    else:
        items = _audio_items(idx.records, "mix")
        fid, F = SC.build(items, workers=a.workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(fid), X=F)
    pos = {i: k for k, i in enumerate(fid)}

    def xy(split):
        ids, y, m = label_matrices(label_items(idx.split(split), "mix"), tax)
        return ids, F[[pos[i] for i in ids]][:, None, :], y, m

    _, X, Y, Mk = xy("train")
    _, Xv, Yv, Mv = xy("val")
    probe = LinearProbe(1, X.shape[2], len(nodes), ProbeConfig(hidden=a.hidden, lr=1e-2 if not a.hidden else 1e-3)).fit(X, Y, Mk, Xv, Yv, Mv)
    for split in a.splits:
        ids, Xt, _, _ = xy(split)
        p = save_predictions(a.out / f"shortcut_{a.root.name}_{split}.npz", ids, nodes,
                             probe.predict(Xt), tax.version, dataset="synthetic",
                             meta={"model": "shortcut_nontimbre", "features": SC.N_FEATS})
        print(f"wrote {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.models")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--moisesdb-window", type=lambda v: v if v == "all" else float(v), default=20.0, help="moisesdb clip window start (s), or 'all'")
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
    q.add_argument("--moisesdb-window", type=lambda v: v if v == "all" else float(v), default=20.0, help="moisesdb clip window start (s), or 'all'")
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
    q.add_argument("--moisesdb-root", type=Path)
    q.add_argument("--golden-root", type=Path)
    for n in ("idmt-bass", "idmt-bass-tracks", "idmt-fx"):
        q.add_argument(f"--{n}-root", type=Path)
    q.add_argument("--commercial-only", action="store_true")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--hidden", type=int, default=0, help="MLP head width (0 = linear probe)")
    q.add_argument("--lr", type=float, default=0.0)
    q.add_argument("--out", required=True, type=Path)
    q.set_defaults(fn=cmd_probe)
    mm = sub.add_parser("mels", help="cache log-mel spectrograms for the from-scratch CNN")
    mm.add_argument("--moisesdb-window", type=lambda v: v if v == "all" else float(v), default=20.0, help="moisesdb clip window start (s), or 'all'")
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
    sc.add_argument("--gain-db", type=float, default=0.0, help="random input gain, +/- dB per clip")
    sc.add_argument("--commercial-only", action="store_true")
    sc.add_argument("--out", required=True, type=Path)
    sc.set_defaults(fn=cmd_scratch)
    sh = sub.add_parser("shortcut", help="M4 rigor: non-timbre shortcut baseline")
    sh.add_argument("--root", required=True, type=Path)
    sh.add_argument("--cache", required=True, type=Path)
    sh.add_argument("--splits", nargs="+", default=["test"])
    sh.add_argument("--hidden", type=int, default=256)
    sh.add_argument("--workers", type=int, default=4)
    sh.add_argument("--out", required=True, type=Path)
    sh.set_defaults(fn=cmd_shortcut)
    sp = sub.add_parser("scratch-predict", help="score a saved from-scratch model on mel caches")
    sp.add_argument("--model", required=True, type=Path)
    sp.add_argument("--cache", required=True, type=Path)
    sp.add_argument("--eval", nargs="+", required=True, help="mel-prefix:split pairs")
    sp.add_argument("--out", required=True, type=Path)
    sp.set_defaults(fn=cmd_scratch_predict)
    pp = sub.add_parser("panns-predict", help="M5: score the incumbent PANNs tagger on a dataset")
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--root", required=True, type=Path)
    pp.add_argument("--split", default="test")
    pp.add_argument("--moisesdb-window", type=lambda v: v if v == "all" else float(v), default=20.0)
    pp.add_argument("--prefix", help="output name prefix (default: dataset name)")
    pp.add_argument("--out", required=True, type=Path)
    pp.set_defaults(fn=cmd_panns_predict)
    pc = sub.add_parser("panns-calibrate", help="M5: per-class Platt scaling for PANNs (OpenMIC train)")
    pc.add_argument("--predictions", required=True, type=Path, help="panns-predict output, OpenMIC train")
    pc.add_argument("--root", required=True, type=Path, help="OpenMIC root (labels)")
    pc.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "pipeline" / "panns_calibration.json")
    pc.set_defaults(fn=cmd_panns_calibrate)
    ex = sub.add_parser("export-onnx", help="M5: package a scratch CNN for the app (ONNX + sidecar)")
    ex.add_argument("--model", required=True, type=Path)
    ex.add_argument("--manifest", required=True, type=Path, help="render manifest the model trained on")
    ex.add_argument("--calib-cache", required=True, type=Path, help="cache dir with mels/openmic_train_mix")
    ex.add_argument("--calib-root", required=True, type=Path, help="OpenMIC root (labels)")
    ex.add_argument("--out", required=True, type=Path)
    ex.set_defaults(fn=cmd_export_onnx)
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
