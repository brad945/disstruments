"""Eval harness CLI (PRD F25 / §8.3, ML_ENGINEERING §6).

    python -m disstruments.ml.eval taxonomy
    python -m disstruments.ml.eval baseline --kind prior --dataset medleydb --root M \
        --split test --out preds.json
    python -m disstruments.ml.eval run --dataset medleydb --root M --split test \
        --predictions preds.json [--name exp1]
    python -m disstruments.ml.eval compare CANDIDATE_RUN_DIR INCUMBENT_RUN_DIR
    python -m disstruments.ml.eval coverage --dataset medleydb --root M
    python -m disstruments.ml.eval make-split --root M --out medleydb_split_vN.json

MedleyDB uses the pinned split (`datasets/medleydb_split_v1.json`) unless
`--generated-split`; a loaded track set that differs from the pin fails loudly.

`run` archives `metrics.json` + `summary.md` + `arrays.npz` (per-item labels, masks and
scores, for `compare`'s paired bootstrap) under
`<runs-dir>/<UTC timestamp>_<name>/` (default runs dir: `$DISS_DATA_DIR/eval_runs`, i.e.
~/.disstruments/eval_runs) with provenance: taxonomy version + sha, mappings sha, git
SHA (+dirty flag), predictions sha256, and a config hash over every setting that
changes the numbers (paths excluded, so the same eval hashes the same on any machine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...config import settings
from ..datasets import DATASETS, load_dataset
from ..datasets.base import MAPPINGS_PATH, artist_leakage, split_hash
from ..taxonomy import get_taxonomy
from .audit import coverage, missingness
from .harness import (DEFAULT_SECONDARIES, EvalArrays, baseline_predictions, default_primary,
                      evaluate, label_items, paired_bootstrap, prior_scores, promotion_gate,
                      summarize)
from .predictions import load_predictions, save_predictions

REPORT_SCHEMA = "disstruments.eval_report/v1"


def _git() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None}


def _loader_opts(a: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {"strict": not a.non_strict}
    if a.dataset == "medleydb":
        opts.update(audio_root=a.audio_root, seed=a.seed, split_file=a.split_file,
                    generated_split=a.generated_split, allow_partial_split=a.allow_partial_split)
    elif a.dataset == "openmic":
        opts.update(val_fraction=a.val_fraction, seed=a.seed,
                    check_split_leakage=not a.no_split_leakage_check)
    elif a.dataset == "slakh":
        opts.update(split_mode=a.slakh_split,
                    min_loudness_lufs=float("-inf") if a.min_loudness_lufs.lower() == "none"
                    else float(a.min_loudness_lufs))
    return opts


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", required=True, choices=DATASETS)
    g.add_argument("--root", required=True, type=Path,
                   help="medleydb: metadata dir; openmic: extracted openmic-2018/; slakh: slakh2100 root")
    g.add_argument("--split", default="test", help="train | val | test | all")
    g.add_argument("--audio-root", type=Path, help="medleydb: MEDLEYDB_PATH/Audio (optional)")
    g.add_argument("--split-file", type=Path,
                   help="medleydb: {track_id: split} JSON (default: packaged medleydb_split_v1.json)")
    g.add_argument("--generated-split", action="store_true",
                   help="medleydb: ignore the pin; seeded artist-disjoint split over the loaded tracks")
    g.add_argument("--allow-partial-split", action="store_true",
                   help="medleydb: allow loading a subset of the pinned tracks")
    g.add_argument("--no-split-leakage-check", action="store_true",
                   help="openmic: do not raise if the official split shares an artist")
    g.add_argument("--min-loudness-lufs", default="-60",
                   help="slakh: stems quieter than this are masked (unknown); 'none' disables")
    g.add_argument("--seed", type=int, default=0, help="split seed (medleydb generated, openmic val)")
    g.add_argument("--val-fraction", type=float, default=0.0, help="openmic: artist-disjoint val carve-out")
    g.add_argument("--slakh-split", default="redux", choices=("redux", "split2", "orig"))
    g.add_argument("--non-strict", action="store_true",
                   help="record unmapped source labels instead of failing (they mask the item)")
    g.add_argument("--unit", default="mix", choices=("mix", "stem"))
    g.add_argument("--exclude-bleed", action="store_true",
                   help="unit=stem: drop stems of tracks flagged has_bleed (MedleyDB)")


def cmd_taxonomy(a: argparse.Namespace) -> int:
    tax = get_taxonomy()
    print(f"taxonomy {tax.version}: {len(tax)} nodes, {len(tax.leaves())} leaves "
          f"(levels: {[len(tax.nodes_at_level(i)) for i in (1, 2, 3)]}); * = planned source")
    print(tax.render())
    return 0


def cmd_baseline(a: argparse.Namespace) -> int:
    tax = get_taxonomy()
    idx = load_dataset(a.dataset, a.root, **_loader_opts(a))
    items = label_items(idx.split(a.split), a.unit, a.exclude_bleed)
    ids = [i for i, _, _ in items]
    train = idx.split(a.fit_split) if a.kind == "prior" else None
    scores = baseline_predictions(a.kind, ids, tax, train_records=train, seed=a.seed, unit=a.unit)
    meta = {"model": f"baseline-{a.kind}", "fit_split": a.fit_split if a.kind == "prior" else None,
            "seed": a.seed}
    out = save_predictions(a.out, ids, tax.nodes, scores, tax.version, a.dataset, meta)
    print(f"wrote {out} ({len(ids)} items x {len(tax)} nodes, baseline={a.kind})")
    return 0


def run_eval(a: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    tax = get_taxonomy()
    idx = load_dataset(a.dataset, a.root, **_loader_opts(a))
    records = idx.split(a.split)
    if not records:
        raise SystemExit(f"no records in split {a.split!r} (have {idx.split_counts()})")
    preds = load_predictions(a.predictions)
    if preds.dataset and preds.dataset != a.dataset:
        raise SystemExit(f"predictions are for {preds.dataset!r}, not {a.dataset!r}")
    name = a.name or Path(a.predictions).stem
    if not _SAFE_NAME.match(name) or ".." in name:
        raise SystemExit(f"--name {name!r}: use letters, digits, '.', '_', '-' only (no paths)")
    result, arrays = evaluate(records, preds, tax, threshold=a.threshold, n_bins=a.bins,
                              propagate=not a.no_propagate, allow_missing=a.allow_missing,
                              unit=a.unit, exclude_bleed=a.exclude_bleed, return_arrays=True)
    result["calibration"]["prior_reference"] = _prior_reference(idx, arrays, tax, a)
    config = {"dataset": a.dataset, "split": a.split, "unit": a.unit, "loader": idx.config,
              "threshold": a.threshold, "bins": a.bins, "propagate": not a.no_propagate,
              "allow_missing": a.allow_missing, "exclude_bleed": a.exclude_bleed,
              "taxonomy_version": tax.version, "taxonomy_sha256": tax.sha256}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    leak = artist_leakage(idx.records)
    now = datetime.now(timezone.utc)
    report = {
        "schema": REPORT_SCHEMA, "name": name, "created_utc": now.isoformat(timespec="seconds"),
        "provenance": {"taxonomy_version": tax.version, "taxonomy_sha256": tax.sha256,
                       "mappings_sha256": hashlib.sha256(MAPPINGS_PATH.read_bytes()).hexdigest(),
                       "git": _git(), "predictions_path": str(Path(a.predictions).resolve()),
                       "predictions_sha256": preds.sha256, "config_hash": config_hash,
                       "python": platform.python_version(), "numpy": np.__version__},
        "config": config,
        "dataset": {"root": str(Path(a.root).resolve()), "split_counts": idx.split_counts(),
                    "split_hash": split_hash(idx.records), "stats": idx.stats,
                    "n_artists_in_split": len({r.artist for r in records if r.artist is not None}),
                    "artist_leakage": {"n_artists": len(leak), "examples": dict(list(leak.items())[:10])}},
        **result,
    }
    runs = Path(a.runs_dir) if a.runs_dir else settings.data_dir / "eval_runs"
    out = runs / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}"
    suffix = 1
    while out.exists():
        out = runs / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}_{suffix}"
        suffix += 1
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "summary.md").write_text(summary_markdown(report))
    arrays.save(out / "arrays.npz")
    return report, out


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _prior_reference(idx, arrays: EvalArrays, tax, a: argparse.Namespace) -> dict[str, Any] | None:
    """Calibration of the label-prior baseline on the SAME items: the floor a model's
    ECE/Brier should be read against (a constant prior is often 'well calibrated')."""
    train = idx.split("train")
    if a.split in ("train", "all") or not train:
        return None
    try:
        prior = prior_scores(train, tax, unit=a.unit)
    except ValueError:
        return None
    pos = {n: j for j, n in enumerate(tax.nodes)}
    ref = EvalArrays(arrays.item_ids, arrays.nodes, arrays.y_full, arrays.m_full,
                     np.tile(prior[[pos[n] for n in arrays.nodes]], (len(arrays.item_ids), 1)),
                     arrays.threshold, arrays.n_bins)
    s = summarize(ref, tax)
    return {"fit_split": "train", "ece": s["ece"], "ece_classwise": s["ece_classwise"],
            "brier": s["brier"]}


def summary_markdown(r: dict[str, Any]) -> str:
    def f(x): return "n/a" if x is None else f"{x:.4f}"
    s, cfg = r["summary"], r["config"]
    lines = [f"# Eval: {r['name']}", "",
             f"- dataset `{cfg['dataset']}` split `{cfg['split']}` unit `{cfg['unit']}`; "
             f"{r['items']['n_evaluated']} items; taxonomy {r['provenance']['taxonomy_version']}",
             f"- git `{r['provenance']['git']['sha']}` dirty={r['provenance']['git']['dirty']}; "
             f"config `{r['provenance']['config_hash']}`; predictions sha256 `{r['provenance']['predictions_sha256'][:16]}`",
             "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {f(v)} |" for k, v in s.items()]
    ref = (r.get("calibration") or {}).get("prior_reference")
    if ref:
        lines += ["", f"Calibration reference (label-prior baseline, same items): ece {f(ref['ece'])}, "
                      f"ece_classwise {f(ref['ece_classwise'])}, brier {f(ref['brier'])}."]
    d = r["macro_detail"]
    m3 = d.get("map_leaf_min3")
    if m3:
        lines += ["", f"Leaf mAP companions: map_leaf_min3 {f(s.get('map_leaf_min3'))} over "
                      f"{m3['n_classes']} leaves with >= {m3['min_positives']} observed positives "
                      f"(map_leaf: {d['map_leaf']['n_classes']} leaves); map_micro_leaf "
                      f"{f(s.get('map_micro_leaf'))} (pooled observed leaf pairs)."]
        pos = sorted(((n, v["n_positive"]) for n, v in r["per_node"].items()
                      if v["leaf"] and v["skipped"] is None), key=lambda x: (x[1], x[0]))
        lines += ["", "Scored leaves by observed positives: "
                      + ", ".join(f"{n} {k}" for n, k in pos) + "."]
    lines += ["", f"Skipped classes: mAP {d['map_macro']['n_skipped']} {d['map_macro']['skipped']}, "
                  f"leaf mAP {d['map_leaf']['n_skipped']}, leaf F1 {d['f1_macro_leaf']['n_skipped']}."]
    return "\n".join(lines) + "\n"


def cmd_run(a: argparse.Namespace) -> int:
    report, out = run_eval(a)
    s = report["summary"]
    print(f"run dir: {out}")
    for k in ("map_macro", "map_leaf", "map_leaf_min3", "map_micro_leaf", "f1_macro_leaf", "f1_micro_leaf", "hier_f1", "ece",
              "ece_classwise", "brier", "openmic20_map"):
        v = s.get(k)
        print(f"  {k:<14} {'n/a' if v is None else f'{v:.4f}'}")
    d = report["macro_detail"]["map_macro"]
    print(f"  classes scored {d['n_classes']}, skipped {d['n_skipped']} {d['skipped']}")
    print(f"  map_leaf over {report['macro_detail']['map_leaf']['n_classes']} leaves; "
          f"map_leaf_min3 over {report['macro_detail']['map_leaf_min3']['n_classes']}")
    return 0


def _run_paths(p: Path) -> tuple[Path, Path]:
    """Accept a run dir or its metrics.json; return (metrics.json, arrays.npz)."""
    d = p if p.is_dir() else p.parent
    return d / "metrics.json", d / "arrays.npz"


def cmd_compare(a: argparse.Namespace) -> int:
    (cm, ca), (im, ia) = _run_paths(Path(a.candidate)), _run_paths(Path(a.incumbent))
    cand, inc = json.loads(cm.read_text()), json.loads(im.read_text())
    primary = a.primary or default_primary((cand.get("config") or {}).get("dataset"))
    secondaries = tuple(a.secondary or DEFAULT_SECONDARIES)
    gate = promotion_gate(cand, inc, primary=primary, secondaries=secondaries, tolerance=a.tolerance)
    if all((r.get("config") or {}).get("split") == "test" for r in (cand, inc)):
        print("WARNING: both runs are on split=test. Gate on val; test is for final reporting "
              "(repeated gating on test overfits the test set).", file=sys.stderr)
    if a.no_bootstrap:
        print("WARNING: --no-bootstrap: significance was NOT tested; a 'promote' here rests on "
              "the point estimate only.", file=sys.stderr)
    boot = None
    if gate["comparable"] and not a.no_bootstrap:
        if not (ca.exists() and ia.exists()):
            gate["promote"] = False
            gate["reasons"].append("arrays.npz missing for a run: cannot bootstrap (re-run, "
                                   "or --no-bootstrap to skip the significance requirement)")
        else:
            tax = get_taxonomy()
            boot = paired_bootstrap(EvalArrays.load(ca), EvalArrays.load(ia), tax,
                                    (primary,) + tuple(k for k in secondaries if k != primary),
                                    n_boot=a.n_boot, seed=a.boot_seed)
            gate = promotion_gate(cand, inc, primary=primary, secondaries=secondaries,
                                  tolerance=a.tolerance, bootstrap=boot)
    gate["bootstrap"] = boot
    print(json.dumps(gate, indent=1))
    return 0 if gate["promote"] else 1


def cmd_coverage(a: argparse.Namespace) -> int:
    tax = get_taxonomy()
    idx = load_dataset(a.dataset, a.root, **_loader_opts(a))
    cov = coverage(idx.records, tax, a.dataset)
    miss = missingness(idx.records, tax)
    flagged = {n: v for n, v in miss.items() if v["flagged"]}
    splits = list(cov["summary"]["splits"])
    print(f"{a.dataset}: {cov['summary']}")
    print(f"{'leaf':34s} " + " ".join(f"{s:>6s}" for s in splits) + "  claim")
    for leaf, c in cov["per_leaf"].items():
        claim = a.dataset in tax.info[leaf].data
        print(f"{leaf:34s} " + " ".join(f"{c[s]:6d}" for s in splits) + f"  {'Y' if claim else '-'}")
    print(f"data: claims without examples: {cov['claims_without_examples']}")
    print(f"examples without data: claim: {cov['examples_without_claim']}")
    print("label-dependent missingness flagged: "
          + (", ".join(f"{n} (observed rate {v['rate_observed']:.2f} vs {v['rate_overall']:.2f})"
                       for n, v in flagged.items()) or "none"))
    if a.out:
        a.out.write_text(json.dumps({"coverage": cov, "missingness": miss}, indent=1))
    problems = bool(flagged or cov["claims_without_examples"] or cov["examples_without_claim"])
    return 1 if (a.strict and problems) else 0


def cmd_make_split(a: argparse.Namespace) -> int:
    from ..datasets.medleydb import make_split
    idx = load_dataset("medleydb", a.root, generated_split=True, seed=a.seed,
                       strict=not a.non_strict)
    doc = make_split(idx, seed=a.seed, n_iter=a.n_iter)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; pins are never regenerated in place")
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: {doc['counts']} split_hash {doc['split_hash']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.eval", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("taxonomy", help="print the taxonomy tree").set_defaults(fn=cmd_taxonomy)

    b = sub.add_parser("baseline", help="write trivial-baseline predictions")
    _add_dataset_args(b)
    b.add_argument("--kind", default="prior", choices=("prior", "random"))
    b.add_argument("--fit-split", default="train", help="prior: split to estimate label rates on")
    b.add_argument("--out", required=True, type=Path)
    b.set_defaults(fn=cmd_baseline)

    r = sub.add_parser("run", help="score a predictions file and archive the report")
    _add_dataset_args(r)
    r.add_argument("--predictions", required=True, type=Path)
    r.add_argument("--name")
    r.add_argument("--runs-dir", type=Path)
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--bins", type=int, default=15)
    r.add_argument("--no-propagate", action="store_true", help="score raw columns (no max-propagation)")
    r.add_argument("--allow-missing", action="store_true", help="evaluate items that have predictions only")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="promotion gate: candidate vs incumbent run (exit 0 = promote)")
    c.add_argument("candidate", type=Path, help="run dir (or its metrics.json)")
    c.add_argument("incumbent", type=Path, help="run dir (or its metrics.json)")
    c.add_argument("--primary", help="default: openmic20_map on openmic, map_leaf otherwise")
    c.add_argument("--secondary", action="append", help="repeatable; default: " + ", ".join(DEFAULT_SECONDARIES))
    c.add_argument("--tolerance", type=float, default=None,
                   help="absolute max regression for every secondary (default: per metric, "
                        "2%% relative for mAP/F1, 0.005 absolute for ECE/Brier)")
    c.add_argument("--n-boot", type=int, default=1000, help="paired bootstrap resamples")
    c.add_argument("--boot-seed", type=int, default=0)
    c.add_argument("--no-bootstrap", action="store_true",
                   help="skip the significance requirement (primary CI must exclude 0)")
    c.set_defaults(fn=cmd_compare)

    v = sub.add_parser("coverage", help="real positives per leaf per split + label audits")
    _add_dataset_args(v)
    v.add_argument("--out", type=Path, help="write the full audit as JSON")
    v.add_argument("--strict", action="store_true", help="exit 1 if any audit problem")
    v.set_defaults(fn=cmd_coverage)

    ms = sub.add_parser("make-split", help="medleydb: build a new pinnable stratified split")
    ms.add_argument("--root", required=True, type=Path)
    ms.add_argument("--out", required=True, type=Path)
    ms.add_argument("--seed", type=int, default=0)
    ms.add_argument("--n-iter", type=int, default=20000)
    ms.add_argument("--non-strict", action="store_true")
    ms.set_defaults(fn=cmd_make_split)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
