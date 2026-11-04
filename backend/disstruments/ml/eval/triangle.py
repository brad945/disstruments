"""Triangle experiment scoring (docs/debriefs/triangle-plan.md).

Reads archived `eval run` dirs (arrays.npz) and reports, per arm, the subset mean AP over a
node list fixed in the plan, with artist-bootstrap CIs and paired deltas drawn exactly as
`harness.paired_bootstrap` (the same resamples as `eval compare` and `eval.adopt`).

Node lists:
- `trained9`: the 9 OpenMIC classes the v1 synthetic leaves cover (the M3 headline metric).
- `trained9-moises`: those of the 9 that pooled MoisesDB can score (>= 3 positive and >= 3
  negative clips).
- `untrained11`: the other 11 OpenMIC classes (only real-trained arms learn them).
- `leaf`: leaves of the v1 trained set that the test set can score (fine-grained retention).

Runs are named `<arm>_s<seed>`; seeds of one arm are averaged (mean and min/max), and the
paired deltas are per seed, reported as the mean delta with each seed's CI.

    python -m disstruments.ml.eval.triangle --nodes trained9 --runs-dir RUNS \
        --pair c2_cnn:c1_cnn --pair c3_cnn:c2_cnn --out table.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..datasets.openmic import class_to_node
from ..taxonomy import Taxonomy, get_taxonomy
from .adopt import _arrays, _paired, scoreable
from .harness import EvalArrays
from .subsets import bootstrap_ci, per_node, restrict

_RUN = re.compile(r"^\d{8}T\d{6}Z_(?P<name>.+)$")


def v1_trained(tax: Taxonomy) -> set[str]:
    """Nodes the v1 synthetic models were trained on (as `models calibrate --trained-only`)."""
    from ..synth.sources import load_registry
    reg = load_registry(taxonomy=tax)
    return set(tax.close_upward(list(reg.v1_leaves) + ["cymbals"]))


def node_list(kind: str, tax: Taxonomy, a: EvalArrays | None = None) -> list[str]:
    trained = v1_trained(tax)
    om = list(class_to_node(tax).values())
    if kind == "trained9":
        return [n for n in om if n in trained]
    if kind == "untrained11":
        return [n for n in om if n not in trained]
    if kind == "trained9-moises":
        if a is None:
            raise ValueError("trained9-moises needs arrays to check which nodes are scoreable")
        return scoreable(a, tax, [n for n in om if n in trained])
    if kind == "leaf":
        if a is None:
            raise ValueError("leaf needs arrays to check which nodes are scoreable")
        return scoreable(a, tax, [n for n in trained if tax.is_leaf(n)])
    raise ValueError(f"unknown node list {kind!r}")


def subset_map(a: EvalArrays, tax: Taxonomy, nodes: Sequence[str]) -> float:
    pn = per_node(restrict(a, nodes), tax)
    return float(np.mean([pn[n]["ap"] for n in nodes]))


def latest_runs(runs_dir: Path) -> dict[str, Path]:
    """name -> newest run dir with that name (re-runs supersede earlier ones)."""
    out: dict[str, Path] = {}
    for d in sorted(runs_dir.iterdir()):
        m = _RUN.match(d.name)
        if m and (d / "arrays.npz").exists():
            out[m["name"]] = d
    return out


def arm_of(name: str) -> tuple[str, int | None]:
    m = re.match(r"^(?P<arm>.+)_s(?P<seed>\d+)$", name)
    return (m["arm"], int(m["seed"])) if m else (name, None)


def table(runs: dict[str, Path], kind: str, pairs: Sequence[tuple[str, str]], *,
          tax: Taxonomy, n_boot: int = 1000, boot_seed: int = 0) -> dict[str, Any]:
    arrays = {n: _arrays(p) for n, p in runs.items()}
    ref = next(iter(arrays.values()))
    nodes = node_list(kind, tax, ref)
    for n, a in arrays.items():                    # every arm must score the same items
        if list(a.item_ids) != list(ref.item_ids):
            raise ValueError(f"{n}: different test items from {next(iter(arrays))}")
    by_arm: dict[str, dict[int | None, str]] = {}
    for n in arrays:
        arm, seed = arm_of(n)
        by_arm.setdefault(arm, {})[seed] = n
    arms = {}
    for arm, seeds in sorted(by_arm.items()):
        vals = {s: subset_map(arrays[n], tax, nodes) for s, n in seeds.items()}
        first = arrays[seeds[min(seeds, key=lambda s: -1 if s is None else s)]]
        ci = bootstrap_ci(restrict(first, nodes), tax, ["map_macro"], n_boot=n_boot,
                          seed=boot_seed)["map_macro"]
        v = list(vals.values())
        arms[arm] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)),
                     "n_seeds": len(v), "per_seed": {str(k): x for k, x in vals.items()},
                     "ci_first_seed": ci}
    out_pairs = []
    for cand, inc in pairs:
        if cand not in by_arm or inc not in by_arm:
            out_pairs.append({"pair": f"{cand} - {inc}", "missing": True})
            continue
        shared = sorted(set(by_arm[cand]) & set(by_arm[inc]), key=lambda s: -1 if s is None else s)
        if not shared:                             # e.g. a single-seed reference arm
            shared_pairs = [(s, next(iter(by_arm[inc]))) for s in by_arm[cand]]
        else:
            shared_pairs = [(s, s) for s in shared]
        per = [{"seeds": [sc, si], **_paired(arrays[by_arm[cand][sc]], arrays[by_arm[inc][si]],
                                            tax, nodes, n_boot, boot_seed)}
               for sc, si in shared_pairs]
        out_pairs.append({"pair": f"{cand} - {inc}", "mean_delta": float(np.mean([p["delta"] for p in per])),
                          "all_ci_exclude_0": all(p["ci"][0] > 0 or p["ci"][1] < 0 for p in per),
                          "per_seed": per})
    return {"node_list": kind, "nodes": nodes, "n_items": len(ref.item_ids), "arms": arms,
            "pairs": out_pairs}


def markdown(t: dict[str, Any]) -> str:
    lines = [f"**{t['node_list']}** ({len(t['nodes'])} nodes, {t['n_items']} items)", "",
             "| arm | mean AP | seeds | range | 95% CI (seed 0) |", "|---|---|---|---|---|"]
    for arm, r in t["arms"].items():
        ci = r["ci_first_seed"].get("ci") or [float("nan")] * 2
        lines.append(f"| {arm} | {r['mean']:.3f} | {r['n_seeds']} | {r['min']:.3f}-{r['max']:.3f} "
                     f"| [{ci[0]:.3f}, {ci[1]:.3f}] |")
    if t["pairs"]:
        lines += ["", "| paired delta | mean | per-seed 95% CIs | every CI excludes 0 |", "|---|---|---|---|"]
        for p in t["pairs"]:
            if p.get("missing"):
                lines.append(f"| {p['pair']} | (missing) | | |")
                continue
            cis = "; ".join(f"[{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}]" for q in p["per_seed"])
            lines.append(f"| {p['pair']} | {p['mean_delta']:+.3f} | {cis} | "
                         f"{'yes' if p['all_ci_exclude_0'] else 'no'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m disstruments.ml.eval.triangle")
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--match", default="", help="only run names starting with this prefix")
    ap.add_argument("--strip", default="", help="prefix removed from run names to get arm_sN")
    ap.add_argument("--nodes", required=True,
                    choices=("trained9", "trained9-moises", "untrained11", "leaf"))
    ap.add_argument("--pair", action="append", default=[], help="CAND:INC arm names")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    runs = {n[len(a.strip):]: p for n, p in latest_runs(a.runs_dir).items() if n.startswith(a.match)}
    if not runs:
        print("no runs matched", file=sys.stderr)
        return 1
    t = table(runs, a.nodes, [tuple(p.split(":")) for p in a.pair], tax=get_taxonomy(), n_boot=a.n_boot)
    md = markdown(t)
    print(md)
    if a.out:
        a.out.write_text(json.dumps(t, indent=1))
        a.out.with_suffix(".md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
