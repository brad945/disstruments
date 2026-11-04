"""Pre-registered adoption checks for a new leaf model (docs/debriefs/v8-plan.md).

Two questions, both on archived harness runs (`eval run` dirs with arrays.npz) of the same
test items, paired artist bootstrap (`harness.paired_bootstrap`, same resamples as
`eval compare`):

- `old`  - no forgetting. Nodes both models were trained on and that the test set can score.
  Pass iff (a) the subset mean AP delta is within the harness tolerance (point estimate,
  >= -REL_TOL x incumbent, as `promotion_gate` treats secondaries) and (b) no single node's
  paired CI lies entirely below -node_floor (the M5 F2 rule: "not worse beyond run noise").
- `new`  - learns the new families. Nodes only the candidate was trained on. Pass iff the
  subset mean AP beats BOTH the incumbent and chance (constant scores = prevalence AP) with
  the paired CI's lower bound > 0. Per node it reports the CI vs chance and the normalized
  lift (AP - prevalence) / (1 - prevalence) that the M5 routing rule uses (>= 0.1 to show).

    python -m disstruments.ml.eval.adopt old --candidate RUN --incumbent RUN \
        --cand-manifest v8serve/manifest.json --inc-manifest v7serve/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..taxonomy import Taxonomy, get_taxonomy
from .harness import REL_TOL, EvalArrays, paired_bootstrap
from .subsets import constant_arrays, per_node, restrict

NODE_FLOOR = 0.01       # M4/M5 run-to-run noise on pooled MoisesDB (+-0.01)


def trained_from_manifest(manifest: dict, tax: Taxonomy) -> set[str]:
    """Nodes a model was trained on, exactly as the ONNX sidecar's `trained` list
    (`export.trained_nodes`): every labelled node with a positive count, closed upward."""
    return set(tax.close_upward(n for n, c in manifest["leaf_counts"].items() if c > 0))


def scoreable(a: EvalArrays, tax: Taxonomy, nodes: Sequence[str], *, min_pos: int = 3,
              min_neg: int = 3, min_groups: int = 1) -> list[str]:
    """Nodes (taxonomy order) with enough observed positives, negatives and positive
    artists in these arrays to give an AP worth bootstrapping."""
    stats = per_node(a, tax)
    keys = a.group_keys()
    tidx = tax.index()
    out = []
    for n in tax.nodes:
        if n not in nodes or n not in stats:
            continue
        s = stats[n]
        if s["skipped"] is not None or s["n_positive"] < min_pos:
            continue
        if s["n_observed"] - s["n_positive"] < min_neg:
            continue
        t = tidx[n]
        pos_groups = {keys[i] for i in np.nonzero(a.y_full[:, t] & a.m_full[:, t])[0]}
        if len(pos_groups) >= min_groups:
            out.append(n)
    return out


def _paired(c: EvalArrays, i: EvalArrays, tax: Taxonomy, nodes: Sequence[str],
            n_boot: int, seed: int) -> dict[str, Any]:
    return paired_bootstrap(restrict(c, nodes), restrict(i, nodes), tax, ["map_macro"],
                            n_boot=n_boot, seed=seed)["map_macro"]


def check_old(c: EvalArrays, i: EvalArrays, tax: Taxonomy, nodes: Sequence[str], *,
              n_boot: int = 1000, seed: int = 0, rel_tol: float = REL_TOL,
              node_floor: float = NODE_FLOOR) -> dict[str, Any]:
    if not nodes:
        return {"pass": False, "reasons": ["no scoreable shared nodes"], "nodes": []}
    sub = _paired(c, i, tax, nodes, n_boot, seed)
    inc_map = float(np.mean([per_node(restrict(i, [n]), tax)[n]["ap"] for n in nodes]))
    allowed = rel_tol * inc_map
    per = {n: _paired(c, i, tax, [n], n_boot, seed) for n in nodes}
    reasons = []
    if sub["delta"] is None or sub["delta"] < -allowed - 1e-12:
        reasons.append(f"subset mean AP delta {sub['delta']} < -{allowed:.4f} (harness tolerance)")
    bad = [n for n, r in per.items() if r["ci"] is not None and r["ci"][1] < -node_floor]
    reasons += [f"{n}: CI {per[n]['ci']} entirely below -{node_floor}" for n in bad]
    return {"pass": not reasons, "reasons": reasons, "nodes": list(nodes), "incumbent_map": inc_map,
            "allowed_regression": allowed, "subset": sub, "per_node": per}


def check_new(c: EvalArrays, i: EvalArrays, tax: Taxonomy, nodes: Sequence[str], *,
              n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    if not nodes:
        return {"pass": False, "reasons": ["no scoreable new nodes"], "nodes": []}
    chance = constant_arrays(c)
    vs_inc = _paired(c, i, tax, nodes, n_boot, seed)
    vs_chance = _paired(c, chance, tax, nodes, n_boot, seed)
    stats = per_node(restrict(c, nodes), tax)
    per = {}
    for n in nodes:
        s = stats[n]
        prev = s["prevalence"] or 0.0
        per[n] = {"ap": s["ap"], "prevalence": prev, "n_positive": s["n_positive"],
                  "lift": None if s["ap"] is None or prev >= 1 else (s["ap"] - prev) / (1 - prev),
                  "vs_chance": _paired(c, chance, tax, [n], n_boot, seed)}
    reasons = []
    for name, r in (("incumbent", vs_inc), ("chance", vs_chance)):
        if r["ci"] is None or r["ci"][0] <= 0:
            reasons.append(f"subset mean AP vs {name}: CI {r['ci']} does not exclude 0")
    return {"pass": not reasons, "reasons": reasons, "nodes": list(nodes),
            "vs_incumbent": vs_inc, "vs_chance": vs_chance, "per_node": per}


def _arrays(run: Path) -> EvalArrays:
    d = run if run.is_dir() else run.parent
    return EvalArrays.load(d / "arrays.npz")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m disstruments.ml.eval.adopt")
    ap.add_argument("check", choices=("old", "new"))
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--cand-manifest", required=True, type=Path, help="render manifest.json of the candidate")
    ap.add_argument("--inc-manifest", required=True, type=Path, help="render manifest.json of the incumbent")
    ap.add_argument("--min-pos", type=int, default=3)
    ap.add_argument("--min-neg", type=int, default=3)
    ap.add_argument("--min-artists", type=int, default=1, help="positive artists a node needs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    tax = get_taxonomy()
    c, i = _arrays(a.candidate), _arrays(a.incumbent)
    tc = trained_from_manifest(json.loads(a.cand_manifest.read_text()), tax)
    ti = trained_from_manifest(json.loads(a.inc_manifest.read_text()), tax)
    pool = (tc & ti) if a.check == "old" else (tc - ti)
    nodes = scoreable(c, tax, pool, min_pos=a.min_pos, min_neg=a.min_neg, min_groups=a.min_artists)
    fn = check_old if a.check == "old" else check_new
    res = fn(c, i, tax, nodes, n_boot=a.n_boot, seed=a.seed)
    res["unscoreable_pool"] = sorted(pool - set(nodes))
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(txt)
    print(txt)
    print(f"{a.check}: {'PASS' if res['pass'] else 'FAIL'} over {len(nodes)} nodes", file=sys.stderr)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
