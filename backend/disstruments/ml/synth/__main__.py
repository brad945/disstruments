"""CLI: python -m disstruments.ml.synth {sources,build}."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .sources import load_registry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.synth")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources", help="validate sources.yaml and print leaf coverage")
    s.add_argument("--registry")
    s.add_argument("--sources-root")
    b = sub.add_parser("build", help="render a synthetic dataset")
    b.add_argument("--midi-root", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--n-clips", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sr", type=int, default=44100)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--registry")
    b.add_argument("--sources-root")
    b.add_argument("--commercial-only", action="store_true")
    b.add_argument("--fake", action="store_true", help="fake engine for every source (dry run)")
    b.add_argument("--limit-files", type=int)
    b.add_argument("--windows-per-file", type=int, default=2)
    b.add_argument("--fx-profile", default="v1", choices=("v1", "v2"))
    b.add_argument("--holdout", nargs="*", default=[], metavar="SOURCE:LEAF",
                   help="leave-source-out pairs: excluded from train/val; test windows are "
                        "re-rendered with them as split 'test_lso'")
    b.add_argument("--stem-fraction", type=float, default=1.0,
                   help="save stem audio for this fraction of clips (labels always complete)")
    au = sub.add_parser("audition", help="dry examples per source -> local HTML listening page")
    au.add_argument("--midi-root", required=True, type=Path)
    au.add_argument("--out", required=True, type=Path)
    au.add_argument("--n", type=int, default=3)
    a = p.parse_args(argv)
    if a.cmd == "audition":
        from .audition import build_audition
        print(build_audition(a.midi_root, a.out, n=a.n))
        return 0
    if a.cmd == "sources":
        reg = load_registry(a.registry, root=a.sources_root)
        missing = [x.id for x in reg.sources if x.path and not x.file(reg.root).exists()]
        print(f"{len(reg.sources)} sources, root {reg.root}")
        for leaf in reg.v1_leaves:
            srcs = reg.for_leaf(leaf)
            clean = sum(x.commercial_clean for x in srcs)
            print(f"  {leaf:32s} {len(srcs)} sources ({clean} commercial-clean): "
                  + ", ".join(x.id for x in srcs))
        print("licences:", dict(Counter(x.license for x in reg.sources)))
        if missing:
            print("MISSING FILES:", missing)
            return 1
        return 0
    from .build import build
    m = build(a.midi_root, a.out, a.n_clips, seed=a.seed, sr=a.sr, workers=a.workers,
              registry_path=a.registry, sources_root=a.sources_root,
              commercial_only=a.commercial_only, fake=a.fake, limit_files=a.limit_files,
              windows_per_file=a.windows_per_file, stem_fraction=a.stem_fraction,
              fx_profile=a.fx_profile, holdout=a.holdout)
    print(json.dumps({k: m[k] for k in ("n_rendered", "splits", "timing_s", "stem_failures")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
