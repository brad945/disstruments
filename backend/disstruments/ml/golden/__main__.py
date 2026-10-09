"""CLI: python -m disstruments.ml.golden page [--out DIR]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..taxonomy import Taxonomy

DEFAULT_DIR = Path.home() / "datasets" / "golden"


def build_page(out_dir: Path) -> Path:
    tax = Taxonomy.load()
    leaves = [{"id": l, "name": tax.info[l].name, "family": l.split(".")[0],
               "path": " › ".join(tax.info[a].name for a in reversed(tax.ancestors(l)))}
              for l in tax.leaves()]
    html = (Path(__file__).with_name("label_template.html").read_text()
            .replace("/*__LEAVES__*/[]", json.dumps(leaves))
            .replace("__TAXVER__", tax.version))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    page = out_dir / "label.html"
    page.write_text(html)
    return page


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m disstruments.ml.golden")
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("page", help="write the local labelling page")
    pg.add_argument("--out", type=Path, default=DEFAULT_DIR)
    co = sub.add_parser("collect", help="move *.labels.json from ~/Downloads into labels/")
    co.add_argument("--out", type=Path, default=DEFAULT_DIR)
    a = p.parse_args(argv)
    if a.cmd == "collect":
        (a.out / "labels").mkdir(parents=True, exist_ok=True)
        moved = []
        for f in (Path.home() / "Downloads").glob("*.labels.json"):
            f.rename(a.out / "labels" / f.name)
            moved.append(f.name)
        print(f"moved {len(moved)}: {moved}")
        return 0
    print(build_page(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
