"""Synthetic clips (M2 S4): `<root>/clips/<clip_id>/labels.json` from `ml.synth.build`.

Labels are perfect and fully observed: every audible stem was rendered from a known leaf
and nothing else is in the audio, so positives are the ancestor-closed rendered leaves and
**every other taxonomy node is a true negative**. The grouping key (`artist`) is the MIDI
artist (Lakh clean_midi `Artist/Title.mid`), which is also the split key: splits are
artist-disjoint and the bootstrap resamples artists.
Options:
- `commercial_only` keeps only clips whose sources *and* MIDI are licence-clean.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..taxonomy import Taxonomy
from .base import DatasetIndex, Record, StemLabels, check_unique_ids

NAME = "synthetic"


def load(root: Path | str, *, commercial_only: bool = False, strict: bool = True,
         taxonomy: Taxonomy | None = None) -> DatasetIndex:
    """`strict` is accepted for CLI symmetry; synthetic labels are never unmapped."""
    tax = taxonomy or Taxonomy.load()
    root = Path(root)
    files = sorted((root / "clips").glob("*/labels.json"))
    if not files:
        raise FileNotFoundError(f"synthetic: no clips/*/labels.json under {root}")
    all_nodes = frozenset(tax.nodes)
    records, skipped, versions = [], 0, set()
    for f in files:
        lab = json.loads(f.read_text())
        if commercial_only and not lab.get("commercial_clean"):
            skipped += 1
            continue
        versions.add(lab.get("taxonomy_version"))
        leaves = lab["positive_leaves"]
        bad = [l for l in leaves if l not in tax]
        if bad:
            raise ValueError(f"synthetic {lab['clip_id']}: leaves not in taxonomy {tax.version}: {bad}")
        cdir = f.parent
        stems = tuple(
            StemLabels(stem_id=s["leaf"], positive=frozenset(tax.close_upward(
                [s["leaf"]] + (list(lab.get("implied_leaves") or {}) if s["leaf"].startswith("drums") else []))),
                       observed=all_nodes, source_labels=(s["source_id"],),
                       audio=(cdir / "stems" / f"{s['leaf']}.flac") if lab.get("stems_saved", True) else None,
                       bleed=False)
            for s in lab["stems"])
        records.append(Record(
            dataset=NAME, item_id=lab["clip_id"], split=lab["split"],
            artist=lab["midi"]["composition"].split("/", 1)[0],   # = split key
            positive=frozenset(tax.close_upward(leaves)), observed=all_nodes,
            source_labels=tuple(s["source_id"] for s in lab["stems"]),
            audio={"mix": cdir / "mix.flac"}, stems=stems,
            extra={"commercial_clean": lab.get("commercial_clean"),
                   "midi_md5": lab["midi"]["midi_md5"]}))
    check_unique_ids(NAME, [r.item_id for r in records])
    config = {"dataset": NAME, "commercial_only": commercial_only,
              "taxonomy_versions": sorted(v for v in versions if v)}
    stats = {"n_clips": len(records), "skipped_not_clean": skipped,
             "n_compositions": len({r.artist for r in records})}
    return DatasetIndex(NAME, records, config, stats)
