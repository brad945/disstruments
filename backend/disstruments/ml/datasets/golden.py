"""Golden set loader (ML_ENGINEERING §3.3): `<root>/labels/*.labels.json` written by the
labelling page (`python -m disstruments.ml.golden page`) + local audio in `<root>/audio/`.

Per leaf: present -> positive (ancestor-closed); unsure -> that leaf unknown (its ancestors
stay observed only through other evidence); absent -> negative *only if* the labeller
confirmed `listened_fully`, otherwise every non-present leaf is unknown. Evaluation only.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..taxonomy import Taxonomy
from .base import DatasetIndex, Record, check_unique_ids

NAME = "golden"


def load(root: Path | str, *, strict: bool = True, taxonomy: Taxonomy | None = None) -> DatasetIndex:
    tax = taxonomy or Taxonomy.load()
    root = Path(root)
    files = sorted((root / "labels").glob("*.labels.json"))
    if not files:
        raise FileNotFoundError(f"golden: no labels/*.labels.json under {root}")
    all_nodes = frozenset(tax.nodes)
    records, missing_audio = [], []
    for f in files:
        d = json.loads(f.read_text())
        labels = d.get("labels") or {}
        bad = [l for l in labels if l not in tax]
        if bad and strict:
            raise ValueError(f"golden {f.name}: unknown leaves {bad}")
        present = [l for l, v in labels.items() if v.get("state") == "present" and l in tax]
        unsure = [l for l, v in labels.items() if v.get("state") == "unsure" and l in tax]
        pos = frozenset(tax.close_upward(present))
        if d.get("listened_fully"):
            unknown = set(unsure)
        else:                                   # only positives are trustworthy
            unknown = set(tax.leaves()) - set(present)
        # A node is unknown if it is an unknown leaf, or an ancestor whose status rests
        # only on unknown children (not already implied positive).
        unk_closed = {a for u in unknown for a in tax.close_upward([u])} - pos
        observed = all_nodes - frozenset(unk_closed)
        audio = root / "audio" / d.get("audio_filename", "")
        if not audio.exists():
            missing_audio.append(d.get("audio_filename"))
        item = (d.get("artist", "") + " - " + d.get("title", "")).strip(" -") or f.stem
        records.append(Record(dataset=NAME, item_id=item, split="test",
                              artist=(d.get("artist") or item).strip().lower(),
                              positive=pos, observed=observed,
                              source_labels=tuple(sorted(labels)),
                              audio={"mix": audio} if audio.exists() else {},
                              extra={"genre": d.get("genre"), "listened_fully": d.get("listened_fully"),
                                     "intervals": {l: v.get("intervals", []) for l, v in labels.items()}}))
    check_unique_ids(NAME, [r.item_id for r in records])
    return DatasetIndex(NAME, records, {"dataset": NAME, "taxonomy_version": tax.version},
                        {"n_songs": len(records), "missing_audio": missing_audio})
