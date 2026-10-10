"""Dataset loaders. All return a `DatasetIndex` of `Record`s over the taxonomy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetIndex, Record, StemLabels, UnknownLabelsError

DATASETS = ("medleydb", "openmic", "slakh", "synthetic", "golden",
            "idmt_bass", "idmt_bass_tracks", "idmt_fx")   # idmt_*: test-only


def load_dataset(name: str, root: Path | str, **opts: Any) -> DatasetIndex:
    """Dispatch to a loader; `opts` are passed through (None values dropped)."""
    opts = {k: v for k, v in opts.items() if v is not None}
    if name == "medleydb":
        from .medleydb import load
    elif name == "openmic":
        from .openmic import load
    elif name == "slakh":
        from .slakh import load
    elif name == "synthetic":
        from .synthetic import load
    elif name == "golden":
        from .golden import load
    elif name == "idmt_bass":
        from .idmt import load_bass as load
    elif name == "idmt_bass_tracks":
        from .idmt import load_bass_tracks as load
    elif name == "idmt_fx":
        from .idmt import load_fx as load
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    return load(root, **opts)


__all__ = ["DATASETS", "DatasetIndex", "Record", "StemLabels", "UnknownLabelsError", "load_dataset"]
