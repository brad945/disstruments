"""Dataset loaders. All return a `DatasetIndex` of `Record`s over the taxonomy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetIndex, Record, StemLabels, UnknownLabelsError

DATASETS = ("medleydb", "openmic", "slakh", "synthetic")


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
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    return load(root, **opts)


__all__ = ["DATASETS", "DatasetIndex", "Record", "StemLabels", "UnknownLabelsError", "load_dataset"]
