"""Our own synth patches (M2 D8): per-leaf parameter recipes with seeded randomization.

Factory presets with unverified licences are never loaded; every patch is built here from
plugin defaults, so the parameter dict is exact ground truth for "how was this sound made"
(docs/DIRECTION_NOTES.md). Recipes are filled in per engine once its parameter names are
inspected (see `describe_parameters`).
"""
from __future__ import annotations

from typing import Any

import numpy as np

RECIPES: dict[str, dict[str, Any]] = {}   # engine -> leaf -> recipe fn (filled in S3)


def describe_parameters(plugin) -> dict[str, Any]:
    """Name -> (current value, range/valid values) for writing recipes."""
    out = {}
    for name, p in plugin.parameters.items():
        out[name] = {"value": getattr(p, "raw_value", None),
                     "string": getattr(p, "string_value", None),
                     "valid": list(getattr(p, "valid_values", []) or [])[:12]}
    return out


def make_patch(engine: str, leaf: str, rng: np.random.Generator, plugin) -> dict[str, Any]:
    """Apply a randomized recipe for `leaf` to `plugin`; return the recipe actually used."""
    recipes = RECIPES.get(engine) or {}
    if leaf not in recipes:
        raise NotImplementedError(f"no {engine} patch recipe for {leaf}")
    return recipes[leaf](plugin, rng)
