"""Dataset build driver (M2 S2+S3): MIDI files -> windows -> leaf assignment -> clips.

    python -m disstruments.ml.synth build --midi-root ~/datasets/disstruments-synth/midi/clean_midi \
        --out ~/datasets/disstruments-synth/v1 --n-clips 10000 --workers 8

Deterministic for (inputs, seed). Resumable: clips already on disk are counted, not redone.
A `manifest.json` records the run config, renderer SHA and per-leaf counts.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..taxonomy import Taxonomy
from .midi import LeafBalancer, Window, assign_leaves, load_windows, split_of
from .render import renderer_sha, render_clip
from .sources import load_registry

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def plan_windows(midi_files: Iterable[Path], midi_root: Path, leaves: Iterable[str],
                 n_clips: int, seed: int, *, windows_per_file: int = 2,
                 max_parts: int = 5) -> list[tuple[Window, str, int]]:
    """Choose windows + leaves for `n_clips` clips: (window, split, clip_seed)."""
    rng = np.random.default_rng(seed)
    files = list(midi_files)
    rng.shuffle(files)
    bal = LeafBalancer(leaves, rng)
    plan: list[tuple[Window, str, int]] = []
    for f in files:
        if len(plan) >= n_clips:
            break
        wins = load_windows(f, midi_root)
        if not wins:
            continue
        for k in rng.choice(len(wins), size=min(windows_per_file, len(wins)), replace=False):
            w = assign_leaves(wins[int(k)], bal, max_parts=max_parts)
            if w.parts:
                plan.append((w, split_of(w.composition, FRACTIONS), int(rng.integers(2 ** 31))))
            if len(plan) >= n_clips:
                break
    return plan


_WORKER: dict[str, Any] = {}


def _init_worker(registry_path: str | None, sources_root: str | None, fake: bool) -> None:
    from .engines import FakeEngine, default_engines
    _WORKER["reg"] = load_registry(registry_path, root=sources_root)
    _WORKER["eng"] = {"fake": FakeEngine()} if fake else default_engines()


def _render_one(args: tuple) -> dict | None:
    w, split, clip_seed, out, sr, commercial_only, tax_version, sha = args
    reg, eng = _WORKER["reg"], _WORKER["eng"]
    if "fake" in eng and len(eng) == 1:                  # dry run: route everything to fake
        from dataclasses import replace
        reg = replace(reg, sources=tuple(replace(s, engine="fake") for s in reg.sources))
    return render_clip(w, reg, eng, seed=clip_seed, out_dir=Path(out), sr=sr, split=split,
                       commercial_only=commercial_only, taxonomy_version=tax_version, sha=sha)


def build(midi_root: Path, out: Path, n_clips: int, *, seed: int = 0, sr: int = 44100,
          workers: int = 1, registry_path: str | None = None, sources_root: str | None = None,
          commercial_only: bool = False, fake: bool = False, limit_files: int | None = None) -> dict:
    tax = Taxonomy.load()
    reg = load_registry(registry_path, tax, root=sources_root)
    files = sorted(midi_root.rglob("*.mid"))[: limit_files or None]
    t0 = time.monotonic()
    plan = plan_windows(files, midi_root, reg.v1_leaves, n_clips, seed)
    t_plan = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    sha = renderer_sha()
    jobs = [(w, sp, cs, str(out), sr, commercial_only, tax.version, sha) for w, sp, cs in plan]
    t1 = time.monotonic()
    results: list[dict | None] = []
    if workers <= 1:
        _init_worker(registry_path, sources_root, fake)
        results = [_render_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(registry_path, sources_root, fake)) as ex:
            results = list(ex.map(_render_one, jobs, chunksize=4))
    ok = [r for r in results if r]
    leaf_counts = Counter(l for r in ok for l in r["positive_leaves"])
    src_counts = Counter(s["source_id"] for r in ok for s in r["stems"])
    fail_counts = Counter(f["error"].split(":")[0] for r in ok for f in r["failures"])
    manifest = {
        "seed": seed, "sr": sr, "n_requested": n_clips, "n_planned": len(plan),
        "n_rendered": len(ok), "renderer_sha": sha, "taxonomy_version": tax.version,
        "commercial_only": commercial_only, "fake": fake, "fractions": FRACTIONS,
        "splits": dict(Counter(r["split"] for r in ok)),
        "leaf_counts": dict(sorted(leaf_counts.items())),
        "source_counts": dict(sorted(src_counts.items())),
        "stem_failures": dict(fail_counts),
        "timing_s": {"plan": round(t_plan, 1), "render": round(time.monotonic() - t1, 1),
                     "per_clip": round((time.monotonic() - t1) / max(1, len(ok)), 3)},
        "workers": workers, "cpu_count": os.cpu_count(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
