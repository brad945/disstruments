"""Clip renderer (M2 S3): window -> stems -> fx -> mix -> master -> files + labels.json.

Output layout under `<out>/clips/<clip_id>/`: `mix.flac` and `stems/<leaf>.flac` (mono,
`sr`, 16-bit FLAC), plus `labels.json` holding the full recipe (DIRECTION_NOTES: every
clip stores what made it, so later "sound match" / "how was it made" work is free).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .engines import Engine
from .fx import apply_chain, master_chain, master_chain_v2, stem_chain, stem_chain_v2
from .midi import CYMBAL_NOTES, Window
from .sources import Registry

SCHEMA_VERSION = 1


def _lufs(audio: np.ndarray, sr: int) -> float | None:
    import pyloudnorm

    if len(audio) < sr // 2 or not np.any(audio):
        return None
    v = pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64))
    return None if not np.isfinite(v) else round(float(v), 2)


def _gain_to(audio: np.ndarray, sr: int, target: float) -> tuple[np.ndarray, float]:
    cur = _lufs(audio, sr)
    if cur is None:
        return audio, 0.0
    g = target - cur
    return audio * np.float32(10 ** (g / 20)), round(g, 2)


def renderer_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def render_clip(window: Window, registry: Registry, engines: Mapping[str, Engine], *,
                seed: int, out_dir: Path, sr: int = 44100, split: str = "train",
                commercial_only: bool = False, taxonomy_version: str = "",
                sha: str = "", midi_clean: bool = False,
                midi_license: str = "CC-BY-4.0 (Lakh MIDI compilation; compositions not cleared)",
                save_stems: bool = True, fx_profile: str = "v1",
                holdout: frozenset = frozenset(), source_role: str = "train") -> dict[str, Any] | None:
    """Render one clip. Returns the labels dict (also written to disk), or None when no
    part could be rendered (e.g. every source failed or was silent)."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    clip_id = hashlib.sha1(f"{window.midi_id}:{window.start}:{seed}"
                           f"{':' + source_role if source_role != 'train' else ''}".encode()).hexdigest()[:16]
    cdir = out_dir / "clips" / clip_id
    if (cdir / "labels.json").exists():                         # resumable builds
        return json.loads((cdir / "labels.json").read_text())
    (cdir / "stems").mkdir(parents=True, exist_ok=True)
    n = int(sr * window.duration)
    stems_meta, stem_audio, failures = [], [], []
    for part in window.parts:
        cands = registry.for_leaf(part.leaf or "", commercial_only=commercial_only)
        cands = [s for s in cands if s.engine in engines]
        # Leave-source-out (M4 rigor): held-out (source, leaf) pairs never appear in the
        # train pool; the LSO test uses them exclusively wherever a leaf has one.
        held = [s for s in cands if (s.id, part.leaf) in holdout]
        if source_role == "lso" and held:
            cands = held
        else:
            cands = [s for s in cands if (s.id, part.leaf) not in holdout]
        if not cands:
            failures.append({"leaf": part.leaf, "error": "no source"})
            continue
        w = np.array([s.weight for s in cands], dtype=float)
        src = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        try:
            dry, emeta = engines[src.engine].render(part, src, registry.root, sr, window.duration, rng)
        except Exception as e:  # noqa: BLE001 — one bad source must not kill the clip
            failures.append({"leaf": part.leaf, "source": src.id, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        chain = (stem_chain_v2 if fx_profile == "v2" else stem_chain)(part.leaf or "", rng)
        wet = apply_chain(dry[None, :], sr, chain)[0]
        wet, _ = _gain_to(wet, sr, -20.0)                       # level-match before mixing
        # A positive label must be audible: measurable loudness AND a real peak after fx
        # (tiny non-zero renders used to pass and quantize to silence in the FLAC).
        if _lufs(wet, sr) is None or float(np.max(np.abs(wet))) < 1e-3:
            failures.append({"leaf": part.leaf, "source": src.id, "error": "silent render"})
            continue
        mix_gain_db = round(float(rng.uniform(-6, 3)), 2)       # random balance
        wet = wet * np.float32(10 ** (mix_gain_db / 20))
        if save_stems:          # disk: stems are kept for a subset of clips only (v2)
            sf.write(cdir / "stems" / f"{part.leaf}.flac", np.clip(wet, -1, 1), sr, subtype="PCM_16")
        stem_audio.append(wet)
        stems_meta.append({
            "leaf": part.leaf, "source_id": src.id, "source_license": src.license,
            "commercial_clean": src.commercial_clean, "engine": src.engine,
            "engine_meta": emeta, "fx_chain": chain, "mix_gain_db": mix_gain_db,
            "lufs": _lufs(wet, sr),
            "midi": {"track_index": part.track_index, "gm_program": part.program,
                     "gm_name": part.gm_name, "is_drum": part.is_drum, "role": part.role,
                     "n_notes": len(part.notes)},
        })
    if not stem_audio:
        for p in (cdir / "stems").glob("*"):
            p.unlink()
        (cdir / "stems").rmdir()
        cdir.rmdir()
        return None
    mix = np.sum(stem_audio, axis=0)
    mchain = (master_chain_v2 if fx_profile == "v2" else master_chain)(rng)
    mix = apply_chain(mix[None, :], sr, mchain)[0]
    target = round(float(rng.uniform(-18, -9)), 2)
    mix, _ = _gain_to(mix, sr, target)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.999:                                            # final safety, recorded
        mix = mix * np.float32(0.999 / peak)
    sf.write(cdir / "mix.flac", mix, sr, subtype="PCM_16")
    # Kits that play hats/cymbals make the level-1 `cymbals` leaf audible: label it.
    implied = sorted({"cymbals"} if any(
        p.is_drum and p.leaf and p.leaf.startswith("drums") and p.leaf in {s["leaf"] for s in stems_meta}
        and any(n.pitch in CYMBAL_NOTES for n in p.notes) for p in window.parts) else set())
    labels = {
        "schema_version": SCHEMA_VERSION, "clip_id": clip_id, "split": split, "seed": seed,
        "renderer_sha": sha, "taxonomy_version": taxonomy_version, "sample_rate": sr,
        "duration_s": window.duration,
        "stems_saved": save_stems, "fx_profile": fx_profile, "source_role": source_role,
        "held_out_stems": sorted(s["leaf"] for s in stems_meta if (s["source_id"], s["leaf"]) in holdout),
        "positive_leaves": sorted({s["leaf"] for s in stems_meta} | set(implied)),
        "implied_leaves": {l: "hat/cymbal notes in the drum part" for l in implied},
        # Lakh compositions are not cleared, so Lakh renders are never commercially clean
        # (licence-clean end state, docs/DIRECTION_NOTES.md); the sources may still be.
        "sources_commercial_clean": all(s["commercial_clean"] for s in stems_meta),
        "commercial_clean": midi_clean and all(s["commercial_clean"] for s in stems_meta),
        "midi": {"midi_md5": window.midi_id, "composition": window.composition,
                 "path": window.path, "window_start_s": window.start, "tempo": window.tempo,
                 "license": midi_license, "commercial_clean": midi_clean},
        "stems": stems_meta,
        "mix": {"master_chain": mchain, "target_lufs": target, "lufs": _lufs(mix, sr),
                "peak": round(float(np.max(np.abs(mix))), 4)},
        "failures": failures,
    }
    (cdir / "labels.json").write_text(json.dumps(labels, indent=1, default=_json_default))
    return labels


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def window_to_dict(w: Window) -> dict[str, Any]:
    return asdict(w)
