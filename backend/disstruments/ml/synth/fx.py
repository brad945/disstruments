"""Randomized per-stem effects + mastering chains (M2 S3; domain randomization, §5).

Rule: an effect may change the *production* of a sound, never its *identity*. So each
leaf family has a whitelist: distortion only where the leaf is distorted (or a little
grit on bass/keys), never on clean guitar; chorus/phaser only on families that commonly
use them. Every applied plugin and its parameters are returned for `labels.json`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Chain = list[dict[str, Any]]   # [{"plugin": name, "params": {...}}, ...]


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(rng.uniform(lo, hi)), 4)


def _family(leaf: str) -> str:
    if leaf == "guitar.electric.distorted":
        return "guitar_dist"
    if leaf.startswith("guitar.electric"):
        return "guitar_clean"
    if leaf.startswith("guitar.acoustic"):
        return "guitar_acoustic"
    for prefix, fam in (("bass", "bass"), ("drums", "drums"), ("keys.synth", "synth"),
                        ("keys", "keys"), ("brass", "wind"), ("woodwinds", "wind")):
        if leaf.startswith(prefix):
            return fam
    return "other"


def stem_chain(leaf: str, rng: np.random.Generator) -> Chain:
    """Randomized effect specs for one stem (applied by `apply_chain`)."""
    fam = _family(leaf)
    ch: Chain = []
    if fam == "guitar_dist":
        ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 18, 40)}})
        ch.append({"plugin": "LowpassFilter", "params": {"cutoff_frequency_hz": _u(rng, 4500, 8000)}})
    if fam == "bass" and rng.random() < 0.25:
        ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 3, 10)}})
    if fam == "keys" and rng.random() < 0.15:
        ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 2, 8)}})
    ch.append({"plugin": "HighpassFilter",
               "params": {"cutoff_frequency_hz": _u(rng, 20, 45) if fam in ("bass", "drums")
                          else _u(rng, 40, 140)}})
    for _ in range(int(rng.integers(0, 3))):                     # 0-2 random EQ bands
        ch.append({"plugin": "PeakFilter", "params": {
            "cutoff_frequency_hz": _u(rng, 150, 8000), "gain_db": _u(rng, -6, 6),
            "q": _u(rng, 0.5, 2.0)}})
    if rng.random() < 0.7:
        ch.append({"plugin": "Compressor", "params": {
            "threshold_db": _u(rng, -30, -10), "ratio": _u(rng, 1.5, 6),
            "attack_ms": _u(rng, 1, 30), "release_ms": _u(rng, 50, 300)}})
    if fam in ("guitar_clean", "keys", "synth") and rng.random() < 0.25:
        ch.append({"plugin": "Chorus", "params": {
            "rate_hz": _u(rng, 0.2, 2.0), "depth": _u(rng, 0.1, 0.4), "mix": _u(rng, 0.2, 0.5)}})
    if fam in ("guitar_clean", "guitar_dist", "synth") and rng.random() < 0.15:
        ch.append({"plugin": "Phaser", "params": {"rate_hz": _u(rng, 0.2, 1.5), "mix": _u(rng, 0.2, 0.5)}})
    if fam not in ("bass",) and rng.random() < 0.3:
        ch.append({"plugin": "Delay", "params": {
            "delay_seconds": _u(rng, 0.08, 0.5), "feedback": _u(rng, 0.1, 0.45),
            "mix": _u(rng, 0.05, 0.25)}})
    if rng.random() < (0.3 if fam in ("bass", "drums") else 0.8):
        ch.append({"plugin": "Reverb", "params": {
            "room_size": _u(rng, 0.1, 0.9), "damping": _u(rng, 0.2, 0.8),
            "wet_level": _u(rng, 0.05, 0.35), "dry_level": _u(rng, 0.7, 1.0)}})
    return ch


def master_chain(rng: np.random.Generator) -> Chain:
    """Mix-bus processing: the strongest single domain-randomization lever (§5)."""
    ch: Chain = []
    if rng.random() < 0.6:
        ch.append({"plugin": "LowShelfFilter", "params": {
            "cutoff_frequency_hz": _u(rng, 80, 200), "gain_db": _u(rng, -3, 3)}})
        ch.append({"plugin": "HighShelfFilter", "params": {
            "cutoff_frequency_hz": _u(rng, 6000, 12000), "gain_db": _u(rng, -3, 4)}})
    if rng.random() < 0.8:
        ch.append({"plugin": "Compressor", "params": {
            "threshold_db": _u(rng, -20, -6), "ratio": _u(rng, 1.5, 4),
            "attack_ms": _u(rng, 5, 40), "release_ms": _u(rng, 80, 400)}})
    ch.append({"plugin": "Limiter", "params": {"threshold_db": _u(rng, -6, -0.5),
                                               "release_ms": _u(rng, 30, 200)}})
    return ch


def apply_chain(audio: np.ndarray, sr: int, chain: Chain) -> np.ndarray:
    """audio: (channels, n) float32. Uses pedalboard (optional `synth` extra)."""
    if not chain:
        return audio
    import pedalboard

    board = pedalboard.Pedalboard([getattr(pedalboard, c["plugin"])(**c["params"]) for c in chain])
    return board(audio.astype(np.float32), sr)
