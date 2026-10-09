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


def _room(rng: np.random.Generator, wet_lo: float, wet_hi: float) -> dict:
    """Synthetic convolution room: exponentially decaying filtered noise + early
    reflections, generated from these params at apply time (licence-clean, recorded)."""
    return {"plugin": "SynthRoomIR", "params": {
        "rt60_s": _u(rng, 0.15, 2.2), "predelay_ms": _u(rng, 0, 40), "n_early": int(rng.integers(2, 8)),
        "brightness": _u(rng, 0.2, 1.0), "mix": _u(rng, wet_lo, wet_hi), "seed": int(rng.integers(2 ** 31))}}


def stem_chain_v2(leaf: str, rng: np.random.Generator) -> Chain:
    """v2 (M4 L2): v1's identity-safe chain plus stronger, realistic production variety:
    amp-style drive + speaker-cab EQ on electric guitar/bass, convolution rooms instead of
    the algorithmic reverb, and more aggressive EQ."""
    fam = _family(leaf)
    ch = [c for c in stem_chain(leaf, rng) if c["plugin"] != "Reverb"]
    if fam in ("guitar_clean", "guitar_dist") or (fam == "bass" and rng.random() < 0.5):
        if fam == "guitar_dist":
            ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 5, 15)}})   # 2nd stage
        elif fam == "guitar_clean" and rng.random() < 0.5:
            ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 1, 4)}})    # amp warmth, still clean
        ch += [{"plugin": "HighpassFilter", "params": {"cutoff_frequency_hz": _u(rng, 70, 120)}},
               {"plugin": "LowpassFilter", "params": {"cutoff_frequency_hz": _u(rng, 3500, 7000)}},
               {"plugin": "PeakFilter", "params": {"cutoff_frequency_hz": _u(rng, 1500, 3500),
                                                   "gain_db": _u(rng, 1, 6), "q": _u(rng, 0.7, 1.5)}}]
    if rng.random() < 0.4:
        ch.append({"plugin": "LowShelfFilter", "params": {"cutoff_frequency_hz": _u(rng, 100, 300),
                                                          "gain_db": _u(rng, -6, 6)}})
    if rng.random() < (0.35 if fam in ("bass", "drums") else 0.85):
        ch.append(_room(rng, 0.05, 0.45))
    return ch


def master_chain_v2(rng: np.random.Generator) -> Chain:
    """v2 mastering: v1 chain plus tape/console saturation, bandwidth limits, a noise floor
    and lossy-codec round trips (OpenMIC/FMA clips are MP3s)."""
    ch = master_chain(rng)
    lim = ch.pop()                                                     # keep limiter last
    if rng.random() < 0.4:
        ch.append({"plugin": "Distortion", "params": {"drive_db": _u(rng, 1, 6)}})
    if rng.random() < 0.25:
        ch.append({"plugin": "LowpassFilter", "params": {"cutoff_frequency_hz": _u(rng, 7000, 15000)}})
    if rng.random() < 0.3:
        ch.append(_room(rng, 0.03, 0.15))                              # bus/room glue
    ch.append(lim)
    if rng.random() < 0.5:
        ch.append({"plugin": "NoiseFloor", "params": {"level_db": _u(rng, -80, -50), "seed": int(rng.integers(2 ** 31))}})
    if rng.random() < 0.6:
        ch.append({"plugin": "MP3Compressor", "params": {"vbr_quality": _u(rng, 0.0, 9.0)}})
    return ch


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


def synth_room_ir(sr: int, rt60_s: float, predelay_ms: float, n_early: int, brightness: float,
                  seed: int, **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * min(rt60_s * 1.2, 3.0))
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / max(rt60_s, 0.05))   # -60 dB at rt60
    k = max(1, int(1 + (1 - brightness) * 20))                              # darker = smoother
    tail = np.convolve(tail, np.ones(k) / k, mode="same")
    ir = np.zeros(n + int(sr * predelay_ms / 1000), np.float32)
    ir[0] = 1.0
    off = int(sr * predelay_ms / 1000)
    ir[off:off + n] += 0.3 * tail.astype(np.float32)
    for _ in range(n_early):
        d = int(rng.uniform(0.003, 0.05) * sr)
        if d < len(ir):
            ir[d] += rng.uniform(0.1, 0.5) * rng.choice([-1, 1])
    return (ir / np.max(np.abs(ir))).astype(np.float32)


def apply_chain(audio: np.ndarray, sr: int, chain: Chain) -> np.ndarray:
    """audio: (channels, n) float32. Uses pedalboard (optional `synth` extra). Custom
    specs (`SynthRoomIR`, `NoiseFloor`) are realized here from their recorded params."""
    if not chain:
        return audio
    import pedalboard

    x = audio.astype(np.float32)
    plugins = []

    def flush():
        nonlocal x, plugins
        if plugins:
            x = pedalboard.Pedalboard(plugins)(x, sr)
            plugins = []

    for c in chain:
        name, p = c["plugin"], c["params"]
        if name == "SynthRoomIR":
            plugins.append(pedalboard.Convolution(synth_room_ir(sr, **p), mix=p["mix"], sample_rate=sr))
        elif name == "NoiseFloor":
            flush()
            n = np.random.default_rng(p["seed"]).standard_normal(x.shape).astype(np.float32)
            x = x + n * np.float32(10 ** (p["level_db"] / 20))
        else:
            plugins.append(getattr(pedalboard, name)(**p))
    flush()
    return x
