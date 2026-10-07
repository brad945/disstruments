"""TR-808-style drum machine, synthesized in numpy (leaf `drums.electronic.tr808`).

No samples, so it's licence-clean by construction, and every voice parameter is recorded.
The design follows the analog 808 circuit topology at the level that matters for timbre:
- kick: a decaying sine with a short pitch sweep and a click
- snare: two detuned tones + filtered noise
- clap: 3–4 noise bursts + a tail
- hats/cymbals: six detuned square oscillators through a high-pass, short (closed) or
  long (open/cymbal) decay
- toms: sine pitch drops
- cowbell: two squares (~540/800 Hz)

GM drum-map notes route to voices; unmapped notes are skipped (and counted).
"""
from __future__ import annotations

from typing import Any

import numpy as np

VOICE_OF_NOTE = {35: "kick", 36: "kick", 37: "rim", 38: "snare", 40: "snare", 39: "clap",
                 42: "hat_closed", 44: "hat_closed", 46: "hat_open", 41: "tom_low", 43: "tom_low",
                 45: "tom_mid", 47: "tom_mid", 48: "tom_high", 50: "tom_high", 49: "cymbal",
                 57: "cymbal", 51: "cymbal", 59: "cymbal", 56: "cowbell", 54: "hat_closed"}
_HAT_FREQS = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0])   # 808 metal oscillators


def _u(rng, lo, hi):
    return round(float(rng.uniform(lo, hi)), 4)


def kit_params(rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """One randomized kit per clip (a 'patch')."""
    return {
        "kick": {"f0": _u(rng, 42, 62), "sweep": _u(rng, 1.5, 4.0), "sweep_t": _u(rng, 0.01, 0.05),
                 "decay": _u(rng, 0.25, 1.4), "click": _u(rng, 0.0, 0.4), "drive": _u(rng, 1.0, 3.0)},
        "snare": {"f1": _u(rng, 160, 240), "f2": _u(rng, 300, 380), "tone_decay": _u(rng, 0.05, 0.15),
                  "noise_decay": _u(rng, 0.08, 0.3), "snappy": _u(rng, 0.3, 0.9), "bp": _u(rng, 2000, 6000)},
        "clap": {"bursts": float(rng.integers(3, 5)), "spacing": _u(rng, 0.008, 0.015),
                 "tail": _u(rng, 0.1, 0.35), "bp": _u(rng, 900, 1600)},
        "hat_closed": {"decay": _u(rng, 0.03, 0.09), "hp": _u(rng, 6000, 9000), "detune": _u(rng, 0.97, 1.03)},
        "hat_open": {"decay": _u(rng, 0.2, 0.6), "hp": _u(rng, 6000, 9000), "detune": _u(rng, 0.97, 1.03)},
        "cymbal": {"decay": _u(rng, 0.8, 2.5), "hp": _u(rng, 4000, 7000), "detune": _u(rng, 0.97, 1.03)},
        "tom_low": {"f0": _u(rng, 80, 110), "decay": _u(rng, 0.2, 0.5), "sweep": _u(rng, 1.2, 2.0)},
        "tom_mid": {"f0": _u(rng, 120, 160), "decay": _u(rng, 0.18, 0.4), "sweep": _u(rng, 1.2, 2.0)},
        "tom_high": {"f0": _u(rng, 170, 230), "decay": _u(rng, 0.15, 0.35), "sweep": _u(rng, 1.2, 2.0)},
        "cowbell": {"f1": _u(rng, 520, 560), "f2": _u(rng, 780, 820), "decay": _u(rng, 0.1, 0.4)},
        "rim": {"f0": _u(rng, 1600, 2000), "decay": _u(rng, 0.01, 0.03)},
    }


def _env(n: int, sr: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (sr * max(decay, 1e-3) / 5.0)).astype(np.float32)


def _bp_noise(rng, n, sr, center, q=1.0):
    import scipy.signal as ss
    noise = rng.standard_normal(n).astype(np.float32)
    lo, hi = center / (1 + 0.5 / q), min(center * (1 + 0.5 / q), sr / 2 - 100)
    sos = ss.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
    return ss.sosfilt(sos, noise).astype(np.float32)


def _hp(x, sr, f):
    import scipy.signal as ss
    sos = ss.butter(2, min(f, sr / 2 - 100), btype="high", fs=sr, output="sos")
    return ss.sosfilt(sos, x).astype(np.float32)


def voice(name: str, p: dict[str, float], vel: float, sr: int, rng) -> np.ndarray:
    if name == "kick":
        n = int(sr * p["decay"] * 1.2)
        t = np.arange(n) / sr
        f = p["f0"] * (1 + (p["sweep"] - 1) * np.exp(-t / p["sweep_t"]))
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, sr, p["decay"])
        x[: int(sr * 0.003)] += p["click"] * rng.standard_normal(int(sr * 0.003))
        x = np.tanh(p["drive"] * x) / np.tanh(p["drive"])
    elif name in ("snare", "rim"):
        if name == "rim":
            n = int(sr * 0.05)
            t = np.arange(n) / sr
            x = np.sin(2 * np.pi * p["f0"] * t) * _env(n, sr, p["decay"])
        else:
            n = int(sr * max(p["noise_decay"], p["tone_decay"]) * 1.3)
            t = np.arange(n) / sr
            tone = (np.sin(2 * np.pi * p["f1"] * t) + 0.6 * np.sin(2 * np.pi * p["f2"] * t)) * _env(n, sr, p["tone_decay"])
            x = (1 - p["snappy"]) * tone + p["snappy"] * _bp_noise(rng, n, sr, p["bp"], 0.7) * _env(n, sr, p["noise_decay"]) * 3
    elif name == "clap":
        n = int(sr * (p["tail"] + 0.06))
        x = np.zeros(n, np.float32)
        nb = _bp_noise(rng, n, sr, p["bp"], 1.2) * 3
        for k in range(int(p["bursts"])):
            a = int(k * p["spacing"] * sr)
            m = min(n - a, int(sr * 0.012))
            x[a:a + m] += nb[a:a + m] * _env(m, sr, 0.012)
        a = int(p["bursts"] * p["spacing"] * sr)
        x[a:] += nb[a:] * _env(n - a, sr, p["tail"]) * 0.7
    elif name.startswith(("hat", "cymbal")):
        n = int(sr * p["decay"] * 1.2)
        t = np.arange(n) / sr
        sq = np.sign(np.sin(2 * np.pi * (_HAT_FREQS * p["detune"])[:, None] * t[None, :])).sum(0) / 6
        x = _hp(sq.astype(np.float32), sr, p["hp"]) * _env(n, sr, p["decay"]) * 2
    elif name.startswith("tom"):
        n = int(sr * p["decay"] * 1.2)
        t = np.arange(n) / sr
        f = p["f0"] * (1 + (p["sweep"] - 1) * np.exp(-t / 0.02))
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, sr, p["decay"])
    elif name == "cowbell":
        n = int(sr * p["decay"] * 1.2)
        t = np.arange(n) / sr
        x = (np.sign(np.sin(2 * np.pi * p["f1"] * t)) + np.sign(np.sin(2 * np.pi * p["f2"] * t))) * 0.3
        x = x * _env(n, sr, p["decay"])
    else:
        return np.zeros(1, np.float32)
    x = np.asarray(x, np.float32)
    f = min(len(x), int(sr * 0.005))                  # 5 ms fade: no click at the cut
    if f > 1:
        x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return vel * x


class Drum808Engine:
    name = "drum808"

    def render(self, part, source, root, sr, duration, rng):
        n = int(sr * duration)
        out = np.zeros(n, np.float32)
        kit = kit_params(rng)
        skipped = 0
        for note in part.notes:
            v = VOICE_OF_NOTE.get(note.pitch)
            if v is None:
                skipped += 1
                continue
            x = voice(v, kit[v], note.velocity / 127, sr, rng)
            a = int(note.start * sr)
            m = min(len(x), n - a)
            if m > 0:
                out[a:a + m] += x[:m]
        out *= 0.3
        return out, {"engine": "drum808", "kit": kit, "skipped_notes": skipped}
