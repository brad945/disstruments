"""Stage C (F8): key, BPM, LUFS. librosa + Krumhansl-Schmuckler; pyloudnorm for LUFS.

Deliberately avoids Essentia (AGPL — see design doc §4 license note).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _detect_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    scores = []
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        scores.append(("major", shift, float(np.corrcoef(rolled, _MAJOR)[0, 1])))
        scores.append(("minor", shift, float(np.corrcoef(rolled, _MINOR)[0, 1])))
    scores.sort(key=lambda t: -t[2])
    mode, shift, best = scores[0]
    second = scores[1][2]
    # confidence: winner's margin over runner-up, squashed to (0,1)
    conf = float(np.clip(0.5 + (best - second) * 4, 0.05, 0.99))
    return f"{_NOTES[shift]} {mode}", round(conf, 2)


def analyze(canonical_wav: Path) -> dict:
    from ..config import settings
    if settings.fake_ml:
        return {"key": {"value": "F# minor", "confidence": 0.8},
                "bpm": {"value": 128.0, "confidence": 0.9},
                "lufs": -9.0, "time_signature": "4/4"}

    import librosa
    import pyloudnorm
    import soundfile as sf

    y, sr = librosa.load(canonical_wav, sr=22050, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    # crude confidence: consistency of inter-beat intervals
    if len(beats) > 3:
        ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
        bpm_conf = float(np.clip(1.0 - np.std(ibi) / (np.mean(ibi) + 1e-9), 0.05, 0.99))
    else:
        bpm_conf = 0.2

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, key_conf = _detect_key(chroma.mean(axis=1))

    data, srate = sf.read(canonical_wav, always_2d=True)
    lufs = float(pyloudnorm.Meter(srate).integrated_loudness(data))

    return {"key": {"value": key, "confidence": key_conf},
            "bpm": {"value": round(tempo, 1), "confidence": round(bpm_conf, 2)},
            "lufs": round(lufs, 1),
            "time_signature": "4/4"}  # real detection is Phase-2; labeled as assumed in UI
