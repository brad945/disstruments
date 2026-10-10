"""Shortcut baseline (M4 rigor): can the labels be predicted *without timbre*?

Features deliberately exclude spectral shape (no MFCC / mel / centroid). Only:
- tempo, onset rate, onset-strength mean/std;
- RMS mean/std, dynamic range;
- chroma mean (12 pitch classes: harmony, not timbre);
- pitch register via the lowest/highest active CQT octave band (coarse register only).

A linear model on these is trained on synthetic train and scored on test. If it gets
close to the real model on a leaf, that leaf's labels leak through arrangement, tempo,
register or loudness, and the timbre model's score on it is suspect. The register
features are the probe for the octave-shift giveaway the M4 L3 agent found.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 22050
N_FEATS = 21


def features(path: Path) -> np.ndarray:
    import librosa
    from ..models.embed import load_clip
    y = load_clip(path, SR, 10.0)
    if not np.any(y):
        return np.zeros(N_FEATS, np.float32)
    oenv = librosa.onset.onset_strength(y=y, sr=SR)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=oenv, sr=SR))[0])
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR)
    rms = librosa.feature.rms(y=y)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)
    C = np.abs(librosa.cqt(y, sr=SR, fmin=librosa.note_to_hz("C1"), n_bins=84, bins_per_octave=12))
    octave_energy = C.reshape(7, 12, -1).mean(axis=(1, 2))
    active = np.where(octave_energy > 0.05 * octave_energy.max())[0]
    lo, hi = (float(active.min()), float(active.max())) if len(active) else (0.0, 0.0)
    f = [tempo / 200, len(onsets) / 10.0, float(oenv.mean()), float(oenv.std()),
         float(db.mean()) / 60, float(db.std()) / 20, float(np.percentile(db, 95) - np.percentile(db, 5)) / 40,
         lo / 7, hi / 7, *chroma.tolist()]
    return np.asarray(f[:N_FEATS] + [0.0] * (N_FEATS - len(f)), np.float32)


def build(items, workers: int = 4) -> tuple[list[str], np.ndarray]:
    from concurrent.futures import ProcessPoolExecutor
    ids = [i for i, _ in items]
    with ProcessPoolExecutor(workers) as ex:
        X = np.stack(list(ex.map(features, [p for _, p in items], chunksize=16)))
    return ids, X
