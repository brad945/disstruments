"""Stages B/A1 (F7): instrument tagging with per-second activations.

Real path: PANNs sound-event detection (AudioSet, framewise) with AudioSet-label →
instrument-taxonomy mapping. Taxonomy = OpenMIC-20 (PRD F7). Confidence and
activation intervals propagate to the report (F15 honesty rule).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import settings

# AudioSet class name -> our taxonomy label (subset relevant to rock/indie/pop first)
AUDIOSET_MAP = {
    "Acoustic guitar": "guitar", "Electric guitar": "guitar", "Guitar": "guitar",
    "Bass guitar": "bass",
    "Drum kit": "drums", "Drum": "drums", "Snare drum": "drums", "Bass drum": "drums",
    "Cymbal": "cymbals", "Hi-hat": "cymbals", "Crash cymbal": "cymbals",
    "Piano": "piano", "Electric piano": "piano",
    "Organ": "organ", "Hammond organ": "organ",
    "Synthesizer": "synthesizer",
    "Violin, fiddle": "violin", "Cello": "cello",
    "Trumpet": "trumpet", "Trombone": "trombone",
    "Saxophone": "saxophone", "Clarinet": "clarinet", "Flute": "flute",
    "Accordion": "accordion", "Banjo": "banjo", "Mandolin": "mandolin",
    "Ukulele": "ukulele", "Glockenspiel": "mallet_percussion",
    "Marimba, xylophone": "mallet_percussion",
    "Singing": "voice", "Female singing": "voice", "Male singing": "voice",
    "Choir": "voice", "Speech": "voice",
}

_sed = None  # lazy singleton (model load is slow)
_sed_device = "cpu"


def _get_sed():
    """PANNs SED on `settings.device` (auto -> mps > cuda > cpu), CPU fallback.

    panns_inference only honours device="cuda" and silently forces everything else to
    CPU, so we load on CPU and move the model ourselves. A tiny warm-up inference proves
    the device works; any failure falls back to CPU rather than failing the stage."""
    global _sed, _sed_device
    if _sed is None:
        import torch
        from panns_inference import SoundEventDetection

        from .separation import _pick_device

        sed = SoundEventDetection(checkpoint_path=None, device="cpu")
        device = _pick_device()
        if device != "cpu":
            try:
                sed.model.to(device)
                sed.device = device
                sed.inference(np.zeros((1, 32000), dtype=np.float32))
            except Exception:  # noqa: BLE001 — unsupported op/device: stay correct on CPU
                sed.model.to("cpu")
                sed.device = device = "cpu"
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        _sed, _sed_device = sed, device
    return _sed


def _probs_to_intervals(probs: np.ndarray, hop_s: float, enter: float, exit_: float):
    """Hysteresis thresholding -> [[start, end], ...]"""
    intervals, active, start = [], False, 0.0
    for i, p in enumerate(probs):
        t = i * hop_s
        if not active and p >= enter:
            active, start = True, t
        elif active and p < exit_:
            active = False
            intervals.append([round(start, 2), round(t, 2)])
    if active:
        intervals.append([round(start, 2), round(len(probs) * hop_s, 2)])
    return intervals


def tag(audio_path: Path, source_stem: str = "mix") -> list[dict]:
    """-> [{label, confidence, stem, activations}] sorted by confidence desc."""
    if settings.fake_ml:
        return _fake_tag(audio_path, source_stem)

    import librosa
    from panns_inference.config import labels as audioset_labels

    y, _ = librosa.load(audio_path, sr=32000, mono=True)
    sed = _get_sed()
    framewise = sed.inference(y[None, :])[0]  # (frames, 527); ~0.32 s hop (10ms*32)
    hop_s = (len(y) / 32000) / framewise.shape[0]

    per_label: dict[str, np.ndarray] = {}
    for idx, as_name in enumerate(audioset_labels):
        ours = AUDIOSET_MAP.get(as_name)
        if ours is None:
            continue
        col = framewise[:, idx]
        per_label[ours] = np.maximum(per_label[ours], col) if ours in per_label else col

    results = []
    show = settings.tag_threshold_show
    for label, probs in per_label.items():
        conf = float(np.percentile(probs, 95))
        if conf < show:
            continue
        results.append({
            "label": label,
            "confidence": round(conf, 3),
            "stem": source_stem,
            "activations": _probs_to_intervals(probs, hop_s, enter=show, exit_=show * 0.7),
        })
    return sorted(results, key=lambda r: -r["confidence"])


def _fake_tag(audio_path: Path, source_stem: str) -> list[dict]:
    """Deterministic from file hash so tests are stable."""
    import soundfile as sf
    info = sf.info(audio_path)
    dur = info.frames / info.samplerate
    seed = int(hashlib.sha256(audio_path.name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    pool = {"mix": ["guitar", "drums", "bass", "voice", "piano"],
            "vocals": ["voice"], "drums": ["drums", "cymbals"],
            "bass": ["bass"], "other": ["guitar", "piano", "synthesizer"]}
    out = []
    for label in pool.get(source_stem, pool["mix"]):
        conf = round(float(rng.uniform(0.35, 0.98)), 3)
        start = round(float(rng.uniform(0, dur * 0.2)), 2)
        out.append({"label": label, "confidence": conf, "stem": source_stem,
                    "activations": [[start, round(dur, 2)]]})
    return sorted(out, key=lambda r: -r["confidence"])
