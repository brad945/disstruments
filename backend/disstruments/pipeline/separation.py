"""Stage A (F6): stem separation via Demucs v4. MPS on Apple Silicon, CUDA, or CPU.

FAKE_ML mode writes band-filtered copies of the mix as stems so the whole app is
exercisable without torch installed (dev/tests/sandbox).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import settings

STEM_NAMES = ["vocals", "drums", "bass", "other"]  # htdemucs 4-stem


def _pick_device() -> str:
    if settings.device != "auto":
        return settings.device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(canonical_wav: Path, out_dir: Path) -> dict:
    """-> {"stems": {name: path}, "model": ..., "version": ..., "device": ..., "gpu_s": float}"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if settings.fake_ml:
        return _fake_separate(canonical_wav, out_dir)

    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio
    from demucs.pretrained import get_model

    device = _pick_device()
    model = get_model(settings.stem_model)
    model.to(device).eval()

    wav = AudioFile(canonical_wav).read(streams=0,
                                        samplerate=model.samplerate,
                                        channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    t0 = time.monotonic()
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, split=True,
                              overlap=0.25, progress=False)[0]
    gpu_s = time.monotonic() - t0

    sources = sources * (ref.std() + 1e-8) + ref.mean()
    stems: dict[str, Path] = {}
    for name, src in zip(model.sources, sources):
        p = out_dir / f"{name}.wav"
        save_audio(src.cpu(), str(p), samplerate=model.samplerate)
        stems[name] = p
    return {"stems": stems, "model": settings.stem_model,
            "version": "demucs-4", "device": device, "gpu_s": gpu_s}


def _fake_separate(canonical_wav: Path, out_dir: Path) -> dict:
    """Deterministic stand-in: crude band-split so stems sound/measure different."""
    data, sr = sf.read(canonical_wav, dtype="float32", always_2d=True)
    n = len(data)
    spectrum = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bands = {"bass": (0, 200), "drums": (200, 2000),
             "vocals": (2000, 6000), "other": (6000, sr / 2)}
    stems: dict[str, Path] = {}
    for name, (lo, hi) in bands.items():
        mask = ((freqs >= lo) & (freqs < hi)).astype("float32")[:, None]
        band = np.fft.irfft(spectrum * mask, n=n, axis=0).astype("float32")
        p = out_dir / f"{name}.wav"
        sf.write(p, band, sr)
        stems[name] = p
    return {"stems": stems, "model": "fake-bandsplit", "version": "0",
            "device": "cpu", "gpu_s": 0.0}
