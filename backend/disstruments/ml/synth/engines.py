"""Render engines (M2 S3): turn one `Part` into mono audio through one `Source`.

- `sfizz`: SFZ sample libraries via the `sfizz_render` CLI (built from source, see
  docs/debriefs/m2.md); path from `DISS_SFIZZ_RENDER`.
- `fluidsynth`: SF2 GM banks via the `fluidsynth` CLI (program chosen per leaf).
- `surge_xt` / `dexed`: VST3 synths hosted by pedalboard, with *our own* randomized patches
  (`patches.py`); the full parameter dict is returned for provenance.
- `fake`: deterministic tones per leaf; used by tests and dry runs (no external tools).

Every engine returns (audio[n] float32 at `sr`, meta dict).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .midi import Part, notes_to_midi
from .sources import Source


class Engine(Protocol):
    name: str

    def render(self, part: Part, source: Source, root: Path, sr: int, duration: float,
               rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]: ...


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    a = audio.astype(np.float32).reshape(-1)
    return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))


def _read_mono(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        import librosa
        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
    return mono


def _jitter_velocity(part: Part, rng: np.random.Generator, spread: int = 12) -> Part:
    from dataclasses import replace
    notes = [replace(n, velocity=int(np.clip(n.velocity + rng.integers(-spread, spread + 1), 1, 127)))
             for n in part.notes]
    return replace(part, notes=notes)


class FakeEngine:
    """Harmonic tone per note; timbre (harmonic rolloff) is a hash of the leaf."""
    name = "fake"

    def render(self, part, source, root, sr, duration, rng):
        n = int(sr * duration)
        out = np.zeros(n, np.float32)
        k = (zlib.crc32((part.leaf or "").encode()) % 7) + 1
        for note in part.notes:
            a, b = int(note.start * sr), min(n, int(note.end * sr))
            if b <= a:
                continue
            t = np.arange(b - a) / sr
            f = 440.0 * 2 ** ((note.pitch - 69) / 12)
            tone = sum(np.sin(2 * np.pi * f * h * t) / h ** (1 + k / 4) for h in range(1, 6))
            out[a:b] += (note.velocity / 127) * 0.2 * tone * np.exp(-t * (1 + k))
        return out, {"engine": "fake", "timbre_k": k}


class SfizzEngine:
    name = "sfizz"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("DISS_SFIZZ_RENDER") or shutil.which("sfizz_render") \
            or str(Path.home() / "datasets/disstruments-synth/tools/sfizz/build/library/bin/sfizz_render")

    def render(self, part, source, root, sr, duration, rng):
        part = _jitter_velocity(part, rng)
        with tempfile.TemporaryDirectory() as td:
            mid = notes_to_midi(part.notes, part.program, part.is_drum, Path(td) / "p.mid")
            wav = Path(td) / "out.wav"
            subprocess.run([self.binary, "--sfz", str(source.file(root)), "--midi", str(mid),
                            "--wav", str(wav), "-s", str(sr)],
                           check=True, capture_output=True, timeout=120)
            audio = _read_mono(wav, sr)
        return _fit(audio, int(sr * duration)), {"engine": "sfizz", "sfz": source.path,
                                                 "velocity_jitter": 12}


class FluidSynthEngine:
    name = "fluidsynth"

    def __init__(self, binary: str | None = None):
        self.binary = binary or shutil.which("fluidsynth") or "fluidsynth"

    def render(self, part, source, root, sr, duration, rng):
        part = _jitter_velocity(part, rng)
        progs = source.programs.get(part.leaf or "", ())
        program = int(rng.choice(progs)) if progs else part.program
        with tempfile.TemporaryDirectory() as td:
            mid = notes_to_midi(part.notes, program, part.is_drum, Path(td) / "p.mid")
            wav = Path(td) / "out.wav"
            subprocess.run([self.binary, "-ni", "-q", "-R", "0", "-C", "0", "-r", str(sr),
                            "-F", str(wav), str(source.file(root)), str(mid)],
                           check=True, capture_output=True, timeout=120)
            audio = _read_mono(wav, sr)
        return _fit(audio, int(sr * duration)), {"engine": "fluidsynth", "sf2": source.path,
                                                 "gm_program": program, "velocity_jitter": 12}


class VST3Engine:
    """pedalboard-hosted synth. Patches come from `patches.make_patch(engine, leaf, rng)`;
    the plugin's complete parameter dict after patching is recorded."""

    def __init__(self, name: str, plugin_path: str):
        self.name, self.plugin_path = name, plugin_path
        self._plugin = None

    def _load(self):
        if self._plugin is None:
            import pedalboard
            self._plugin = pedalboard.load_plugin(self.plugin_path)
        return self._plugin

    def render(self, part, source, root, sr, duration, rng):
        import mido

        from .patches import make_patch

        plugin = self._load()
        plugin.reset()
        patch = make_patch(self.name, part.leaf or "", rng, plugin)
        pitch_shift = int(patch.get("_transpose", 0))
        msgs = []
        for n in _jitter_velocity(part, rng).notes:
            p = int(np.clip(n.pitch + pitch_shift, 0, 127))
            msgs.append(mido.Message("note_on", note=p, velocity=n.velocity, time=n.start))
            msgs.append(mido.Message("note_off", note=p, velocity=0, time=n.end))
        audio = plugin(msgs, duration=duration, sample_rate=sr, num_channels=2)
        params = {k: _param_value(v) for k, v in plugin.parameters.items()}
        return _fit(np.asarray(audio).mean(axis=0), int(sr * duration)), {
            "engine": self.name, "patch_recipe": {k: v for k, v in patch.items()},
            "synth_parameters": params}


def _param_value(p) -> Any:
    try:
        return round(float(p.raw_value), 6)
    except Exception:  # noqa: BLE001
        return str(getattr(p, "string_value", p))


def default_engines() -> dict[str, Engine]:
    plug = Path.home() / "Library/Audio/Plug-Ins/VST3"
    return {"sfizz": SfizzEngine(), "fluidsynth": FluidSynthEngine(),
            "surge_xt": VST3Engine("surge_xt", os.environ.get("DISS_SURGE_VST3", str(plug / "Surge XT.vst3"))),
            "dexed": VST3Engine("dexed", os.environ.get("DISS_DEXED_VST3", str(plug / "Dexed.vst3"))),
            "fake": FakeEngine()}
