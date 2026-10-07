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

from .drum808 import Drum808Engine
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


_KEYRANGE: dict[str, tuple[int, int] | None] = {}


def probe_key_range(binary: str, sfz: Path, sr: int = 22050, cache: Path | None = None,
                    thresh: float = 0.05) -> tuple[int, int] | None:
    """Playable key range of an SFZ, *measured*: render a chromatic sweep (MIDI 21-108,
    one 0.25 s note every 0.35 s) and keep the keys whose window peak reaches `thresh` of
    the loudest note. SFZ files hide regions behind #include / #define / default_path in
    many dialects, so measuring beats parsing. Cached in memory and in `cache` (json)."""
    import json as _json
    import soundfile as sf

    from .midi import Note
    key = str(sfz)
    if key in _KEYRANGE:
        return _KEYRANGE[key]
    disk = {}
    if cache and cache.exists():
        disk = _json.loads(cache.read_text())
        if key in disk:
            _KEYRANGE[key] = tuple(disk[key]) if disk[key] else None
            return _KEYRANGE[key]
    pitches = list(range(21, 109))
    step = 0.35
    notes = [Note(p, k * step, k * step + 0.25, 110) for k, p in enumerate(pitches)]
    with tempfile.TemporaryDirectory() as td:
        mid = notes_to_midi(notes, 0, False, Path(td) / "sweep.mid")
        wav = Path(td) / "sweep.wav"
        subprocess.run([binary, "--sfz", str(sfz), "--midi", str(mid), "--wav", str(wav),
                        "-s", str(sr)], check=True, capture_output=True, timeout=300)
        a, _ = sf.read(str(wav), always_2d=True)
    a = np.abs(a).max(axis=1)
    peaks = np.array([a[int(k * step * sr): int((k * step + 0.25) * sr)].max(initial=0.0)
                      for k in range(len(pitches))])
    ok = [p for p, v in zip(pitches, peaks) if peaks.max() > 0 and v >= thresh * peaks.max()]
    rng_ = (min(ok), max(ok)) if ok else None
    _KEYRANGE[key] = rng_
    if cache:
        disk[key] = list(rng_) if rng_ else None
        cache.write_text(_json.dumps(disk, indent=1))
    return rng_


def fit_octaves(pitches: list[int], lo: int, hi: int) -> int:
    """Octave shift that puts the most notes inside [lo, hi] (ties: smallest |shift|)."""
    best = max(range(-4, 5), key=lambda o: (sum(lo <= p + 12 * o <= hi for p in pitches), -abs(o)))
    return 12 * best


class SfizzEngine:
    name = "sfizz"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("DISS_SFIZZ_RENDER") or shutil.which("sfizz_render") \
            or str(Path.home() / "datasets/disstruments-synth/tools/sfizz/build/library/bin/sfizz_render")

    def render(self, part, source, root, sr, duration, rng):
        from dataclasses import replace
        part = _jitter_velocity(part, rng)
        shift, dropped = 0, 0
        if part.is_drum and source.note_map:          # kit with its own key layout
            notes = [replace(n, pitch=source.note_map[n.pitch]) for n in part.notes
                     if n.pitch in source.note_map]
            dropped = len(part.notes) - len(notes)
            part = replace(part, notes=notes, is_drum=False)   # keys are now literal
        elif not part.is_drum:
            rng_keys = probe_key_range(self.binary, source.file(root),
                                       cache=Path(root) / ".key_ranges.json")
            if rng_keys:
                shift = fit_octaves([n.pitch for n in part.notes], *rng_keys)
        with tempfile.TemporaryDirectory() as td:
            mid = notes_to_midi(part.notes, part.program, part.is_drum, Path(td) / "p.mid",
                                pitch_shift=shift)
            wav = Path(td) / "out.wav"
            subprocess.run([self.binary, "--sfz", str(source.file(root)), "--midi", str(mid),
                            "--wav", str(wav), "-s", str(sr)],
                           check=True, capture_output=True, timeout=120)
            audio = _read_mono(wav, sr)
        return _fit(audio, int(sr * duration)), {"engine": "sfizz", "sfz": source.path,
                                                 "velocity_jitter": 12, "octave_shift": shift // 12,
                                                 "dropped_unmapped_notes": dropped}


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
    """pedalboard-hosted synth with *one plugin instance per leaf*. Surge renames some
    parameters when the oscillator type changes, so restoring an init state by name is
    unreliable; instead each leaf's recipe always sets the same parameter keys on its own
    instance, so patches can never leak across leaves (or across renders of one leaf).
    Patches come from `patches.make_patch`; the plugin's complete parameter dict after
    patching is recorded."""

    def __init__(self, name: str, plugin_path: str):
        self.name, self.plugin_path = name, plugin_path
        self._plugins: dict[str, Any] = {}

    def _load(self, leaf: str):
        if leaf not in self._plugins:
            import pedalboard
            self._plugins[leaf] = pedalboard.load_plugin(self.plugin_path)
        return self._plugins[leaf]

    def render(self, part, source, root, sr, duration, rng):
        import mido

        from .patches import make_patch

        plugin = self._load(part.leaf or "")
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
            "engine": self.name, "patch_recipe": dict(patch), "synth_parameters": params}


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
            "drum808": Drum808Engine(), "fake": FakeEngine()}
