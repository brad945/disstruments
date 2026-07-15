"""F2: validate + transcode to canonical 44.1kHz stereo wav via ffmpeg. F3: content hash."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from ..config import settings


class TranscodeError(Exception):
    pass


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, check=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise TranscodeError(f"could not read audio file: {e}") from e
    info = json.loads(out.stdout)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise TranscodeError("no audio stream found")
    duration = float(info.get("format", {}).get("duration") or streams[0].get("duration") or 0)
    if duration <= 0:
        raise TranscodeError("could not determine duration")
    if duration > settings.max_duration_s:
        raise TranscodeError(
            f"track is {duration:.0f}s; max is {settings.max_duration_s}s")
    return {"duration_s": duration,
            "codec": streams[0].get("codec_name", "unknown"),
            "sample_rate": int(streams[0].get("sample_rate", 0))}


def to_canonical(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(dst)],
            capture_output=True, check=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise TranscodeError(f"transcode failed: {e.stderr.decode()[:400]}") from e
