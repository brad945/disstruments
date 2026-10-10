"""Bandwidth-efficient MoisesDB window fetcher — TEST-ONLY eval set (see moisesdb_clips).

MoisesDB ships as one 88.8 GB zip with per-track WAVs and no premixed mixture. For each
song this reads ONE window (default 20-30 s) of every track straight out of the remote
zip with HTTP Range requests, sums the tracks into a mono 24 kHz FLAC and stores per-track
levels for activity labels — without downloading whole stems:

- The zip central directory is pre-extracted to `<root>/index.json`
  ([filename, file_size, compress_size, header_offset, compress_type] per member);
  song metadata to `<root>/meta/<song_uuid>.json`. Local file headers may use a data
  descriptor (sizes 0 in the local header), so sizes always come from the index.
- STORED members: read the local header + the first KB of the member (WAV header,
  RIFF chunks walked properly: JUNK/LIST/bext/... before `fmt `/`data`), then range-read
  exactly the window's PCM bytes.
- DEFLATE members: stream the member from its start through `zlib` (raw deflate) and
  stop as soon as the window's PCM bytes are produced; a dropped connection resumes at
  the current compressed offset with the same decompressor.
- Sample rate / channels / bit depth are checked per track; tracks at another rate are
  resampled (soxr) before summing. Mono tracks are duplicated to stereo, then all tracks
  are summed, then stereo -> mono (mean), then resampled to 24 kHz. If the sum would clip,
  it is scaled down and the gain recorded (`mix.gain_db`).

The presigned URL is read from `<root>/.url` at runtime and never logged: every log line
and error message goes through `Redactor`.

Usage:
  python -m disstruments.ml.datasets.moisesdb_fetch --root ~/datasets/moisesdb --window-start 20
Resumable (songs with both .flac and .json are skipped; .json is written last, atomically).
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence
from urllib.parse import urlsplit

import numpy as np

from .moisesdb_clips import SCHEMA, classify_activity, window_tag

LFH_SIG = 0x04034B50
ZIP_PREFIX = "moisesdb/moisesdb_v0.1"
TARGET_SR = 24000
FRAME_S = 0.05
STORED, DEFLATED = 0, 8
HEAD_PROBE = 16384              # bytes after the local header read to find the WAV header


class FatalFetchError(RuntimeError):
    """Not retryable (expired/forbidden URL, corrupt member)."""


# ---------------------------------------------------------------------- redaction
class Redactor:
    """Removes the URL (whole, path, query, host) from any text before it is logged."""

    def __init__(self, url: str | None):
        parts: list[str] = []
        if url:
            s = urlsplit(url)
            parts = [url, url.split("?")[0], s.query, s.netloc, s.hostname or ""]
            parts += [p for p in s.query.split("&") if len(p) > 8]
        self.parts = sorted({p for p in parts if p and len(p) >= 6}, key=len, reverse=True)

    def __call__(self, text: object) -> str:
        out = str(text)
        for p in self.parts:
            out = out.replace(p, "<redacted>")
        return re.sub(r"https?://\S+", "<url>", out)


# ---------------------------------------------------------------------- range sources
class RangeSource(Protocol):
    bytes_read: int

    def stream(self, start: int, end: int) -> Iterator[bytes]:
        """Yield the bytes [start, end] (inclusive) in chunks; may stop early at EOF."""


class HTTPRangeSource:
    def __init__(self, url: str, timeout: float = 60.0, chunk: int = 65536):
        self.url, self.timeout, self.chunk = url, timeout, chunk
        self.bytes_read = 0
        self._ctx = None
        try:
            import certifi
            self._ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:       # noqa: BLE001 - fall back to the system store
            self._ctx = None

    def stream(self, start: int, end: int) -> Iterator[bytes]:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                raise FatalFetchError(f"HTTP {e.code} (presigned URL expired or invalid?)") from None
            raise OSError(f"HTTP {e.code}") from None
        with r:
            if r.status != 206:
                raise FatalFetchError(f"server ignored Range (status {r.status})")
            while True:
                b = r.read(self.chunk)
                if not b:
                    return
                self.bytes_read += len(b)
                yield b


class FileRangeSource:
    """Local file stand-in (tests, or a locally downloaded zip)."""

    def __init__(self, path: Path | str, chunk: int = 65536):
        self.path, self.chunk = Path(path), chunk
        self.bytes_read = 0

    def stream(self, start: int, end: int) -> Iterator[bytes]:
        with open(self.path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                b = f.read(min(self.chunk, left))
                if not b:
                    return
                left -= len(b)
                self.bytes_read += len(b)
                yield b


def resilient_stream(src: RangeSource, start: int, end: int, *, retries: int = 8,
                     log: Callable[[str], None] = lambda m: None,
                     sleep: Callable[[float], None] = time.sleep) -> Iterator[bytes]:
    """`src.stream` that resumes at the current offset after transient failures."""
    pos, failures = start, 0
    while pos <= end:
        inner = src.stream(pos, end)
        try:
            got_any = False
            for chunk in inner:
                pos += len(chunk)
                got_any = True
                failures = 0
                yield chunk
            return                                   # clean end (or EOF)
        except FatalFetchError:
            raise
        except (OSError, EOFError, http.client.HTTPException) as e:   # timeouts, resets, IncompleteRead
            failures += 1
            if failures > retries:
                raise
            wait = min(60.0, 2.0 ** failures)
            log(f"retry {failures}/{retries} at offset {pos} after {type(e).__name__}: {e} "
                f"(sleep {wait:.0f}s){'' if got_any else ' [no bytes]'}")
            sleep(wait)
        finally:
            inner.close()


class ByteReader:
    """Pull-style reader over a chunk iterator."""

    def __init__(self, it: Iterator[bytes]):
        self.it, self.buf = it, bytearray()

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            try:
                self.buf += next(self.it)
            except StopIteration:
                break
        out, self.buf = bytes(self.buf[:n]), self.buf[n:]
        return out

    def chunks(self) -> Iterator[bytes]:
        if self.buf:
            b, self.buf = bytes(self.buf), bytearray()
            yield b
        yield from self.it

    def close(self) -> None:
        close = getattr(self.it, "close", None)
        if close:
            close()


# ---------------------------------------------------------------------- WAV
@dataclass(frozen=True)
class WavInfo:
    format_tag: int         # 1 = PCM int, 3 = IEEE float (EXTENSIBLE resolved)
    channels: int
    sample_rate: int
    bits: int
    block_align: int
    data_offset: int        # byte offset of the PCM data from the start of the file
    data_size: int

    @property
    def frames(self) -> int:
        return self.data_size // self.block_align


def parse_wav_header(buf: bytes) -> WavInfo | None:
    """Walk RIFF chunks; None if `buf` does not yet contain the `data` chunk header."""
    if len(buf) < 12:
        return None
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise FatalFetchError(f"not a RIFF/WAVE file (magic {buf[:4]!r} {buf[8:12]!r})")
    pos, fmt = 12, None
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4]
        size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            if pos + 8 + min(size, 40) > len(buf):
                return None
            tag, ch, sr, _bps, ba, bits = struct.unpack("<HHIIHH", buf[pos + 8:pos + 24])
            if tag == 0xFFFE and size >= 40:            # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", buf[pos + 32:pos + 34])[0]
            fmt = (tag, ch, sr, bits, ba)
        elif cid == b"data":
            if fmt is None:
                raise FatalFetchError("WAV `data` chunk before `fmt `")
            tag, ch, sr, bits, ba = fmt
            if tag not in (1, 3) or ch < 1 or ba != ch * ((bits + 7) // 8):
                raise FatalFetchError(f"unsupported WAV format tag={tag} ch={ch} bits={bits} ba={ba}")
            return WavInfo(tag, ch, sr, bits, ba, pos + 8, size)
        pos += 8 + size + (size & 1)
    return None


def decode_pcm(raw: bytes, info: WavInfo) -> np.ndarray:
    """PCM bytes -> float32 (channels, frames) in [-1, 1)."""
    n = len(raw) // info.block_align
    raw = raw[:n * info.block_align]
    if info.format_tag == 3:
        x = np.frombuffer(raw, "<f4" if info.bits == 32 else "<f8").astype(np.float32)
    elif info.bits == 8:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif info.bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif info.bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif info.bits == 32:
        x = (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise FatalFetchError(f"unsupported bit depth {info.bits}")
    return x.reshape(n, info.channels).T.copy()


# ---------------------------------------------------------------------- member windows
@dataclass
class TrackWindow:
    audio: np.ndarray        # float32 (channels, frames) at info.sample_rate
    info: WavInfo
    frames_available: int    # < requested if the file ends inside the window
    bytes_read: int          # bytes pulled from the source for this member


def _local_header(reader: ByteReader, name: str) -> int:
    """Consume the local file header; return its total length."""
    h = reader.read(30)
    if len(h) < 30:
        raise OSError("short read in local file header")
    sig, _ver, _flag, _meth, _t, _d, _crc, _cs, _us, nl, el = struct.unpack("<IHHHHHIIIHH", h)
    if sig != LFH_SIG:
        raise FatalFetchError("bad local file header signature (index offset wrong?)")
    fn = reader.read(nl)
    if fn.decode("utf-8", "replace") != name:
        raise FatalFetchError("local header filename != index filename")
    if len(reader.read(el)) < el:
        raise OSError("short read in local header extra field")
    return 30 + nl + el


def read_member_window(src: RangeSource, entry: Sequence[Any], start_s: float, dur_s: float, *,
                       log: Callable[[str], None] = lambda m: None,
                       sleep: Callable[[float], None] = time.sleep) -> TrackWindow:
    """Read [start_s, start_s + dur_s) of one WAV zip member, downloading as little as possible."""
    name, _fsize, csize, hoff, ctype = entry[:5]
    got = [0]                                   # bytes pulled for THIS member (thread-safe)

    def fetch(a: int, b: int) -> Iterator[bytes]:
        inner = resilient_stream(src, a, b, log=log, sleep=sleep)
        try:
            for chunk in inner:
                got[0] += len(chunk)
                yield chunk
        finally:
            inner.close()
    name_len = len(name.encode("utf-8"))
    if ctype == STORED:
        head_end = hoff + 30 + name_len + 1024 + HEAD_PROBE - 1
        reader = ByteReader(fetch(hoff, head_end))
        try:
            hlen = _local_header(reader, name)
            head = reader.read(head_end - hoff + 1 - hlen)
        finally:
            reader.close()
        info = parse_wav_header(head)
        if info is None:
            raise FatalFetchError(f"WAV header not within the first {len(head)} bytes")
        data_start = hoff + hlen
        sf, nf = round(start_s * info.sample_rate), round(dur_s * info.sample_rate)
        avail = max(0, min(nf, info.frames - sf))
        if avail == 0:
            raw = b""
        else:
            a = data_start + info.data_offset + sf * info.block_align
            b = a + avail * info.block_align - 1
            raw = b"".join(fetch(a, b))
            if len(raw) != avail * info.block_align:
                raise OSError(f"short read: {len(raw)} of {avail * info.block_align} PCM bytes")
    elif ctype == DEFLATED:
        end = hoff + 30 + name_len + 65535 + csize - 1        # extra-field length not yet known
        reader = ByteReader(fetch(hoff, end))
        try:
            hlen = _local_header(reader, name)
            budget = csize                                     # compressed bytes in this member
            d = zlib.decompressobj(-15)
            out = bytearray()
            info, need = None, None
            for chunk in reader.chunks():
                if budget <= 0:
                    break
                chunk = chunk[:budget]
                budget -= len(chunk)
                out += d.decompress(chunk)
                if info is None:
                    info = parse_wav_header(bytes(out[:1 << 20]))
                    if info is not None:
                        sf = round(start_s * info.sample_rate)
                        nf = round(dur_s * info.sample_rate)
                        avail = max(0, min(nf, info.frames - sf))
                        need = info.data_offset + (sf + avail) * info.block_align
                if need is not None and len(out) >= need:
                    break
                if d.eof:
                    break
        finally:
            reader.close()
        if info is None:
            raise OSError("stream ended before the WAV header")
        if len(out) < need:
            if d.eof or budget <= 0:
                raise FatalFetchError("member ended before its declared data size")
            raise OSError("stream ended early")
        a = info.data_offset + sf * info.block_align
        raw = bytes(out[a:a + avail * info.block_align])
    else:
        raise FatalFetchError(f"unsupported zip compression {ctype}")
    audio = decode_pcm(raw, info)
    if avail < nf:
        audio = np.pad(audio, ((0, 0), (0, nf - avail)))
    return TrackWindow(audio, info, avail, got[0])


# ---------------------------------------------------------------------- mixing & levels
def db(x: float) -> float | None:
    return None if x <= 0 else round(10.0 * math.log10(x), 2)


def levels(audio: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    """(window RMS dBFS, max 50-ms-frame RMS dBFS) over all channels."""
    sq = np.asarray(audio, np.float64) ** 2
    rms = db(float(sq.mean())) if sq.size else None
    hop = max(1, int(round(FRAME_S * sr)))
    n = sq.shape[-1] // hop
    peak = db(float(sq[..., :n * hop].reshape(sq.shape[0], n, hop).mean(axis=(0, 2)).max())) if n else rms
    return rms, peak


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    if x.shape[0] == 2:
        return x
    raise FatalFetchError(f"{x.shape[0]}-channel track not supported")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr
    y = soxr.resample(np.ascontiguousarray(x.T, dtype=np.float32), sr_in, sr_out, quality="HQ")
    return np.ascontiguousarray(np.atleast_2d(y.T if y.ndim == 2 else y[None, :]), dtype=np.float32)


def mix_tracks(windows: Sequence[TrackWindow], dur_s: float, ref_sr: int = 44100
               ) -> tuple[np.ndarray, int]:
    """Sum all tracks (stereo, at ref_sr), stereo -> mono, resample to TARGET_SR."""
    n = round(dur_s * ref_sr)
    acc = np.zeros((2, n), np.float64)
    for w in windows:
        x = resample(to_stereo(w.audio), w.info.sample_rate, ref_sr)
        m = min(n, x.shape[1])
        acc[:, :m] += x[:, :m]
    mono = acc.mean(axis=0, keepdims=True).astype(np.float32)
    return resample(mono, ref_sr, TARGET_SR)[0], ref_sr


# ---------------------------------------------------------------------- per song
def zip_path(song_id: str, stem_name: str, track_id: str) -> str:
    return f"{ZIP_PREFIX}/{song_id}/{stem_name}/{track_id}.wav"


def song_tracks(meta: dict) -> list[dict]:
    return [{"track_id": t["id"], "trackType": t["trackType"], "stemName": s["stemName"],
             "has_bleed": bool(t.get("has_bleed", False))}
            for s in meta["stems"] for t in s["tracks"]]


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.name + ".part")
    write(tmp)
    os.replace(tmp, path)


def process_song(src: RangeSource, song_id: str, meta: dict, index: dict[str, Sequence[Any]],
                 out_dir: Path, start_s: float, dur_s: float, *, workers: int = 3,
                 log: Callable[[str], None] = lambda m: None,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    import soundfile as sf

    tracks = song_tracks(meta)
    if not tracks:
        raise FatalFetchError("song has no tracks")
    missing = [t["track_id"] for t in tracks
               if zip_path(song_id, t["stemName"], t["track_id"]) not in index]
    if missing:
        raise FatalFetchError(f"{len(missing)} track(s) not in the zip index: {missing[:3]}")
    entries = [index[zip_path(song_id, t["stemName"], t["track_id"])] for t in tracks]
    t0 = time.time()

    def one(e: Sequence[Any]) -> TrackWindow:
        return read_member_window(src, e, start_s, dur_s, log=log, sleep=sleep)

    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            wins = list(ex.map(one, entries))
    else:
        wins = [one(e) for e in entries]
    srs = [w.info.sample_rate for w in wins]
    ref_sr = max(set(srs), key=srs.count)
    mono, _ = mix_tracks(wins, dur_s, ref_sr)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    tag = window_tag(start_s)
    flac = out_dir / f"{song_id}_{tag}.flac"
    _atomic_write(flac, lambda p: sf.write(str(p), (mono * gain).astype(np.float32), TARGET_SR,
                                           format="FLAC", subtype="PCM_24"))
    lv = [levels(w.audio, w.info.sample_rate) for w in wins]
    states = classify_activity(lv)
    total_bytes = sum(w.bytes_read for w in wins)
    doc = {
        "schema": SCHEMA, "song_id": song_id, "song": meta.get("song"), "artist": meta.get("artist"),
        "genre": meta.get("genre"),
        "window": {"start_s": float(start_s), "dur_s": float(dur_s)},
        "mix": {"path": flac.name, "sr": TARGET_SR, "channels": 1, "summed_at_sr": ref_sr,
                "peak_before_gain": round(peak, 5), "gain_db": round(20 * math.log10(gain), 3)},
        "activity_thresholds_at_fetch": "see moisesdb_clips.ABS_FLOOR_DB/REL_DB/SILENCE_DB",
        "tracks": [dict(t, rms_db=r, peak_frame_db=p, activity=s, active=(s == "active"),
                        sample_rate=w.info.sample_rate, channels=w.info.channels, bits=w.info.bits,
                        float=w.info.format_tag == 3, frames_available=w.frames_available,
                        bytes_read=w.bytes_read, compress_type=e[4], compress_size=e[2],
                        file_size=e[1])
                   for t, (r, p), s, w, e in zip(tracks, lv, states, wins, entries)],
        "bytes_downloaded": total_bytes,
        "seconds": round(time.time() - t0, 1),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_write(out_dir / f"{song_id}_{tag}.json",
                  lambda p: p.write_text(json.dumps(doc, indent=1)))
    return doc


# ---------------------------------------------------------------------- driver
def artist_round_robin(metas: dict[str, dict]) -> list[str]:
    """Song order cycling through artists, so a partial run covers many artists."""
    by: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(metas):
        by[str(metas[sid].get("artist", "")).casefold()].append(sid)
    queues = [by[a] for a in sorted(by, key=lambda a: (-len(by[a]), a))]
    order: list[str] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def load_index(root: Path) -> dict[str, Sequence[Any]]:
    return {e[0]: e for e in json.loads((root / "index.json").read_text()) if e[0].endswith(".wav")}


def load_metas(root: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((root / "meta").glob("*.json"))}


def run(root: Path, src: RangeSource, *, start_s: float = 20.0, dur_s: float = 10.0,
        workers: int = 3, limit: int | None = None, songs: Sequence[str] | None = None,
        stop_if_mb_per_song: float | None = None, min_free_gb: float = 40.0,
        redact: Callable[[object], str] = str, sleep: Callable[[float], None] = time.sleep,
        echo: bool = True) -> dict:
    root = Path(root)
    out_dir = root / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(root / "fetch.log", "a")

    def log(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {redact(msg)}"
        logf.write(line + "\n")
        logf.flush()
        if echo:
            print(line, flush=True)

    index, metas = load_index(root), load_metas(root)
    tag = window_tag(start_s)
    order = list(songs) if songs else artist_round_robin(metas)

    def done(sid: str) -> bool:
        return (out_dir / f"{sid}_{tag}.json").exists() and (out_dir / f"{sid}_{tag}.flac").exists()

    prev = [json.loads((out_dir / f"{s}_{tag}.json").read_text()) for s in metas if done(s)]
    total_prev = sum(d.get("bytes_downloaded", 0) for d in prev)
    todo = [s for s in order if not done(s)]
    if limit is not None:
        todo = todo[:limit]
    log(f"start: window {start_s}+{dur_s}s, {len(prev)}/{len(metas)} songs done "
        f"({total_prev / 1e6:.1f} MB so far), {len(todo)} to fetch this run, workers={workers}")
    run_bytes, run_secs, ok, failed = 0, 0.0, 0, []
    for i, sid in enumerate(todo, 1):
        free_gb = _free_gb(root)
        if free_gb < min_free_gb:
            log(f"STOP: only {free_gb:.1f} GB free (< {min_free_gb})")
            break
        t0 = time.time()
        try:
            doc = process_song(src, sid, metas[sid], index, out_dir, start_s, dur_s,
                               workers=workers, log=log, sleep=sleep)
        except FatalFetchError as e:
            log(f"FATAL {sid}: {e}")
            failed.append(sid)
            if "URL" in str(e) or "Range" in str(e):
                break
            continue
        except Exception as e:      # noqa: BLE001 - keep going; the song is retried next run
            log(f"FAILED {sid}: {type(e).__name__}: {e}")
            failed.append(sid)
            continue
        dt = time.time() - t0
        ok += 1
        run_bytes += doc["bytes_downloaded"]
        run_secs += dt
        mbps = run_bytes / 1e6 / max(run_secs, 1e-9)
        per_song = run_bytes / 1e6 / ok
        remaining = len(todo) - i
        eta_h = remaining * (run_secs / ok) / 3600
        n_act = sum(t["active"] for t in doc["tracks"])
        log(f"[{len(prev) + ok}/{len(metas)}] {sid} {doc['artist']!s:.24} | {len(doc['tracks'])} trk "
            f"({n_act} active) | {doc['bytes_downloaded'] / 1e6:.1f} MB in {dt:.0f}s | run avg "
            f"{per_song:.1f} MB/song {mbps:.2f} MB/s | total {(total_prev + run_bytes) / 1e6:.0f} MB "
            f"| ETA {eta_h:.1f} h for {remaining}")
        if stop_if_mb_per_song and ok >= 3 and per_song > stop_if_mb_per_song:
            log(f"STOP: {per_song:.1f} MB/song > --stop-if-mb-per-song {stop_if_mb_per_song}")
            break
    log(f"end: {ok} fetched, {len(failed)} failed {failed[:10]}, {run_bytes / 1e6:.1f} MB this run, "
        f"{(total_prev + run_bytes) / 1e6:.1f} MB total")
    logf.close()
    return {"ok": ok, "failed": failed, "run_bytes": run_bytes, "total_bytes": total_prev + run_bytes}


def _free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, required=True, help="dir with .url, index.json, meta/")
    p.add_argument("--window-start", type=float, default=20.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=3, help="parallel track downloads per song")
    p.add_argument("--limit", type=int, help="fetch at most N songs this run")
    p.add_argument("--songs", nargs="*", help="only these song uuids")
    p.add_argument("--stop-if-mb-per-song", type=float,
                   help="stop after >= 3 songs if the run average exceeds this")
    p.add_argument("--min-free-gb", type=float, default=40.0)
    a = p.parse_args(argv)
    root = a.root.expanduser()
    url = (root / ".url").read_text().strip()
    redact = Redactor(url)
    try:
        res = run(root, HTTPRangeSource(url), start_s=a.window_start, dur_s=a.duration,
                  workers=a.workers, limit=a.limit, songs=a.songs,
                  stop_if_mb_per_song=a.stop_if_mb_per_song, min_free_gb=a.min_free_gb,
                  redact=redact)
    except BaseException as e:      # never let a traceback print the URL
        print(redact(f"aborted: {type(e).__name__}: {e}"), file=sys.stderr)
        return 1
    return 0 if not res["failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
