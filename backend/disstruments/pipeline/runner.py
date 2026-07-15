"""Pipeline DAG (design doc §3). Stage failure -> partial report, never total loss (F12).

Draft-1 stages: separation (A) -> per-stem tagging (A1); full-mix tagging (B);
attributes (C). Structure (D) and chords (E) are marked "skipped" in the report
until draft 2 — the schema slot exists so the UI never guesses.
"""
from __future__ import annotations

import datetime as dt
import tempfile
import time
from pathlib import Path

from ..config import settings
from ..db import Analysis, Instrument, Job, Song, StageRun, Stem, get_session
from ..events import bus
from ..jobqueue import record_job_cost
from ..storage import storage
from . import attributes, separation, tagging
from .transcode import to_canonical

SCHEMA_VERSION = 1


def _stage(job_id: int, name: str, model: str, version: str, fn, *args, **kw):
    """Run one stage; record StageRun; publish progress; swallow errors -> (result|None, run)."""
    bus.publish(job_id, {"type": "stage", "stage": name, "status": "running"})
    t0 = time.monotonic()
    run = StageRun(job_id=job_id, stage=name, model_name=model, model_version=version)
    try:
        result = fn(*args, **kw)
        run.status = "ok"
    except Exception as e:  # noqa: BLE001 — graceful degradation is the contract
        result = None
        run.status = "failed"
        run.error = f"{type(e).__name__}: {e}"[:2000]
    run.wall_ms = int((time.monotonic() - t0) * 1000)
    with get_session() as s:
        s.add(run)
        s.commit()
    bus.publish(job_id, {"type": "stage", "stage": name, "status": run.status})
    return result, run


def run_job(job_id: int) -> None:
    with get_session() as s:
        job = s.get(Job, job_id)
        song = s.get(Song, job.song_id)
        job.status = "running"
        s.commit()
    bus.publish(job_id, {"type": "job", "status": "running"})

    original_key = f"songs/{song.id}/original{Path(song.fmt).suffix or '.' + song.fmt}"
    work = Path(tempfile.mkdtemp(prefix=f"diss-job{job_id}-"))
    canonical = work / "canonical.wav"

    failed_any = ok_any = False

    # -- pre: canonical wav ----------------------------------------------------
    res, _ = _stage(job_id, "transcode", "ffmpeg", "system",
                    to_canonical, storage.path(original_key), canonical)
    if not canonical.exists():
        _finish(job_id, song, {}, status="failed")
        return

    # -- A: separation ---------------------------------------------------------
    sep, sep_run = _stage(job_id, "separation", settings.stem_model, "demucs-4",
                          separation.separate, canonical, work / "stems")
    stems_meta, stem_keys = [], {}
    if sep:
        ok_any = True
        record_job_cost(song.user_id, gpu_s=sep["gpu_s"])
        with get_session() as s:
            s.query(StageRun).filter(StageRun.id == sep_run.id).update(
                {"gpu_s": sep["gpu_s"], "model_name": sep["model"],
                 "params_json": {"device": sep["device"]}})
            s.commit()
        for name, path in sep["stems"].items():
            key = f"songs/{song.id}/stems/{name}.wav"
            size = storage.save(key, path)
            record_job_cost(song.user_id, storage_bytes=size)
            stem_keys[name] = key
            stems_meta.append({"name": name, "object_key": key, "bytes": size})
    else:
        failed_any = True

    # -- B: full-mix tagging / A1: per-stem tagging ------------------------------
    instruments: list[dict] = []
    mix_tags, _ = _stage(job_id, "tagging_mix", "panns-sed", "cnn14",
                         tagging.tag, canonical, "mix")
    if mix_tags is not None:
        ok_any = True
    else:
        failed_any = True

    stem_tags_all: list[dict] = []
    if sep:
        for name, path in sep["stems"].items():
            tags, _ = _stage(job_id, f"tagging_{name}", "panns-sed", "cnn14",
                             tagging.tag, path, name)
            if tags:
                stem_tags_all.extend(tags)

    # merge: per-stem wins (it knows *where*); mix fills gaps (design doc §3)
    seen = set()
    for t in sorted(stem_tags_all, key=lambda x: -x["confidence"]):
        if t["label"] not in seen:
            seen.add(t["label"])
            instruments.append(t)
    for t in (mix_tags or []):
        if t["label"] not in seen:
            seen.add(t["label"])
            instruments.append(t)

    # -- C: attributes -----------------------------------------------------------
    attrs, _ = _stage(job_id, "attributes", "librosa+ks", "0.10",
                      attributes.analyze, canonical)
    if attrs is not None:
        ok_any = True
    else:
        failed_any = True

    # -- F: assemble ---------------------------------------------------------------
    with get_session() as s:
        runs = s.query(StageRun).filter(StageRun.job_id == job_id).all()
        provenance = [{"stage": r.stage, "model": r.model_name,
                       "version": r.model_version, "status": r.status} for r in runs]

    report = {
        "schema_version": SCHEMA_VERSION,
        "song": {"id": song.id, "title": song.title, "artist": song.artist,
                 "duration_s": song.duration_s},
        "global": attrs or {"status": "failed"},
        "instruments": instruments,
        "instruments_status": "ok" if (mix_tags is not None or stem_tags_all) else "failed",
        "stems": stems_meta,
        "stems_status": "ok" if sep else "failed",
        "structure": {"status": "skipped"},   # draft 2 (F9)
        "chords": {"status": "skipped"},      # draft 2 (F10)
        "midi": {"status": "skipped"},        # draft 2 (F11)
        "provenance": {"stages": provenance},
    }

    status = "done" if (ok_any and not failed_any) else ("partial" if ok_any else "failed")
    _finish(job_id, song, report, status=status, instruments=instruments, stems=stems_meta)

    if not settings.keep_canonical:  # legal posture: transient processing (design doc §9)
        canonical.unlink(missing_ok=True)


def _finish(job_id: int, song, report: dict, status: str,
            instruments: list | None = None, stems: list | None = None):
    with get_session() as s:
        job = s.get(Job, job_id)
        job.status = status
        job.finished_at = dt.datetime.now(dt.timezone.utc)
        if report:
            analysis = Analysis(song_id=song.id, job_id=job_id,
                                schema_version=SCHEMA_VERSION, report_json=report)
            s.add(analysis)
            s.flush()
            for i in instruments or []:
                s.add(Instrument(analysis_id=analysis.id, song_id=song.id,
                                 label=i["label"], confidence=i["confidence"],
                                 stem=i["stem"]))
            for st in stems or []:
                s.add(Stem(analysis_id=analysis.id, song_id=song.id,
                           name=st["name"], object_key=st["object_key"],
                           bytes=st["bytes"]))
        s.commit()
    bus.publish(job_id, {"type": "job", "status": status})
