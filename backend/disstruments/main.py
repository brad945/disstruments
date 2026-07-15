"""FastAPI app — routes per design doc §6 (/api/v1)."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import settings
from .db import (LOCAL_USER_ID, Analysis, GenreTag, Job, Song, StageRun, Stem,
                 get_session, init_db)
from .events import bus
from .jobqueue import AdmissionError, InProcessQueue
from .pipeline.runner import run_job
from .pipeline.transcode import TranscodeError, content_hash, probe
from .ratelimit import rate_limit_middleware
from .storage import init_storage
from . import storage as storage_mod

queue: InProcessQueue | None = None


def create_app() -> FastAPI:
    global queue
    init_db()
    init_storage()
    app = FastAPI(title="Disstruments", version="0.1.0")
    app.middleware("http")(rate_limit_middleware)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                       allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining",
                                       "X-RateLimit-Reset", "Retry-After"])
    queue = InProcessQueue(run_job)
    queue.start()

    @app.on_event("startup")
    async def _bind_loop():
        bus.bind_loop(asyncio.get_running_loop())

    # ---------------- songs ----------------

    @app.post("/api/v1/songs")
    async def upload_song(file: UploadFile, title: str = Form(""),
                          artist: str = Form(""), genres: str = Form("")):
        if file.size and file.size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"max upload is {settings.max_upload_mb} MB")

        import os
        fd, tmp_name = tempfile.mkstemp(suffix=Path(file.filename or "u").suffix)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(file.file, f)
            if tmp.stat().st_size > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"max upload is {settings.max_upload_mb} MB")
            try:
                meta = probe(tmp)          # F2 validate
            except TranscodeError as e:
                raise HTTPException(422, str(e))
            chash = content_hash(tmp)      # F3 dedup

            with get_session() as s:
                existing = s.query(Song).filter(Song.content_hash == chash).one_or_none()
                if existing:
                    job = (s.query(Job).filter(Job.song_id == existing.id)
                           .order_by(Job.id.desc()).first())
                    return {"song_id": existing.id, "job_id": job.id if job else None,
                            "deduplicated": True}

                song = Song(user_id=LOCAL_USER_ID, content_hash=chash,
                            title=title or Path(file.filename or "untitled").stem,
                            artist=artist, duration_s=meta["duration_s"],
                            fmt=Path(file.filename or "").suffix.lstrip(".") or meta["codec"])
                s.add(song)
                s.flush()
                for tag in [g.strip().lower() for g in genres.split(",") if g.strip()]:
                    s.add(GenreTag(song_id=song.id, tag=tag))
                job = Job(song_id=song.id)
                s.add(job)
                s.commit()
                song_id, job_id = song.id, job.id

            key = f"songs/{song_id}/original{tmp.suffix or '.bin'}"
            storage_mod.storage.save(key, tmp)

            try:
                queue.submit(job_id, LOCAL_USER_ID)
            except AdmissionError as e:
                resp = JSONResponse(status_code=429,
                                    content={"code": e.code, "retry_after_s": e.retry_after_s})
                resp.headers["Retry-After"] = str(e.retry_after_s)
                return resp
            return {"song_id": song_id, "job_id": job_id, "deduplicated": False}
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/api/v1/songs")
    def list_songs(instrument: str | None = None, genre: str | None = None,
                   q: str | None = None):
        from .db import Instrument as InstrumentRow
        with get_session() as s:
            query = s.query(Song)
            if q:
                query = query.filter(Song.title.ilike(f"%{q}%"))
            if genre:
                ids = [g.song_id for g in s.query(GenreTag).filter(GenreTag.tag == genre.lower())]
                query = query.filter(Song.id.in_(ids))
            if instrument:
                ids = [i.song_id for i in s.query(InstrumentRow)
                       .filter(InstrumentRow.label == instrument.lower())]
                query = query.filter(Song.id.in_(ids))
            out = []
            for song in query.order_by(Song.uploaded_at.desc()).all():
                job = (s.query(Job).filter(Job.song_id == song.id)
                       .order_by(Job.id.desc()).first())
                out.append({"id": song.id, "title": song.title, "artist": song.artist,
                            "duration_s": song.duration_s,
                            "genres": [g.tag for g in song.genre_tags],
                            "job_id": job.id if job else None,
                            "job_status": job.status if job else None})
            return out

    @app.get("/api/v1/songs/{song_id}/analysis")
    def get_analysis(song_id: int):
        with get_session() as s:
            a = (s.query(Analysis).filter(Analysis.song_id == song_id)
                 .order_by(Analysis.id.desc()).first())
            if not a:
                raise HTTPException(404, "no analysis yet")
            return a.report_json

    @app.get("/api/v1/songs/{song_id}/stems/{name}")
    def get_stem(song_id: int, name: str):
        _assert_owner(song_id)  # NG7: stems only to uploader — enforced even single-user
        with get_session() as s:
            stem = (s.query(Stem).filter(Stem.song_id == song_id, Stem.name == name)
                    .order_by(Stem.id.desc()).first())
            if not stem:
                raise HTTPException(404, "stem not found")
        return FileResponse(storage_mod.storage.path(stem.object_key), media_type="audio/wav")

    @app.get("/api/v1/songs/{song_id}/audio")
    def get_original(song_id: int):
        _assert_owner(song_id)
        with get_session() as s:
            song = s.get(Song, song_id)
            if not song:
                raise HTTPException(404)
        key = f"songs/{song_id}/original.{song.fmt}"
        p = storage_mod.storage.path(key)
        if not p.exists():  # fmt/suffix mismatch fallback
            parent = storage_mod.storage.path(f"songs/{song_id}")
            candidates = list(parent.glob("original.*")) if parent.exists() else []
            if not candidates:
                raise HTTPException(404)
            p = candidates[0]
        return FileResponse(p)

    @app.post("/api/v1/songs/{song_id}/reanalyze")
    def reanalyze(song_id: int):
        with get_session() as s:
            song = s.get(Song, song_id)
            if not song:
                raise HTTPException(404)
            job = Job(song_id=song_id)
            s.add(job)
            s.commit()
            job_id = job.id
        try:
            queue.submit(job_id, LOCAL_USER_ID)
        except AdmissionError as e:
            resp = JSONResponse(status_code=429,
                                content={"code": e.code, "retry_after_s": e.retry_after_s})
            resp.headers["Retry-After"] = str(e.retry_after_s)
            return resp
        return {"job_id": job_id}

    # ---------------- jobs ----------------

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: int):
        with get_session() as s:
            job = s.get(Job, job_id)
            if not job:
                raise HTTPException(404)
            runs = (s.query(StageRun).filter(StageRun.job_id == job_id)
                    .order_by(StageRun.id).all())
            return {"id": job.id, "song_id": job.song_id, "status": job.status,
                    "stages": [{"name": r.stage, "status": r.status,
                                "wall_ms": r.wall_ms, "error": r.error or None}
                               for r in runs]}

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: int):
        with get_session() as s:
            job = s.get(Job, job_id)
            if not job:
                raise HTTPException(404)
            if job.status in ("done", "failed", "partial"):
                async def _done():
                    import json as _json
                    yield f"data: {_json.dumps({'type': 'job', 'status': job.status})}\n\n"
                return StreamingResponse(_done(), media_type="text/event-stream")
        return StreamingResponse(bus.subscribe(job_id), media_type="text/event-stream")

    return app


def _assert_owner(song_id: int, user_id: int = LOCAL_USER_ID):
    with get_session() as s:
        song = s.get(Song, song_id)
        if not song:
            raise HTTPException(404)
        if song.user_id != user_id:
            raise HTTPException(403, "stems are only available to the uploader")


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
