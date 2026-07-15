"""SQLAlchemy models — schema mirrors design doc §5 so the Postgres swap is config-only."""
from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String,
                        Text, UniqueConstraint, create_engine, event)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)

from .config import settings


class Base(DeclarativeBase):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class Song(Base):
    __tablename__ = "songs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    content_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(String)
    artist: Mapped[str] = mapped_column(String, default="")
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    fmt: Mapped[str] = mapped_column(String, default="")
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    genre_tags: Mapped[list[GenreTag]] = relationship(cascade="all, delete-orphan")


class GenreTag(Base):
    __tablename__ = "genre_tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    tag: Mapped[str] = mapped_column(String, index=True)
    __table_args__ = (UniqueConstraint("song_id", "tag"),)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|done|failed|partial
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    stage_runs: Mapped[list[StageRun]] = relationship(cascade="all, delete-orphan")


class StageRun(Base):
    __tablename__ = "stage_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    stage: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String, default="")
    model_version: Mapped[str] = mapped_column(String, default="")
    weights_sha: Mapped[str] = mapped_column(String, default="")
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String, default="pending")  # ok|failed|skipped
    wall_ms: Mapped[int] = mapped_column(Integer, default=0)
    gpu_s: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    report_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


class Instrument(Base):
    """Derived index for library queries (F18/F20) — rebuildable from report_json."""
    __tablename__ = "instruments"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    label: Mapped[str] = mapped_column(String, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    stem: Mapped[str] = mapped_column(String, default="")


class Stem(Base):
    __tablename__ = "stems"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    object_key: Mapped[str] = mapped_column(String)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)  # null = never (local)


class CostLedger(Base):
    """Always-on cost signals (F24) — the rate-limiting substrate for Phase 1."""
    __tablename__ = "cost_ledger"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    day: Mapped[str] = mapped_column(String, index=True)  # YYYY-MM-DD
    gpu_s: Mapped[float] = mapped_column(Float, default=0.0)
    storage_bytes: Mapped[int] = mapped_column(Integer, default=0)
    jobs_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("user_id", "day"),)


_engine = None
SessionLocal: sessionmaker | None = None

LOCAL_USER_ID = 1  # F26: single seeded local user; real auth is a Phase-1 swap


def init_db(url: str | None = None):
    global _engine, SessionLocal
    _engine = create_engine(url or settings.db_url,
                            connect_args={"check_same_thread": False},
                            json_serializer=lambda o: json.dumps(o))

    @event.listens_for(_engine, "connect")
    def _fk_on(dbapi_conn, _):  # SQLite FK enforcement
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(_engine)
    SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    with SessionLocal() as s:
        if not s.get(User, LOCAL_USER_ID):
            s.add(User(id=LOCAL_USER_ID, email="local@disstruments"))
            s.commit()
    return _engine


def get_session() -> Session:
    assert SessionLocal is not None, "init_db() not called"
    return SessionLocal()
