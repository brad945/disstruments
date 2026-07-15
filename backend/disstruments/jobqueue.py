"""JobQueue interface with an in-process worker (Phase 0). Celery adapter at Phase 1.

Admission control (PRD §7.2 job layer) lives HERE, not in routes, so any future
queue backend inherits it. Enforcement is gated by settings.rate_limiting_enabled;
the cost ledger is always-on (F24).
"""
from __future__ import annotations

import datetime as dt
import queue
import threading
import traceback

from .config import UNLIMITED, settings
from .db import CostLedger, Job, get_session
from .events import bus


class AdmissionError(Exception):
    def __init__(self, code: str, retry_after_s: int = 60):
        self.code = code  # concurrency | quota_daily | queue_full  (PRD §7.3)
        self.retry_after_s = retry_after_s
        super().__init__(code)


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def check_admission(user_id: int) -> None:
    """Raises AdmissionError when a Phase-1 limit would be exceeded."""
    with get_session() as s:
        active = s.query(Job).filter(Job.status == "running").count()
        queued = s.query(Job).filter(Job.status == "queued").count()

        lim = settings.limit("max_concurrent_jobs_per_user")
        if lim != UNLIMITED and active >= lim:
            raise AdmissionError("concurrency", 120)
        lim = settings.limit("max_queued_jobs_per_user")
        if lim != UNLIMITED and queued >= lim:
            raise AdmissionError("queue_full", 300)
        lim = settings.limit("global_queue_depth")
        if lim != UNLIMITED and (active + queued) >= lim:
            raise AdmissionError("queue_full", 300)
        lim = settings.limit("daily_job_quota")
        if lim != UNLIMITED:
            row = (s.query(CostLedger)
                   .filter(CostLedger.user_id == user_id, CostLedger.day == _today())
                   .one_or_none())
            if row and row.jobs_count >= lim:
                raise AdmissionError("quota_daily", 3600)


def record_job_cost(user_id: int, gpu_s: float = 0.0, storage_bytes: int = 0, jobs: int = 0):
    with get_session() as s:
        row = (s.query(CostLedger)
               .filter(CostLedger.user_id == user_id, CostLedger.day == _today())
               .one_or_none())
        if row is None:
            row = CostLedger(user_id=user_id, day=_today())
            s.add(row)
        row.gpu_s += gpu_s
        row.storage_bytes += storage_bytes
        row.jobs_count += jobs
        s.commit()


class InProcessQueue:
    """Single worker thread; stages idempotent so a crashed job can be re-run."""

    def __init__(self, runner):
        self._runner = runner  # callable(job_id)
        self._q: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="diss-worker")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._q.put(-1)

    def submit(self, job_id: int, user_id: int):
        check_admission(user_id)
        record_job_cost(user_id, jobs=1)
        self._q.put(job_id)
        bus.publish(job_id, {"type": "job", "status": "queued"})

    def _loop(self):
        while not self._stop.is_set():
            job_id = self._q.get()
            if job_id == -1:
                return
            try:
                self._runner(job_id)
            except Exception:  # runner handles its own errors; this is the last resort
                traceback.print_exc()
                with get_session() as s:
                    job = s.get(Job, job_id)
                    if job:
                        job.status = "failed"
                        job.finished_at = dt.datetime.now(dt.timezone.utc)
                        s.commit()
                bus.publish(job_id, {"type": "job", "status": "failed"})
