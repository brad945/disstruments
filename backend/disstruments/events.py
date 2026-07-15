"""In-memory job event bus for SSE progress (F22). Redis pub/sub at Phase 1."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict


class JobEventBus:
    def __init__(self):
        self._subs: dict[int, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def publish(self, job_id: int, event: dict):
        """Thread-safe: called from the worker thread."""
        if self._loop is None:
            return
        for q in list(self._subs.get(job_id, [])):
            self._loop.call_soon_threadsafe(q.put_nowait, event)

    async def subscribe(self, job_id: int):
        q: asyncio.Queue = asyncio.Queue()
        self._subs[job_id].append(q)
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "job" and event.get("status") in ("done", "failed", "partial"):
                    return
        finally:
            self._subs[job_id].remove(q)


bus = JobEventBus()
