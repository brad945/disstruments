"""Request-layer rate limiting (F23): token buckets per user per endpoint class.

Dormant locally (RATE_LIMITING_ENABLED=false) but fully implemented and unit-tested
with limits ON, per PRD 'no gaps'. In-memory buckets here; the Phase-1 adapter is
the same sliding-window logic as a Redis Lua script shared with the Next.js BFF.

Headers (PRD §7.3): X-RateLimit-Limit / -Remaining / -Reset always present;
429 carries Retry-After header + {"code": ..., "retry_after_s": ...} body.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import UNLIMITED, settings

_WINDOW_S = 60


def classify(path: str, method: str) -> str:
    if method == "POST" and path.endswith("/songs"):
        return "upload"
    if "/stems/" in path or "/audio" in path or "/midi/" in path:
        return "download"
    return "general"


_LIMIT_FIELD = {"general": "rl_general_per_min",
                "upload": "rl_upload_per_min",
                "download": "rl_download_per_min"}


class SlidingWindow:
    def __init__(self):
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int) -> tuple[bool, int, int]:
        """-> (allowed, remaining, reset_epoch_s)"""
        now = time.monotonic()
        wall = int(time.time())
        dq = self._hits[key]
        while dq and dq[0] <= now - _WINDOW_S:
            dq.popleft()
        if limit == UNLIMITED:
            return True, UNLIMITED, wall + _WINDOW_S
        if len(dq) >= limit:
            reset = wall + int(_WINDOW_S - (now - dq[0])) + 1
            return False, 0, reset
        dq.append(now)
        return True, limit - len(dq), wall + _WINDOW_S


window = SlidingWindow()


async def rate_limit_middleware(request: Request, call_next):
    cls = classify(request.url.path, request.method)
    user = "1"  # F26 single local user; real identity at Phase 1
    limit = settings.limit(_LIMIT_FIELD[cls])
    allowed, remaining, reset = window.check(f"rl:{user}:{cls}", limit)

    if not allowed:
        retry_after = max(1, reset - int(time.time()))
        resp = JSONResponse(status_code=429,
                            content={"code": f"rate_{cls}", "retry_after_s": retry_after})
        resp.headers["Retry-After"] = str(retry_after)
    else:
        resp = await call_next(request)

    resp.headers["X-RateLimit-Limit"] = "unlimited" if limit == UNLIMITED else str(limit)
    resp.headers["X-RateLimit-Remaining"] = "unlimited" if remaining == UNLIMITED else str(remaining)
    resp.headers["X-RateLimit-Reset"] = str(reset)
    return resp
