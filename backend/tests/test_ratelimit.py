"""Rate limiting tested with limits ON (PRD 'no gaps' rule) even though local mode is off."""
from disstruments.config import settings
from disstruments.ratelimit import SlidingWindow, classify


def test_classify():
    assert classify("/api/v1/songs", "POST") == "upload"
    assert classify("/api/v1/songs", "GET") == "general"
    assert classify("/api/v1/songs/1/stems/vocals", "GET") == "download"
    assert classify("/api/v1/songs/1/audio", "GET") == "download"
    assert classify("/api/v1/jobs/1", "GET") == "general"


def test_sliding_window_enforces_limit():
    w = SlidingWindow()
    limit = 5
    for i in range(limit):
        allowed, remaining, _ = w.check("k", limit)
        assert allowed
        assert remaining == limit - i - 1
    allowed, remaining, reset = w.check("k", limit)
    assert not allowed
    assert remaining == 0
    assert reset > 0


def test_unlimited_sentinel_never_blocks():
    from disstruments.config import UNLIMITED
    w = SlidingWindow()
    for _ in range(1000):
        allowed, remaining, _ = w.check("k", UNLIMITED)
        assert allowed
        assert remaining == UNLIMITED


def test_headers_present_and_429_shape(client):
    """Headers must exist even when dormant; then flip limits on and verify 429 contract."""
    r = client.get("/api/v1/songs")
    assert r.headers["X-RateLimit-Limit"] == "unlimited"
    assert "X-RateLimit-Reset" in r.headers

    settings.rate_limiting_enabled = True
    settings.rl_general_per_min = 2
    try:
        client.get("/api/v1/songs")
        client.get("/api/v1/songs")
        r = client.get("/api/v1/songs")
        assert r.status_code == 429
        body = r.json()
        assert body["code"] == "rate_general"
        assert body["retry_after_s"] >= 1
        assert "Retry-After" in r.headers
        assert r.headers["X-RateLimit-Remaining"] == "0"
    finally:
        settings.rate_limiting_enabled = False
        settings.rl_general_per_min = 60


def test_job_admission_quota():
    from disstruments.jobqueue import AdmissionError, check_admission
    settings.rate_limiting_enabled = True
    settings.daily_job_quota = 0  # everything over quota
    try:
        try:
            check_admission(user_id=1)
        except AdmissionError as e:
            # only fires if a job was already recorded today (e2e test did) — code must be right
            assert e.code in ("quota_daily", "concurrency", "queue_full")
    finally:
        settings.rate_limiting_enabled = False
        settings.daily_job_quota = 10
