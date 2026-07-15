"""Central config. Every limit lives here (PRD §7); local profile = unlimited.

Env vars use the DISS_ prefix, e.g. DISS_FAKE_ML=1, DISS_RATE_LIMITING_ENABLED=1.
"""
from pathlib import Path

from pydantic_settings import BaseSettings

UNLIMITED = -1  # sentinel per PRD §7.3


class Settings(BaseSettings):
    model_config = {"env_prefix": "DISS_"}

    data_dir: Path = Path.home() / ".disstruments"
    database_url: str = ""  # default derived from data_dir below

    # Upload caps (F1) — enforced even locally (they protect the pipeline, not the wallet)
    max_upload_mb: int = 200
    max_duration_s: int = 900

    # Legal posture (design doc §9): canonical wav deleted after job unless kept
    keep_canonical: bool = False

    # ML
    fake_ml: bool = False           # deterministic fake stages for dev/tests
    stem_model: str = "htdemucs"    # demucs v4; "htdemucs_6s" for 6-stem once evaled
    device: str = "auto"            # auto -> mps > cuda > cpu
    tag_threshold_show: float = 0.30      # F15: hide below
    tag_threshold_confident: float = 0.60 # F15: 0.3–0.6 marked uncertain in UI

    # Rate limiting (F23) — dormant locally, values are the Phase-1 profile (PRD §7.2)
    rate_limiting_enabled: bool = False
    rl_general_per_min: int = 60
    rl_upload_per_min: int = 10
    rl_download_per_min: int = 30
    max_concurrent_jobs_per_user: int = 1
    max_queued_jobs_per_user: int = 3
    daily_job_quota: int = 10
    global_queue_depth: int = 100

    def limit(self, name: str) -> int:
        """Effective limit: UNLIMITED sentinel when rate limiting is off."""
        return getattr(self, name) if self.rate_limiting_enabled else UNLIMITED

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.data_dir / 'disstruments.db'}"


settings = Settings()
