import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Test profile: fake ML, isolated data dir. Set BEFORE importing the app.
_tmp = tempfile.mkdtemp(prefix="diss-test-")
os.environ["DISS_FAKE_ML"] = "1"
os.environ["DISS_DATA_DIR"] = _tmp
os.environ["DISS_DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from disstruments.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def wav_file():
    """3-second 440Hz + 110Hz stereo test tone."""
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    sig = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)
    data = np.stack([sig, sig], axis=1).astype("float32")
    p = Path(_tmp) / "tone.wav"
    sf.write(p, data, sr)
    return p


def wait_job(client, job_id: int, timeout=30):
    import time
    for _ in range(timeout * 10):
        r = client.get(f"/api/v1/jobs/{job_id}")
        status = r.json()["status"]
        if status in ("done", "failed", "partial"):
            return status
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish")
