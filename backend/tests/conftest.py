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


# ---------------------------------------------------------------------- ML (disstruments.ml)
_MDB_PUBLIC_DIGEST = Path(__file__).resolve().parent / "fixtures" / "ml" / "medleydb_public_labels.json"


def materialize_medleydb_public(dest: Path) -> Path:
    """Write the committed label-only digest of the 196 public MedleyDB tracks back out as
    minimal `<TrackId>_METADATA.yaml` files (the fields the loader reads)."""
    import json

    import yaml
    doc = json.loads(_MDB_PUBLIC_DIGEST.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for tid, t in doc["tracks"].items():
        stems = {}
        for sid, s in t["stems"].items():
            e = {"instrument": s["instrument"]}
            if s.get("raw"):
                e["raw"] = {f"R{i + 1:02d}": {"instrument": r} for i, r in enumerate(s["raw"])}
            stems[sid] = e
        meta = {"artist": t["artist"], "genre": t["genre"], "has_bleed": t["has_bleed"],
                "stems": stems}
        (dest / f"{tid}_METADATA.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return dest


@pytest.fixture(scope="session")
def mdb_public_root(tmp_path_factory) -> Path:
    return materialize_medleydb_public(tmp_path_factory.mktemp("mdb_public"))
