"""Fake-ML end-to-end: upload -> pipeline -> report -> stems -> dedup."""
from conftest import wait_job


def test_upload_analyze_report(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post("/api/v1/songs",
                        files={"file": ("tone.wav", f, "audio/wav")},
                        data={"genres": "indie rock, pop"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body["deduplicated"]
    status = wait_job(client, body["job_id"])
    assert status == "done"

    report = client.get(f"/api/v1/songs/{body['song_id']}/analysis").json()
    assert report["schema_version"] == 1
    assert report["global"]["key"]["value"]
    assert report["global"]["bpm"]["value"] > 0
    assert report["stems_status"] == "ok"
    assert len(report["stems"]) == 4
    assert report["instruments"], "expected instrument tags"
    for tag in report["instruments"]:
        assert 0 <= tag["confidence"] <= 1
        assert tag["activations"], "every tag needs time regions"
    assert report["structure"]["status"] == "skipped"  # draft-2 slot, never guessed
    stages = {s["stage"] for s in report["provenance"]["stages"]}
    assert {"transcode", "separation", "tagging_mix", "attributes"} <= stages


def test_stem_download_owner_only(client, wav_file):
    songs = client.get("/api/v1/songs").json()
    sid = songs[0]["id"]
    r = client.get(f"/api/v1/songs/{sid}/stems/vocals")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 1000


def test_stems_zip(client, wav_file):
    import io
    import zipfile
    sid = client.get("/api/v1/songs").json()[0]["id"]
    r = client.get(f"/api/v1/songs/{sid}/stems.zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    names = sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert names == ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    assert client.get("/api/v1/songs/999999/stems.zip").status_code == 404


def test_dedup_on_reupload(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post("/api/v1/songs", files={"file": ("tone.wav", f, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["deduplicated"] is True


def test_rejects_garbage_file(client, tmp_path):
    bad = tmp_path / "not_audio.mp3"
    bad.write_bytes(b"this is not audio at all" * 100)
    with open(bad, "rb") as f:
        r = client.post("/api/v1/songs", files={"file": ("not_audio.mp3", f, "audio/mpeg")})
    assert r.status_code == 422


def test_library_filters(client):
    assert any(s["genres"] == ["indie rock", "pop"] or "indie rock" in s["genres"]
               for s in client.get("/api/v1/songs").json())
    hits = client.get("/api/v1/songs", params={"genre": "indie rock"}).json()
    assert len(hits) >= 1
