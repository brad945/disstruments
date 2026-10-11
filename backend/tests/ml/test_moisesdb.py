"""MoisesDB (test-only): window fetcher over a locally built zip (STORED + DEFLATED members,
RIFF files with JUNK/LIST chunks, 24-bit/16-bit, mono/stereo, mixed sample rates) and the
contextual label semantics. No network."""
import io
import json
import struct
import zipfile

import numpy as np
import pytest
import soundfile as sf

from disstruments.ml.datasets import DATASETS, UnknownLabelsError, load_dataset
from disstruments.ml.datasets import moisesdb_clips as mc
from disstruments.ml.datasets import moisesdb_fetch as mf
from disstruments.ml.eval import cli
from disstruments.ml.taxonomy import Taxonomy

TAX = Taxonomy.load()
A, U, S = mc.ACTIVE, mc.UNCERTAIN, mc.SILENT
SR = 24000
DUR = 3.0
NOSLEEP = {"sleep": lambda s: None}


# ---------------------------------------------------------------- fixtures
def wav_bytes(x, sr, bits=24, junk=True, odd_chunk=True):
    """x: float (frames, ch). RIFF with JUNK + odd-sized LIST chunk before fmt/data."""
    ch = x.shape[1]
    if bits == 24:
        v = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32).reshape(-1)
        b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8)
        pcm = b.tobytes()
    else:
        pcm = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes()
    ba = ch * bits // 8
    chunks = b""
    if junk:
        chunks += b"JUNK" + struct.pack("<I", 28) + b"\0" * 28
    if odd_chunk:
        chunks += b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\0"       # pad byte
    chunks += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ba, ba, bits)
    chunks += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def noise(seed, n, ch, level):
    return np.random.default_rng(seed).uniform(-1, 1, (n, ch)) * level


SONG = "11111111-aaaa-bbbb-cccc-000000000001"
SONG2 = "22222222-aaaa-bbbb-cccc-000000000002"
# (stem, track_id, trackType, has_bleed, kind) ; kind -> signal spec
TRACKS = [
    ("bass", "t-bass", "bass_guitar", False, dict(level=0.3, ch=2, sr=SR, bits=24, comp=zipfile.ZIP_DEFLATED)),
    ("guitar", "t-gtr", "distorted_electric_guitar", False, dict(level=0.2, ch=2, sr=SR, bits=24, comp=zipfile.ZIP_STORED)),
    ("vocals", "t-vox", "lead_male_singer", True, dict(level=0.1, ch=1, sr=48000, bits=16, comp=zipfile.ZIP_DEFLATED)),
    ("other_keys", "t-pad", "synth_pad", False, dict(level=0.0, ch=2, sr=SR, bits=24, comp=zipfile.ZIP_STORED)),
]


def build_root(tmp_path, songs=((SONG, "Band A", "rock", TRACKS),)):
    root = tmp_path / "moisesdb"
    (root / "meta").mkdir(parents=True)
    zpath = root / "moisesdb.zip"
    signals = {}
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("moisesdb/", b"")
        for sid, artist, genre, tracks in songs:
            stems = {}
            for k, (stem, tid, ttype, bleed, spec) in enumerate(tracks):
                x = noise(sum(map(ord, tid)), int(DUR * spec["sr"]), spec["ch"], spec["level"])
                if stem == "guitar":                       # silence first, then noise
                    x[: spec["sr"] // 2] = 0
                signals[(sid, tid)] = (x, spec["sr"], spec["bits"])
                z.writestr(zipfile.ZipInfo(mf.zip_path(sid, stem, tid)),
                           wav_bytes(x, spec["sr"], spec["bits"]), compress_type=spec["comp"])
                stems.setdefault(stem, []).append({"trackType": ttype, "id": tid,
                                                   "has_bleed": bleed, "extension": "wav"})
            meta = {"song": f"song {sid[:4]}", "artist": artist, "genre": genre,
                    "stems": [{"id": s, "stemName": s, "tracks": t} for s, t in stems.items()]}
            (root / "meta" / f"{sid}.json").write_text(json.dumps(meta))
    with zipfile.ZipFile(zpath) as z:
        index = [[i.filename, i.file_size, i.compress_size, i.header_offset, i.compress_type]
                 for i in z.infolist()]
    (root / "index.json").write_text(json.dumps(index))
    return root, zpath, signals


def entry(root, name):
    return mf.load_index(root)[name]


# ---------------------------------------------------------------- WAV parsing
def test_parse_wav_header_walks_chunks():
    x = noise(0, 100, 2, 0.5)
    b = wav_bytes(x, 44100, 24)
    info = mf.parse_wav_header(b)
    assert (info.channels, info.sample_rate, info.bits, info.block_align) == (2, 44100, 24, 6)
    assert info.data_offset == 12 + 36 + 14 + 24 + 8 and info.frames == 100
    assert mf.parse_wav_header(b[:40]) is None                     # header not complete yet
    y = mf.decode_pcm(b[info.data_offset:], info)
    assert y.shape == (2, 100) and np.abs(y - x.T).max() < 2 ** -22
    with pytest.raises(mf.FatalFetchError):
        mf.parse_wav_header(b"RIFX" + b[4:])


def test_decode_16bit_mono():
    x = noise(1, 50, 1, 0.5)
    b = wav_bytes(x, 8000, 16, junk=False, odd_chunk=False)
    info = mf.parse_wav_header(b)
    assert info.data_offset == 44
    assert np.abs(mf.decode_pcm(b[44:], info)[0] - x[:, 0]).max() < 1 / 32768


# ---------------------------------------------------------------- member windows
def test_stored_member_reads_only_the_window(tmp_path):
    root, zpath, sig = build_root(tmp_path)
    e = entry(root, mf.zip_path(SONG, "guitar", "t-gtr"))
    assert e[4] == zipfile.ZIP_STORED
    src = mf.FileRangeSource(zpath)
    w = mf.read_member_window(src, e, 1.0, 1.0, **NOSLEEP)
    x, sr, _ = sig[(SONG, "t-gtr")]
    assert w.audio.shape == (2, SR) and np.abs(w.audio - x[SR:2 * SR].T).max() < 2 ** -22
    window_bytes = SR * 6
    assert w.bytes_read == src.bytes_read
    assert window_bytes <= w.bytes_read <= window_bytes + mf.HEAD_PROBE + 1200
    assert w.bytes_read < e[1] / 2                                   # far less than the member


def test_deflated_member_stops_early(tmp_path):
    root, zpath, sig = build_root(tmp_path)
    e = entry(root, mf.zip_path(SONG, "bass", "t-bass"))
    assert e[4] == zipfile.ZIP_DEFLATED
    src = mf.FileRangeSource(zpath, chunk=4096)
    w = mf.read_member_window(src, e, 0.5, 1.0, **NOSLEEP)
    x, _, _ = sig[(SONG, "t-bass")]
    assert np.abs(w.audio - x[SR // 2: SR // 2 + SR].T).max() < 2 ** -22
    assert w.bytes_read < 0.7 * e[2]                                  # stopped well before the end


def test_window_past_end_is_zero_padded(tmp_path):
    root, zpath, sig = build_root(tmp_path)
    e = entry(root, mf.zip_path(SONG, "bass", "t-bass"))
    w = mf.read_member_window(mf.FileRangeSource(zpath), e, DUR - 0.5, 1.0, **NOSLEEP)
    assert w.frames_available == SR // 2 and w.audio.shape == (2, SR)
    assert not w.audio[:, SR // 2:].any()


class FlakySource(mf.FileRangeSource):
    """Drops the connection once after `fail_after` bytes of a stream."""

    def __init__(self, path, fail_after, exc=ConnectionResetError, msg="reset", times=1):
        super().__init__(path, chunk=1024)
        self.fail_after, self.exc, self.msg, self.times, self.failed = fail_after, exc, msg, times, 0

    def stream(self, start, end):
        n = 0
        for b in super().stream(start, end):
            n += len(b)
            if self.failed < self.times and n > self.fail_after:
                self.failed += 1
                raise self.exc(self.msg)
            yield b


@pytest.mark.parametrize("track", ["t-bass", "t-gtr"])
def test_resume_after_dropped_connection(tmp_path, track):
    root, zpath, _ = build_root(tmp_path)
    stem = "bass" if track == "t-bass" else "guitar"
    e = entry(root, mf.zip_path(SONG, stem, track))
    ref = mf.read_member_window(mf.FileRangeSource(zpath), e, 1.0, 1.0, **NOSLEEP)
    logs = []
    src = FlakySource(zpath, fail_after=20000)
    w = mf.read_member_window(src, e, 1.0, 1.0, log=logs.append, **NOSLEEP)
    assert src.failed == 1 and any("retry 1/" in m for m in logs)
    assert np.array_equal(w.audio, ref.audio)


def test_fatal_errors_are_not_retried(tmp_path):
    root, zpath, _ = build_root(tmp_path)
    e = list(entry(root, mf.zip_path(SONG, "bass", "t-bass")))
    e[3] += 1                                                         # wrong header offset
    with pytest.raises(mf.FatalFetchError, match="signature"):
        mf.read_member_window(mf.FileRangeSource(zpath), e, 0, 1, **NOSLEEP)


# ---------------------------------------------------------------- whole run
def test_run_end_to_end_and_resume(tmp_path):
    root, zpath, sig = build_root(tmp_path)
    res = mf.run(root, mf.FileRangeSource(zpath), start_s=1.0, dur_s=1.0, workers=2,
                 min_free_gb=0, echo=False, **NOSLEEP)
    assert res["ok"] == 1 and not res["failed"]
    doc = json.loads((root / "clips" / f"{SONG}_w1.json").read_text())
    audio, sr = sf.read(root / "clips" / f"{SONG}_w1.flac", dtype="float64")
    assert sr == 24000 and audio.ndim == 1 and len(audio) == 24000
    # independent mix: stereo-ize, resample the 48 kHz track, sum, mean channels
    acc = np.zeros((2, SR))
    for (_, tid), (x, tsr, _) in sig.items():
        seg = x[int(1.0 * tsr): int(2.0 * tsr)].T
        seg = np.repeat(seg, 2, 0) if seg.shape[0] == 1 else seg
        acc += mf.resample(seg.astype(np.float32), tsr, SR)[:, :SR]
    expect = acc.mean(0) * 10 ** (doc["mix"]["gain_db"] / 20)
    assert np.abs(audio - expect).max() < 1e-4
    t = {x["track_id"]: x for x in doc["tracks"]}
    assert t["t-pad"]["rms_db"] is None and t["t-pad"]["activity"] == "silent"
    assert t["t-vox"]["sample_rate"] == 48000 and t["t-vox"]["channels"] == 1 and t["t-vox"]["bits"] == 16
    assert t["t-bass"]["rms_db"] == pytest.approx(20 * np.log10(0.3 / np.sqrt(3)), abs=0.2)
    assert doc["bytes_downloaded"] == sum(x["bytes_read"] for x in doc["tracks"]) > 0
    assert doc["artist"] == "Band A" and doc["genre"] == "rock"
    # resumable: nothing re-downloaded
    src2 = mf.FileRangeSource(zpath)
    assert mf.run(root, src2, start_s=1.0, dur_s=1.0, min_free_gb=0, echo=False)["ok"] == 0
    assert src2.bytes_read == 0
    assert "1/1 songs done" in (root / "fetch.log").read_text()


def test_url_never_logged(tmp_path):
    root, zpath, _ = build_root(tmp_path)
    url = "https://acct123.r2.cloudflarestorage.com/moisesdb.zip?X-Amz-Signature=deadbeefcafe1234&X-Amz-Credential=AKIAxyz%2F2026"
    red = mf.Redactor(url)
    src = FlakySource(zpath, fail_after=10, exc=OSError, times=10 ** 6,
                      msg=f"cannot reach {url} (acct123.r2.cloudflarestorage.com)")
    mf.run(root, src, start_s=1.0, dur_s=1.0, workers=1, min_free_gb=0, echo=False, redact=red,
           sleep=lambda s: None)
    log = (root / "fetch.log").read_text()
    assert "FAILED" in log and "retry" in log
    for secret in ("deadbeefcafe1234", "cloudflarestorage", "X-Amz", "acct123", "https://"):
        assert secret not in log


def test_round_robin_spreads_artists():
    metas = {f"s{i}": {"artist": "Big" if i < 4 else f"a{i}"} for i in range(6)}
    order = mf.artist_round_robin(metas)
    assert order[:3] == ["s0", "s4", "s5"] and sorted(order) == sorted(metas)


# ---------------------------------------------------------------- activity + labels
def test_classify_activity():
    st = mc.classify_activity([(-12.0, -6.0), (-40.0, -30.0), (-49.0, -40.0), (-55.0, -45.0),
                               (-90.0, -75.0), (None, None)])
    assert st == [A, A, U, U, S, S]          # -49 is > 35 dB below -12 -> uncertain, not active
    assert mc.classify_activity([(-60.0, -50.0)]) == [U]             # loudest but under the floor


def _tracks(*types, bleed=()):
    return [{"trackType": t, "has_bleed": t in bleed, "track_id": f"x{i}"} for i, t in enumerate(types)]


def lab(types, states, genre="rock", bleed=()):
    pos, obs, unmapped, prov = mc.song_labels(_tracks(*types, bleed=bleed), states, genre, TAX)
    return pos, obs, obs - pos, unmapped


def test_every_track_type_is_mapped():
    lm = mc.label_map(TAX)
    assert len(lm.labels()) == 38      # all distinct trackTypes in the 240 data.json files


def test_no_keys_song_keeps_keys_observed_negative():
    pos, obs, neg, _ = lab([mc.BASS_GUITAR, mc.CLEAN_GUITAR], [A, A])
    assert {"keys", "keys.synth", "keys.synth.lead", "keys.piano", "voice", "drums"} <= neg
    assert not {"keys.synth.pluck", "keys.synth.brass"} & obs              # constant leaf mask
    assert "bass.electric" in pos and not set(TAX.children("bass.electric")) & obs
    assert "cymbals" not in obs          # Q1: cymbals scored on OpenMIC only (always masked)


def test_silent_track_is_negative_uncertain_is_unknown():
    _, obs, neg, _ = lab([mc.BASS_GUITAR, mc.SYNTH_LEAD, mc.GRAND_PIANO], [A, S, U])
    assert "keys.synth.lead" in neg
    assert "keys.piano" not in obs and "keys" not in obs            # may be present
    assert "keys.organ" in neg


def test_bleed_unknown_unless_clean_track_of_same_label():
    pos, obs, _, _ = lab([mc.LEAD_MALE], [A], bleed={mc.LEAD_MALE})
    assert "voice.sung" not in obs and "voice" not in obs
    pos, obs, _, _ = lab([mc.LEAD_MALE, mc.LEAD_FEMALE], [A, A], bleed={mc.LEAD_MALE})
    assert "voice.sung" in pos
    assert "voice.rap" not in obs and "voice.screamed" not in obs   # delivery unknown
    assert "voice.choir" in obs                                     # no backing vocals


def test_rap_genre_vocals():
    pos, obs, _, _ = lab([mc.LEAD_MALE], [A], genre="rap")
    assert "voice" in pos and not set(TAX.children("voice")) & obs


def test_drum_machine_context():
    pos, obs, _, _ = lab([mc.KICK, mc.SNARE, mc.DRUM_MACHINE, mc.HI_HAT], [A, A, A, A])
    assert {"drums", "drums.electronic"} <= pos and "cymbals" not in pos
    assert "drums.acoustic_kit" not in obs
    pos, obs, _, _ = lab([mc.KICK, mc.DRUM_MACHINE, mc.OVERHEADS], [A, A, A])
    assert {"drums.acoustic_kit", "drums.electronic"} <= pos
    pos, obs, neg, _ = lab([mc.KICK, mc.SNARE, mc.CYMBALS], [A, A, S])
    assert "drums.acoustic_kit" in pos and "drums.electronic" in neg
    assert "cymbals" not in neg                   # Q1: cymbals never scored on MoisesDB
    pos, obs, _, _ = lab([mc.KICK, mc.SNARE], [A, A], genre="electronic")
    assert "drums" in pos and not {"drums.acoustic_kit", "drums.electronic"} & obs


def test_cymbal_context():
    _, obs, _, _ = lab([mc.KICK, mc.SNARE, mc.OVERHEADS], [A, A, A])
    assert "cymbals" not in obs                   # only inside overheads/bleed
    _, obs, _, _ = lab([mc.FULL_KIT, mc.HI_HAT], [A, S])
    assert "cymbals" not in obs                   # a full-kit track contains the hats
    pos, _, _, _ = lab([mc.KICK, mc.HI_HAT], [A, A])
    assert "cymbals" not in pos and "cymbals" not in obs   # Q1: never scored here
    _, obs, _, _ = lab([mc.FX], [A])
    assert not {"cymbals", "keys.synth", "drums.electronic"} & obs and "keys.piano" in obs


def test_epiano_masks_piano_and_other_keys():
    pos, obs, _, _ = lab([mc.EPIANO], [A])
    assert "keys.electric_piano" in pos and "keys.piano" not in obs
    _, obs, _, _ = lab([mc.OTHER_KEYS], [A])
    assert not set(TAX.descendants("keys", include_self=True)) & obs
    assert "guitar" in obs


def test_load_and_unknown_track_type(tmp_path):
    root, zpath, _ = build_root(tmp_path)
    mf.run(root, mf.FileRangeSource(zpath), start_s=1.0, dur_s=1.0, min_free_gb=0, echo=False)
    idx = load_dataset("moisesdb", root, window_start=1.0)
    (r,) = idx.records
    assert r.split == "test" and r.artist == "band a" and r.item_id == f"{SONG}_w1"
    assert r.audio["mix"].exists()
    assert {"bass.electric", "guitar.electric.distorted"} <= r.positive
    assert "voice.sung" not in r.observed                          # only vocal track has bleed
    assert "keys.synth.pad" in r.negative                          # silent pad track
    assert "moisesdb" in DATASETS
    with pytest.raises(FileNotFoundError):
        load_dataset("moisesdb", root)                             # no w20 clips
    p = root / "clips" / f"{SONG}_w1.json"
    doc = json.loads(p.read_text())
    doc["tracks"][0]["trackType"] = "kazoo"
    p.write_text(json.dumps(doc))
    with pytest.raises(UnknownLabelsError):
        load_dataset("moisesdb", root, window_start=1.0)
    r = load_dataset("moisesdb", root, window_start=1.0, strict=False).records[0]
    assert r.observed == r.positive and r.unknown_labels == ("kazoo",)


def test_eval_cli(tmp_path):
    songs = ((SONG, "Band A", "rock", TRACKS), (SONG2, "Band B", "pop", TRACKS[:2]))
    root, zpath, _ = build_root(tmp_path, songs)
    mf.run(root, mf.FileRangeSource(zpath), start_s=1.0, dur_s=1.0, min_free_gb=0, echo=False)
    preds = tmp_path / "rand.npz"
    args = ["--dataset", "moisesdb", "--root", str(root), "--split", "test", "--window-start", "1"]
    assert cli.main(["baseline", "--kind", "random", *args, "--out", str(preds)]) == 0
    assert cli.main(["run", *args, "--predictions", str(preds),
                     "--runs-dir", str(tmp_path / "runs")]) == 0
    rep = json.loads(next((tmp_path / "runs").iterdir()).joinpath("metrics.json").read_text())
    assert rep["config"]["dataset"] == "moisesdb" and rep["config"]["loader"]["window_start"] == 1.0



def test_bleed_only_type_is_unknown_even_when_silent():
    """Label-independent masking: a type whose every track has bleed is UNKNOWN whether
    it sounds or not (otherwise `voice` would be observed only when absent)."""
    _, obs_silent, _, _ = lab([mc.LEAD_MALE, mc.BASS_GUITAR], [S, A], bleed={mc.LEAD_MALE})
    _, obs_active, _, _ = lab([mc.LEAD_MALE, mc.BASS_GUITAR], [A, A], bleed={mc.LEAD_MALE})
    assert "voice.sung" not in obs_silent and "voice.sung" not in obs_active


def test_pooled_windows(tmp_path):
    root, zpath, _ = build_root(tmp_path)
    for start in (0.0, 1.0):
        mf.run(root, mf.FileRangeSource(zpath), start_s=start, dur_s=1.0, min_free_gb=0, echo=False)
    idx = load_dataset("moisesdb", root, window_start="all")
    assert sorted(r.item_id for r in idx.records) == [f"{SONG}_w0", f"{SONG}_w1"]
    assert len({r.artist for r in idx.records}) == 1               # same group for bootstrap
    assert idx.config["window_start"] == "all"
