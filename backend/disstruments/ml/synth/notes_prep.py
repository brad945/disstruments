"""Prepare per-note collections for the `notes` engine (M4 L3c).

Output layout (what `engines.NotesEngine` reads): `<out>/<instrument>-<pitch:03d>-<vel:03d>.wav`,
mono, leading/trailing silence trimmed. Every kept note is pitch-checked with pYIN against the
pitch its filename (or scale position) claims; mismatches are dropped and counted.

    python -m disstruments.ml.synth.notes_prep iowa  --out ~/datasets/disstruments-synth/sources/notes
    python -m disstruments.ml.synth.notes_prep nsynth --src ~/datasets/nsynth --out ...

Iowa MIS (University of Iowa Electronic Music Studios, "may be downloaded and used for any
projects, without restrictions"): double bass pizzicato, alto/soprano sax and trumpet come as
one file per pitch; the guitar (Raimundo 118, a Spanish classical = nylon) only as chromatic
runs, which are split at onsets.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


def _urlopen(url: str, timeout: int = 120):
    """urllib with certifi's CA bundle (the uv Python build ships without system certs)."""
    import ssl

    import certifi
    return urllib.request.urlopen(url, timeout=timeout,
                                  context=ssl.create_default_context(cafile=certifi.where()))

BASE = "https://theremin.music.uiowa.edu/"
NOTE_RE = re.compile(r"^([A-G])(b|#)?(-?\d)$")
DYN = {"pp": 40, "p": 55, "mf": 80, "f": 100, "ff": 112}
SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SR = 44100

# instrument id -> (per-pitch page, filename filter)
IOWA_PITCH_PAGES = {
    "iowa_doublebass_pizz": ("MIS-Pitches-2012/MISDoubleBass2012.html", "pizz"),
    "iowa_altosax": ("MIS-Pitches-2012/MISEbAltoSaxophone2012.html", ""),
    "iowa_sopranosax": ("MIS-Pitches-2012/MISBbSopranoSaxophone2012.html", ""),
    "iowa_trumpet": ("MIS-Pitches-2012/MISBbTrumpet2012.html", ""),
}
IOWA_GUITAR_ZIP = "sound files/MIS/Piano_Other/guitar/Guitar.mono.1644.1.zip"


def note_to_midi(name: str) -> int | None:
    m = NOTE_RE.match(name)
    if not m:
        return None
    letter, acc, octave = m.groups()
    return 12 * (int(octave) + 1) + SEMI[letter] + {"b": -1, "#": 1, None: 0}[acc]


def parse_tokens(stem: str) -> tuple[list[int], int | None]:
    """MIDI pitches named in a filename (one per pitch file, or the two ends of a range like
    `E2B2`) and the velocity implied by its dynamic marking."""
    pitches, vel = [], None
    for tok in stem.split("."):
        if tok.lower() in DYN:
            vel = DYN[tok.lower()]
        p = note_to_midi(tok)
        if p is not None:
            pitches.append(p)
            continue
        m = re.fullmatch(r"([A-G][b#]?-?\d)([A-G][b#]?-?\d)", tok)      # range, e.g. C3Bb3
        if m:
            pitches += [note_to_midi(m.group(1)), note_to_midi(m.group(2))]
    return pitches, vel


def trim(y: np.ndarray, sr: int, floor_db: float = -45.0, pre_s: float = 0.01) -> np.ndarray:
    """Cut leading/trailing audio quieter than `floor_db` relative to the peak."""
    env = np.abs(y)
    thr = env.max() * 10 ** (floor_db / 20) if env.max() > 0 else 0
    loud = np.nonzero(env > thr)[0]
    if not len(loud):
        return y[:0]
    a = max(0, loud[0] - int(pre_s * sr))
    return y[a:loud[-1] + 1]


def median_pitch(y: np.ndarray, sr: int) -> float | None:
    import librosa
    seg = y[: int(1.0 * sr)]
    f0, voiced, _ = librosa.pyin(seg, fmin=30, fmax=2000, sr=sr, frame_length=4096)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(librosa.hz_to_midi(np.median(f0))) if len(f0) >= 3 else None


def _write(out: Path, inst: str, pitch: int, vel: int, y: np.ndarray, sr: int) -> bool:
    import soundfile as sf
    path = out / f"{inst}-{pitch:03d}-{vel:03d}.wav"
    if path.exists():                               # first take wins (e.g. two strings, one pitch)
        return False
    sf.write(path, (y / max(1e-9, np.abs(y).max()) * 0.9).astype(np.float32), sr)
    return True


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with _urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    return dest


def iowa(out: Path, cache: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for inst, (page, filt) in IOWA_PITCH_PAGES.items():
        html = _urlopen(BASE + page, 60).read().decode("latin-1")
        hrefs = sorted({h for h in re.findall(r'href="([^"]+\.aiff?)"', html, re.I)
                        if filt.lower() in h.lower()})
        kept = bad = 0
        for h in hrefs:
            url = urllib.parse.urljoin(BASE + page, urllib.parse.quote(h, safe="/:.-_"))
            f = _fetch(url, cache / inst / Path(h).name)
            pitches, vel = parse_tokens(f.stem)
            if len(pitches) != 1:
                continue
            y, sr = _load_mono(f)
            y = trim(y, sr)
            est = median_pitch(y, sr) if len(y) else None
            if est is None or abs(est - pitches[0]) > 0.6:
                bad += 1
                continue
            kept += _write(out, inst, pitches[0], vel or 80, y, sr)
        report[inst] = {"files": len(hrefs), "kept": kept, "pitch_mismatch": bad}
    report["iowa_guitar_nylon"] = _iowa_guitar(out, cache)
    return report


def _iowa_guitar(out: Path, cache: Path) -> dict:
    import librosa
    z = _fetch(BASE + urllib.parse.quote(IOWA_GUITAR_ZIP, safe="/:.-_"), cache / "guitar" / "Guitar.mono.1644.1.zip")
    kept = bad = 0
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".aif", ".aiff")) or name.startswith("__MACOSX"):
                continue
            dst = cache / "guitar" / Path(name).name
            if not dst.exists():
                dst.write_bytes(zf.read(name))
            pitches, vel = parse_tokens(dst.stem)
            if len(pitches) != 2:
                continue
            lo, hi = sorted(pitches)
            y, sr = _load_mono(dst)
            onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="samples",
                                                pre_max=20, post_max=20, delta=0.2, wait=int(0.3 * sr / 512))
            bounds = list(onsets) + [len(y)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                seg = trim(y[a:b], sr)
                if len(seg) < 0.2 * sr:
                    continue
                est = median_pitch(seg, sr)
                p = int(round(est)) if est is not None else None
                if p is None or abs(est - p) > 0.4 or not lo <= p <= hi:
                    bad += 1
                    continue
                kept += _write(out, "iowa_guitar_nylon", p, vel or 80, seg, sr)
    return {"kept": kept, "rejected_segments": bad}


def nsynth(src: Path, out: Path, families: tuple[str, ...]) -> dict:
    """Move NSynth notes into the shared layout (names are already `<inst>-<pitch>-<vel>`),
    keeping only `families`; write the per-instrument metadata the labelling page needs."""
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for split in ("valid", "test", "train"):
        d = src / f"nsynth-{split}"
        ex = d / "examples.json"
        if not ex.exists():
            continue
        for key, e in json.loads(ex.read_text()).items():
            inst = e["instrument_str"]
            if not inst.startswith(families):
                continue
            meta.setdefault(inst, {"family": e["instrument_family_str"], "source": e["instrument_source_str"],
                                   "split": split, "pitches": []})["pitches"].append(e["pitch"])
            f = d / "audio" / f"{key}.wav"
            if f.exists():
                f.rename(out / f.name)
    for m in meta.values():
        m["pitches"] = sorted(set(m["pitches"]))
    (out / "nsynth_instruments.json").write_text(json.dumps(meta, indent=1))
    return {"instruments": len(meta), "notes": len(list(out.glob("*.wav")))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m disstruments.ml.synth.notes_prep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("iowa")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--cache", type=Path, default=Path.home() / "datasets/iowa")
    n = sub.add_parser("nsynth")
    n.add_argument("--src", type=Path, default=Path.home() / "datasets/nsynth")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--families", nargs="+",
                   default=["bass", "guitar", "keyboard", "reed", "synth_lead", "organ", "brass"])
    a = ap.parse_args(argv)
    if a.cmd == "iowa":
        print(json.dumps(iowa(a.out.expanduser(), a.cache.expanduser()), indent=1))
    else:
        print(json.dumps(nsynth(a.src.expanduser(), a.out.expanduser(), tuple(a.families)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
