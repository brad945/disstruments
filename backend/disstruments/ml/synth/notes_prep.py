"""Prepare per-note collections for the `notes` engine (M4 L3c).

Output layout (what `engines.NotesEngine` reads): `<out>/<instrument>-<pitch:03d>-<vel:03d>.wav`,
mono, leading/trailing silence trimmed. Every kept note is pitch-checked with pYIN against the
pitch its filename (or scale position) claims; mismatches are dropped and counted.

    python -m disstruments.ml.synth.notes_prep iowa  --out ~/datasets/disstruments-synth/sources/notes
    python -m disstruments.ml.synth.notes_prep nsynth --src ~/datasets/nsynth --out ...
    # v8 (only after Bradley approves the downloads; docs/debriefs/v8-plan.md):
    python -m disstruments.ml.synth.notes_prep iowa --set v8 --out .../sources/notes
    python -m disstruments.ml.synth.notes_prep vcsl --src ~/datasets/vcsl --out .../sources/notes

Percussion one-shots (v8) use the same layout with the GM drum note as the "pitch"
(`vcsl_claps-039-080.wav`); the notes engine plays drum parts key-exact, without resampling.

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

# v8 (pending Bradley's approval; docs/debriefs/v8-plan.md): instrument id ->
# (page, include regex, exclude regex, pYIN fmax Hz, octave-tolerant pitch check).
# Mallets: the named pitch is kept but checked modulo the octave (bar partials make pYIN's
# octave unreliable; glockenspiel parts are written 2 octaves below sounding).
IOWA_V8_PITCH_PAGES: dict[str, tuple[str, str, str, float, bool]] = {
    "iowa_violin_arco": ("MIS-Pitches-2012/MISViolin2012.html", r"\.arco\.", "", 2700, False),
    "iowa_viola_arco": ("MIS-Pitches-2012/MISViola2012.html", r"\.arco\.", "", 2000, False),
    "iowa_cello_arco": ("MIS-Pitches-2012/MISCello2012.html", r"\.arco\.", "", 2000, False),
    "iowa_flute": ("MIS-Pitches-2012/MISFlute2012.html", "", "", 2700, False),
    "iowa_oboe": ("MIS-Pitches-2012/MISOboe2012.html", "", "", 2000, False),
    "iowa_clarinet": ("MIS-Pitches-2012/MISBbClarinet2012.html", "", "", 2000, False),
    "iowa_bassoon": ("MIS-Pitches-2012/MISBassoon2012.html", "", "", 2000, False),
    "iowa_horn": ("MIS-Pitches-2012/MISHorn2012.html", "", "", 2000, False),
    "iowa_trombone": ("MIS-Pitches-2012/MISTenorTrombone2012.html", "", "", 2000, False),
    "iowa_tuba": ("MIS-Pitches-2012/MISTuba2012.html", "", "", 2000, False),
    "iowa_marimba": ("MIS-Pitches-2012/MISMarimba2012.html", r"\.yarn\.", r"roll", 2700, True),
    "iowa_vibraphone": ("MIS-Pitches-2012/MISVibraphone2012.html", r"\.sustain\.", "", 2700, True),
    "iowa_xylophone": ("MIS-Pitches-2012/MISxylophone2012.html", r"rubber", r"roll|gliss", 4200, True),
    "iowa_bells": ("MIS-Pitches-2012/MISBells2012.html", r"\.brass\.", "", 4200, True),
}
# Unpitched one-shots: instrument -> (page, include regex, GM drum keys it plays). Each file is
# split at onsets (a file may hold several strokes); strokes become round robins.
IOWA_V8_ONESHOTS: dict[str, tuple[str, str, tuple[int, ...]]] = {
    f"iowa_tambourine{k}": ("MIS-Pitches-2012/MISTambourines2012.html",
                            rf"^tambourine{k}\.(normal|sforzando)\.", (54,)) for k in (1, 2, 3)
}


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


def median_pitch(y: np.ndarray, sr: int, fmax: float = 2000) -> float | None:
    import librosa
    seg = y[: int(1.0 * sr)]
    f0, voiced, _ = librosa.pyin(seg, fmin=30, fmax=fmax, sr=sr, frame_length=4096)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(librosa.hz_to_midi(np.median(f0))) if len(f0) >= 3 else None


def pitch_ok(est: float | None, named: int, *, octave_ok: bool = False, tol: float = 0.6) -> bool:
    """Does a pYIN estimate confirm the pitch a file name claims? `octave_ok` accepts any
    octave (pitch class within `tol` semitones), for mallets whose octave pYIN can't judge."""
    if est is None:
        return False
    d = est - named
    if octave_ok:
        d = (d + 6) % 12 - 6
    return abs(d) <= tol


VCSL_DYN = {"ppp": 30, "pp": 40, "p": 55, "mp": 68, "mf": 80, "f": 100, "ff": 112, "fff": 120,
            "soft": 50, "med": 80, "medium": 80, "loud": 110, "hard": 110, "normal": 85}


def vcsl_tokens(stem: str) -> tuple[int | None, int, int]:
    """(MIDI pitch or None, velocity, round-robin index) from a VCSL file name, e.g.
    `glock_loud_C6_01` -> (84, 110, 1), `Vibes_hard_A2_v2_rr1_Main` -> (45, 74, 1),
    `KSHarp_A2_mf1` -> (45, 80, 1), `Clap_rr3` -> (None, 80, 3)."""
    pitch, vel, rr = None, 80, 1
    for tok in re.split(r"[_\s]+", stem):
        t = tok.lower()
        if t in VCSL_DYN:
            vel = VCSL_DYN[t]
            continue
        m = re.fullmatch(r"([a-z]+)(\d)", t)                 # 'mf1' = dynamic + take number
        if m and m.group(1) in VCSL_DYN:
            vel, rr = VCSL_DYN[m.group(1)], int(m.group(2))
            continue
        if re.fullmatch(r"v(l)?\d", t):                       # velocity layer v1..v5 / vl1..
            vel = 30 + 22 * int(t[-1])
            continue
        if re.fullmatch(r"rr\d+", t):
            rr = int(t[2:])
            continue
        if re.fullmatch(r"\d{1,2}", t):                       # trailing take number `_01`
            rr = int(t)
            continue
        p = note_to_midi(tok)
        if p is not None:
            pitch = p
    return pitch, vel, rr


def free_velocity(taken: set[int], vel: int) -> int:
    """Nearest unused velocity at or above `vel` (round robins of one dynamic share a key, so
    each gets its own velocity slot; the notes engine's +-12 velocity jitter then alternates
    between them)."""
    v = max(1, min(127, vel))
    while v in taken and v < 127:
        v += 1
    while v in taken and v > 1:
        v -= 1
    taken.add(v)
    return v


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


def _page_hrefs(page: str) -> list[str]:
    html = _urlopen(BASE + page, 60).read().decode("latin-1")
    return sorted(set(re.findall(r'href="([^"]+\.aiff?)"', html, re.I)))


def _onsets(y: np.ndarray, sr: int) -> list[int]:
    import librosa
    on = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="samples",
                                    pre_max=20, post_max=20, delta=0.2, wait=int(0.15 * sr / 512))
    return list(on) or [0]


def iowa_v8(out: Path, cache: Path) -> dict:
    """v8 orchestral instruments + tambourines (DOWNLOADS ~1.2 GB of AIFF into `cache`;
    run only after Bradley approves, docs/debriefs/v8-plan.md)."""
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for inst, (page, inc, exc, fmax, octave_ok) in IOWA_V8_PITCH_PAGES.items():
        hrefs = [h for h in _page_hrefs(page) if re.search(inc, Path(h).name, re.I)
                 and not (exc and re.search(exc, Path(h).name, re.I))]
        kept = bad = 0
        for h in hrefs:
            url = urllib.parse.urljoin(BASE + page, urllib.parse.quote(h, safe="/:.-_"))
            f = _fetch(url, cache / inst / Path(h).name)
            pitches, vel = parse_tokens(f.stem)
            if len(pitches) != 1:
                continue
            y, sr = _load_mono(f)
            y = trim(y, sr)
            est = median_pitch(y, sr, fmax=fmax) if len(y) else None
            if not pitch_ok(est, pitches[0], octave_ok=octave_ok):
                bad += 1
                continue
            kept += _write(out, inst, pitches[0], vel or 80, y, sr)
        report[inst] = {"files": len(hrefs), "kept": kept, "pitch_mismatch": bad}
    for inst, (page, inc, keys) in IOWA_V8_ONESHOTS.items():
        hrefs = [h for h in _page_hrefs(page) if re.search(inc, Path(h).name, re.I)]
        kept = 0
        taken: dict[int, set[int]] = {k: set() for k in keys}
        for h in hrefs:
            url = urllib.parse.urljoin(BASE + page, urllib.parse.quote(h, safe="/:.-_"))
            f = _fetch(url, cache / "tambourines" / Path(h).name)
            y, sr = _load_mono(f)
            base = 115 if "sforzando" in f.name else 90
            bounds = _onsets(y, sr) + [len(y)]
            for a, b in list(zip(bounds[:-1], bounds[1:]))[:8]:      # <= 8 strokes per file
                seg = trim(y[a:b], sr)
                if len(seg) < 0.05 * sr:
                    continue
                for k in keys:
                    kept += _write(out, inst, k, free_velocity(taken[k], base), seg, sr)
        report[inst] = {"files": len(hrefs), "kept": kept}
    return report


# VCSL folder (relative to a sparse checkout) -> (instrument id, GM keys or None = pitched,
# file-name regex to skip). Pitched folders are pitch-checked like Iowa (mallets octave-tolerant).
VCSL_SETS: dict[str, tuple[str, tuple[int, ...] | None, str]] = {
    "Idiophones/Struck Idiophones/Shaker, Small": ("vcsl_shaker_small", (69, 70, 82), r"roll"),
    "Idiophones/Struck Idiophones/Shaker, Large": ("vcsl_shaker_large", (69, 70, 82), r"roll"),
    "Idiophones/Struck Idiophones/Cabasa": ("vcsl_cabasa", (69, 70, 82), r"rub"),
    "Idiophones/Struck Idiophones/Claps": ("vcsl_claps", (39,), r"^$"),
    "Idiophones/Struck Idiophones/Tambourine 1": ("vcsl_tambourine1", (54,), r"roll|shake"),
    "Idiophones/Struck Idiophones/Tambourine 2": ("vcsl_tambourine2", (54,), r"roll|shake"),
    "Membranophones/Struck Membranophones/Bongos": ("vcsl_bongos", (60, 61, 62, 63, 64), r"roll"),
    "Membranophones/Struck Membranophones/Conga": ("vcsl_conga", (60, 61, 62, 63, 64), r"roll"),
    "Membranophones/Struck Membranophones/Darbuka": ("vcsl_darbuka", (60, 61, 62, 63, 64), r"roll"),
    "Idiophones/Struck Idiophones/Glockenspiel": ("vcsl_glockenspiel", None, r"^$"),
    "Idiophones/Struck Idiophones/Marimba": ("vcsl_marimba", None, r"roll"),
    "Idiophones/Struck Idiophones/Vibraphone": ("vcsl_vibraphone", None, r"bow"),
    "Idiophones/Struck Idiophones/Xylophone": ("vcsl_xylophone", None, r"roll"),
    "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C": ("vcsl_harmonica_special20", None, r"_rel_|releases"),
    "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F": ("vcsl_harmonica_special20", None, r"_rel_|releases"),
    "Aerophones/Free Aerophones/Harmonica-Hohner-Super64": ("vcsl_harmonica_super64", None, r"_rel_|releases"),
    "Chordophones/Composite Chordophones/Concert Harp": ("vcsl_concert_harp", None, r"^$"),
    "Chordophones/Composite Chordophones/Folk Harp": ("vcsl_folk_harp", None, r"^$"),
}
MALLETS = ("vcsl_glockenspiel", "vcsl_marimba", "vcsl_vibraphone", "vcsl_xylophone")


def vcsl(src: Path, out: Path) -> dict:
    """VCSL folders (an approved sparse checkout at `src`) -> notes layout. One-shot kits
    write every hit under each of their GM keys (the leaf, not the drum's pitch, is the
    label); pitched folders keep the named pitch after a pYIN check."""
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for folder, (inst, keys, skip) in VCSL_SETS.items():
        d = src / folder
        files = sorted(p for p in d.rglob("*.wav") if not re.search(skip, str(p.relative_to(d)), re.I)) \
            if d.is_dir() else []
        r = report.setdefault(inst, {"files": 0, "kept": 0, "pitch_mismatch": 0})
        r["files"] += len(files)
        taken: dict[int, set[int]] = {}
        for f in files:
            pitch, vel, _ = vcsl_tokens(f.stem)
            y, sr = _load_mono(f)
            y = trim(y, sr)
            if len(y) < 0.03 * sr:
                continue
            if keys is not None:
                for k in keys:
                    r["kept"] += _write(out, inst, k, free_velocity(taken.setdefault(k, set()), vel), y, sr)
                continue
            if pitch is None:
                continue
            est = median_pitch(y, sr, fmax=4200)
            if not pitch_ok(est, pitch, octave_ok=inst in MALLETS):
                r["pitch_mismatch"] += 1
                continue
            r["kept"] += _write(out, inst, pitch, free_velocity(taken.setdefault(pitch, set()), vel), y, sr)
    return report


def nsynth(src: Path, out: Path, families: tuple[str, ...]) -> dict:
    """Move NSynth notes into the shared layout (names are already `<inst>-<pitch>-<vel>`),
    keeping only `families`; write the per-instrument metadata the labelling page needs.
    Metadata is merged into an existing nsynth_instruments.json (v8 adds families
    incrementally; overwriting would drop the instruments of earlier runs)."""
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "nsynth_instruments.json"
    old: dict[str, dict] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
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
    merged = old | meta
    meta_path.write_text(json.dumps(merged, indent=1))
    return {"instruments": len(meta), "instruments_total": len(merged),
            "notes": len(list(out.glob("*.wav")))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m disstruments.ml.synth.notes_prep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("iowa")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--cache", type=Path, default=Path.home() / "datasets/iowa")
    i.add_argument("--set", default="v7", choices=("v7", "v8"),
                   help="v7: the M4 instruments; v8: orchestral + mallets + tambourines (needs approval)")
    v = sub.add_parser("vcsl", help="v8: VCSL percussion/mallets/harmonica/harp (needs approval)")
    v.add_argument("--src", type=Path, default=Path.home() / "datasets/vcsl")
    v.add_argument("--out", type=Path, required=True)
    n = sub.add_parser("nsynth")
    n.add_argument("--src", type=Path, default=Path.home() / "datasets/nsynth")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--families", nargs="+",
                   default=["bass", "guitar", "keyboard", "reed", "synth_lead", "organ", "brass"])
    a = ap.parse_args(argv)
    if a.cmd == "iowa":
        fn = iowa_v8 if a.set == "v8" else iowa
        print(json.dumps(fn(a.out.expanduser(), a.cache.expanduser()), indent=1))
    elif a.cmd == "vcsl":
        print(json.dumps(vcsl(a.src.expanduser(), a.out.expanduser()), indent=1))
    else:
        print(json.dumps(nsynth(a.src.expanduser(), a.out.expanduser(), tuple(a.families)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
