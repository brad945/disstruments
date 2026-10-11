"""Hand-labelling NSynth instruments at leaf level (M4 L3c).

NSynth (CC BY 4.0) only says "acoustic guitar" or "electronic bass". Bradley listens to one
short phrase per instrument and picks the fine leaf (nylon/steel, picked/fingered/slap, ...);
`unsure` keeps the coarse node (its subtree is masked in training), `skip` drops it.

    python -m disstruments.ml.synth.nsynth_label page     # ~/datasets/nsynth/label.html + audio
    python -m disstruments.ml.synth.nsynth_label collect  # Downloads -> synth/nsynth_labels.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

NOTES = Path.home() / "datasets/disstruments-synth/sources/notes"
OUT = Path.home() / "datasets/nsynth"
LABELS = Path(__file__).with_name("nsynth_labels.json")

# family prefix -> (choices [(label, node)], coarse node for "unsure", max instruments)
FAMILIES: dict[str, tuple[list[tuple[str, str]], str, int]] = {
    "guitar_acoustic": ([("nylon (classical)", "guitar.acoustic.nylon"),
                         ("steel-string", "guitar.acoustic.steel")], "guitar.acoustic", 30),
    "guitar_electronic": ([("clean electric", "guitar.electric.clean"),
                           ("distorted / overdriven", "guitar.electric.distorted"),
                           ("lap / pedal steel", "guitar.lap_steel")], "guitar.electric", 20),
    "bass_electronic": ([("fingered", "bass.electric.fingered"), ("picked", "bass.electric.picked"),
                         ("slap", "bass.electric.slap"), ("fretless", "bass.electric.fretless"),
                         ("synth bass", "bass.synth.analog")], "bass.electric", 30),
    "bass_acoustic": ([("upright / double bass", "bass.upright")], "bass", 10),
    "bass_synthetic": ([("analog-style synth bass", "bass.synth.analog"),
                        ("808 / sub", "bass.synth.sub_808")], "bass.synth", 12),
    "keyboard_electronic": ([("Rhodes", "keys.electric_piano.rhodes"),
                             ("Wurlitzer", "keys.electric_piano.wurlitzer"),
                             ("FM / DX7 e-piano", "keys.electric_piano.fm"),
                             ("acoustic piano", "keys.piano"), ("clavinet", "keys.clavinet"),
                             ("organ", "keys.organ")], "keys.electric_piano", 25),
    "reed_acoustic": ([("soprano sax", "woodwinds.saxophone.soprano"),
                       ("alto sax", "woodwinds.saxophone.alto"),
                       ("tenor sax", "woodwinds.saxophone.tenor"),
                       ("baritone sax", "woodwinds.saxophone.baritone"),
                       ("clarinet", "woodwinds.clarinet"), ("harmonica", "woodwinds.harmonica")],
                      "woodwinds.saxophone", 15),
}


def _phrase(inst: str, pitches: list[int], sr: int = 16000) -> np.ndarray:
    """Arpeggio (root, 3rd, 5th, octave; 0.45 s each) then one held note (1.6 s)."""
    import soundfile as sf
    mid = pitches[len(pitches) // 2]
    root = min(pitches, key=lambda p: abs(p - (mid - 5)))

    def note(p: int, dur: float) -> np.ndarray:
        q = min(pitches, key=lambda x: abs(x - p))
        vels = sorted(NOTES.glob(f"{inst}-{q:03d}-*.wav"))
        f = next((v for v in vels if v.stem.endswith("-100")), vels[-1] if vels else None)
        if f is None:
            return np.zeros(int(dur * sr), np.float32)
        y, _ = sf.read(str(f), dtype="float32")
        y = y[: int(dur * sr)]
        y = np.pad(y, (0, int(dur * sr) - len(y)))
        y[-int(0.05 * sr):] *= np.linspace(1, 0, int(0.05 * sr))
        return y

    parts = [note(root + d, 0.45) for d in (0, 4, 7, 12)] + [np.zeros(int(0.15 * sr)), note(root, 1.6)]
    y = np.concatenate(parts)
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


def page(seed: int = 0) -> Path:
    import soundfile as sf
    meta = json.loads((NOTES / "nsynth_instruments.json").read_text())
    have = json.loads(LABELS.read_text())["labels"] if LABELS.exists() else {}
    rnd = random.Random(seed)
    items = []
    aud = OUT / "audition"
    aud.mkdir(parents=True, exist_ok=True)
    for fam, (choices, coarse, cap) in FAMILIES.items():
        insts = sorted(i for i in meta if i.rsplit("_", 1)[0] == fam and i not in have)
        rnd.shuffle(insts)
        for inst in sorted(insts[:cap]):
            wav = aud / f"{inst}.wav"
            if not wav.exists():
                sf.write(wav, _phrase(inst, meta[inst]["pitches"]), 16000)
            items.append({"id": inst, "family": fam, "audio": f"audition/{inst}.wav",
                          "choices": choices + [("unsure", coarse), ("skip (not this family / broken)", "skip")]})
    html = (Path(__file__).with_name("nsynth_label_template.html").read_text()
            .replace("/*__ITEMS__*/[]", json.dumps(items)))
    out = OUT / "label.html"
    out.write_text(html)
    print(f"{len(items)} instruments -> {out}")
    return out


def collect(downloads: Path = Path.home() / "Downloads") -> int:
    files = sorted(downloads.glob("nsynth_labels*.json"))
    if not files:
        print("no nsynth_labels*.json in Downloads")
        return 1
    merged = json.loads(LABELS.read_text()) if LABELS.exists() else {"schema": 1, "source": "nsynth", "labels": {}}
    for f in files:
        merged["labels"].update(json.loads(f.read_text())["labels"])
        shutil.move(str(f), str(OUT / f.name))
    LABELS.write_text(json.dumps(merged, indent=1, sort_keys=True))
    n = sum(v != "skip" for v in merged["labels"].values())
    print(f"{len(merged['labels'])} labelled ({n} usable) -> {LABELS}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m disstruments.ml.synth.nsynth_label")
    ap.add_argument("cmd", choices=("page", "collect"))
    a = ap.parse_args(argv)
    if a.cmd == "page":
        page()
        return 0
    return collect()


if __name__ == "__main__":
    raise SystemExit(main())
