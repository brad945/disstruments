"""Regenerate the synthetic ML fixtures (metadata + tiny arrays only; never audio).

    backend/.venv/bin/python backend/tests/fixtures/ml/make_fixtures.py

Real-format files that are copied verbatim, not generated:
- medleydb/Metadata/*.yaml       marl/medleydb medleydb/data/Metadata (MIT repo)
- slakh/.../train/Track00001/metadata.yaml   mirdata tests/resources/mir_datasets/slakh
- openmic class-map.json          cosmir/openmic-2018 class-map.json
- medleydb_public_labels.json     label-only digest of the 196 public MedleyDB v1+v2 track
  YAMLs (marl/medleydb, MIT; ids from resources/tracklist_v1.txt + tracklist_v2.txt):
  per track `artist`, `genre`, `has_bleed`, and per stem `instrument` + raw-track
  `instrument`s. Rebuild from a marl/medleydb checkout with
  `make_fixtures.py --medleydb-digest <repo>/medleydb` (`make_medleydb_digest`); tests
  re-materialize it as METADATA.yaml files (tests/conftest.py `mdb_public_root`).
The Slakh tracks below reuse real Slakh2100 track ids + MIDI md5s (slakh-utils
duplicates.json) so the vendored split tables apply: Track00004/Track01939 share a MIDI
(01939 omitted in redux), Track00003 moves train->test in split2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def _stem(inst_class, program, name, plugin, rendered=True, drum=False):
    s = {"audio_rendered": rendered, "inst_class": inst_class, "is_drum": drum,
         "midi_program_name": name, "midi_saved": rendered, "plugin_name": plugin,
         "program_num": program}
    if rendered:
        s["integrated_loudness"] = -18.0
    return s


SLAKH = {
    ("train", "Track00004", "c613ddce3ded35d2e8cadf3511a1ffb9", "TRAAAAA128F0000001"): {
        "S00": _stem("Guitar", 24, "Acoustic Guitar (nylon)", "nylon_guitar2.nkm"),
        "S01": _stem("Bass", 36, "Slap Bass 1", "scarbee_jay_bass_slap_both.nkm"),
        "S02": _stem("Drums", 128, "Drums", "pop_kit.nkm", drum=True),
        "S03": _stem("Strings", 40, "Violin", "solo_strings.nkm"),
        "S04": _stem("Sound Effects", 103, "FX 8 (sci-fi)", "None", rendered=False),
    },
    ("omitted", "Track01939", "c613ddce3ded35d2e8cadf3511a1ffb9", "TRAAAAA128F0000001"): {
        "S00": _stem("Guitar", 24, "Acoustic Guitar (nylon)", "nylon_guitar.nkm"),
        "S01": _stem("Bass", 36, "Slap Bass 1", "scarbee_jay_bass_slap_both.nkm"),
        "S02": _stem("Drums", 128, "Drums", "funk_kit.nkm", drum=True),
    },
    ("train", "Track00003", "136e224ff9848e53c33322f58d58f1b3", "TRBBBBB128F0000002"): {
        "S00": _stem("Strings", 42, "Cello", "cello_solo.nkm"),
        "S01": _stem("Strings", 46, "Orchestral Harp", "harp.nkm"),
        "S02": _stem("Pipe", 73, "Flute", "flute.nkm"),
        "S03": _stem("Chromatic Percussion", 12, "Marimba", "marimba.nkm"),
    },
    ("validation", "Track01501", "52dc589baea0c387807aa964094e0cba", "TRCCCCC128F0000003"): {
        "S00": _stem("Piano", 4, "Electric Piano 1", "wurly_ep.nkm"),
        "S01": _stem("Organ", 16, "Drawbar Organ", "tonewheel_organ_b3.nkm"),
        "S02": _stem("Strings", 40, "Violin", "violin_solo.nkm"),
        "S03": _stem("Drums", 128, "Drums", "stadium_kit_full.nkm", drum=True),
    },
    # same MSD song id as Track00003 (different MIDI): a measured song-level leak
    ("test", "Track01876", "0f12ae105c1dc821c287b74632748b4e", "TRBBBBB128F0000002"): {
        "S00": _stem("Guitar", 25, "Acoustic Guitar (steel)", "/Library/Audio/Plug-Ins/Components/AGML2.component"),
        "S01": _stem("Piano", 4, "Electric Piano 1", "scarbee_mark_I.nkm"),
        "S02": _stem("Brass", 56, "Trumpet", "trumpet_section.nkm"),
        "S03": _stem("Strings (continued)", 52, "Choir Aahs", "choir_a.nkm"),
        "S04": _stem("Synth Lead", 81, "Lead 2 (sawtooth)", "december_saw.nkm"),
    },
}


def make_slakh() -> None:
    root = HERE / "slakh" / "slakh2100_flac_redux"
    for (split_dir, tid, md5, msd), stems in SLAKH.items():
        d = root / split_dir / tid
        d.mkdir(parents=True, exist_ok=True)
        meta = {"UUID": md5, "audio_dir": f"{tid}/stems",
                "lmd_midi_dir": f"lmd_matched/{msd[2]}/{msd[3]}/{msd[4]}/{msd}/{md5}.mid",
                "midi_dir": f"{tid}/MIDI", "normalization_factor": -13.0, "normalized": True,
                "overall_gain": 0.2, "stems": stems, "target_peak": -1.0}
        (d / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=True))


# OpenMIC: 12 clips, 4 artists; official-style split is artist-disjoint (a1,a2 train; a3,a4 test)
OPENMIC_CLIPS = [
    # sample_key, artist_id, split, {class: (relevance, observed)}
    ("000046_3840", "4", "train", {"guitar": 1.0, "voice": 0.8, "piano": 0.0}),
    ("000046_7680", "4", "train", {"guitar": 0.9, "drums": 0.7, "trumpet": 0.1}),
    ("000135_483840", "52", "train", {"piano": 0.83, "violin": 0.0}),
    ("000135_9600", "52", "train", {"synthesizer": 1.0, "guitar": 0.2}),
    ("000139_119040", "52", "train", {"drums": 1.0, "cymbals": 0.6, "voice": 0.5}),
    ("000141_153600", "4", "train", {"organ": 0.67, "bass": 0.9}),
    ("000144_30720", "4", "train", {"accordion": 0.0, "banjo": 0.0}),
    ("000178_3840", "60", "test", {"guitar": 0.9, "voice": 0.0, "drums": 0.66}),
    ("000308_61440", "60", "test", {"piano": 0.0, "violin": 1.0, "cello": 0.6}),
    ("000312_184320", "71", "test", {"saxophone": 0.75, "trumpet": 0.8, "piano": 0.4}),
    ("000319_145920", "71", "test", {"voice": 1.0, "guitar": 0.0, "mallet_percussion": 0.5}),
    ("071826_3840", "71", "test", {"ukulele": 0.9, "mandolin": 0.3}),   # README errata id
]


def make_openmic() -> None:
    root = HERE / "openmic" / "openmic-2018"
    (root / "partitions").mkdir(parents=True, exist_ok=True)
    class_map = json.loads((HERE / "vocab" / "openmic_class-map.json").read_text())
    (root / "class-map.json").write_text(json.dumps(class_map, indent=4))
    n = len(OPENMIC_CLIPS)
    y = np.zeros((n, len(class_map)), dtype=np.float64)
    m = np.zeros((n, len(class_map)), dtype=bool)
    for i, (_, _, _, labels) in enumerate(OPENMIC_CLIPS):
        for cls, rel in labels.items():
            y[i, class_map[cls]] = rel
            m[i, class_map[cls]] = True
    keys = np.array([c[0] for c in OPENMIC_CLIPS])
    np.savez_compressed(root / "openmic-2018.npz", X=np.zeros((n, 10, 128), dtype=np.uint8),
                        Y_true=y, Y_mask=m, sample_key=keys)
    for split in ("train", "test"):
        (root / "partitions" / f"split01_{split}.csv").write_text(
            "".join(f"{c[0]}\n" for c in OPENMIC_CLIPS if c[2] == split))
    header = "track_id,artist_id,artist_name,sample_key,start_time\n"
    rows = "".join(f"{int(c[0].split('_')[0])},{c[1]},Artist {c[1]},{c[0]},{int(c[0].split('_')[1]) / 1000}\n"
                   for c in OPENMIC_CLIPS)
    (root / "openmic-2018-metadata.csv").write_text(header + rows)


def make_medleydb_digest(repo_pkg: Path) -> None:
    """Label-only digest of the public tracks from a marl/medleydb checkout (`<repo>/medleydb`)."""
    ids = []
    for name in ("tracklist_v1.txt", "tracklist_v2.txt"):
        ids += [l.strip() for l in (repo_pkg / "resources" / name).read_text().splitlines() if l.strip()]
    tracks = {}
    for t in sorted(ids):
        m = yaml.safe_load((repo_pkg / "data" / "Metadata" / f"{t}_METADATA.yaml").read_text())
        stems = {}
        for sid, st in sorted((m.get("stems") or {}).items()):
            raws = [r.get("instrument") for _, r in sorted((st.get("raw") or {}).items())]
            stems[sid] = {"instrument": st.get("instrument"), **({"raw": raws} if raws else {})}
        tracks[t] = {"artist": m.get("artist"), "genre": m.get("genre"),
                     "has_bleed": m.get("has_bleed"), "stems": stems}
    doc = {"_about": "Label-only digest of the 196 public MedleyDB v1+v2 track metadata YAMLs "
                     "(marl/medleydb medleydb/data/Metadata, MIT). Fields: artist, genre, has_bleed, "
                     "stems.Sxx.instrument, raw-track instruments. No audio. Rebuild: "
                     "tests/fixtures/ml/make_fixtures.py --medleydb-digest <repo>/medleydb.",
           "source_tracklists": ["medleydb/resources/tracklist_v1.txt",
                                 "medleydb/resources/tracklist_v2.txt"],
           "tracks": tracks}
    (HERE / "medleydb_public_labels.json").write_text(json.dumps(doc, indent=0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--medleydb-digest":
        make_medleydb_digest(Path(sys.argv[2]))
    else:
        make_slakh()
        make_openmic()
    print("fixtures written under", HERE)
