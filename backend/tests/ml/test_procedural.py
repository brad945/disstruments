"""PRD A2: procedural MIDI is licence-clean and fits the renderer's expectations."""
import json

import pytest

pytest.importorskip("pretty_midi")

from disstruments.ml.synth import midi as M  # noqa: E402
from disstruments.ml.synth import procedural as P  # noqa: E402


def test_songs_are_deterministic_and_long_enough():
    a, b = P.song(7), P.song(7)
    assert [len(i.notes) for i in a.instruments] == [len(i.notes) for i in b.instruments]
    assert a.get_end_time() >= 30.0


def test_every_window_part_is_usable_and_drums_stay_in_the_kit(tmp_path):
    root = P.generate(tmp_path / "proc", n_songs=40)
    files = sorted(root.rglob("*.mid"))
    assert len(files) == 40
    roles, cymbal_windows, n_windows = set(), 0, 0
    for f in files:
        ws = M.load_windows(f, root)
        assert len(ws) >= 2, f"{f.name}: need >= 2 windows"
        for w in ws:
            n_windows += 1
            roles |= {p.role for p in w.parts}
            drums = [p for p in w.parts if p.is_drum]
            cymbal_windows += any(n.pitch in M.CYMBAL_NOTES for p in drums for n in p.notes)
    assert {"drums", "bass", "keys_piano", "organ", "guitar", "synth_lead", "synth_pad"} <= roles
    assert cymbal_windows / n_windows > 0.6
    for f in files:                                   # no drum note is dropped by the kit filter
        import pretty_midi
        for inst in pretty_midi.PrettyMIDI(str(f)).instruments:
            if inst.is_drum:
                assert all(n.pitch in M.KIT_NOTES for n in inst.notes)


def test_splits_by_group_and_licence_marker(tmp_path):
    root = P.generate(tmp_path / "proc", n_songs=25, songs_per_group=10)
    keys = {M.composition_key(f, root) for f in root.rglob("*.mid")}
    assert {M.artist_key(k) for k in keys} == {"g0000", "g0001", "g0002"}
    assert P.midi_licence(root)[0] is True
    assert json.loads((root / "MIDI_LICENSE.json").read_text())["commercial_clean"] is True
    assert P.midi_licence(tmp_path)[0] is False, "no marker = Lakh = never clean"
