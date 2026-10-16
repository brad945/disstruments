"""v8 synthetic families (docs/debriefs/v8-plan.md): GM role/leaf mapping for strings, brass,
woodwinds, mallets, accordion/harmonica and drum-channel percussion; the solo-polyphony rule;
cymbals staying kit-only; the distinct-instrument gate; LSO holdouts; one-shot notes; and the
v8 note-prep parsers. No audio sources needed (fake engine / tiny generated notes)."""
import json
from pathlib import Path

import numpy as np
import pytest

pretty_midi = pytest.importorskip("pretty_midi")
pytest.importorskip("pedalboard")
pytest.importorskip("pyloudnorm")

from disstruments.ml.synth import midi as M  # noqa: E402
from disstruments.ml.synth.sources import (SourceError, leaf_gate, load_registry,  # noqa: E402
                                           resolve_leaf_set, v8_holdout)
from disstruments.ml.taxonomy import Taxonomy  # noqa: E402

V1 = ["guitar.electric.clean", "bass.electric.picked", "keys.piano", "drums.acoustic_kit"]
NEW = ["strings.bowed.violin", "strings.bowed.section", "brass.trombone",
       "percussion.tambourine", "percussion.claps", "percussion.shaker"]


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.load()


def _track(prog, notes, drum=False):
    inst = pretty_midi.Instrument(program=prog, is_drum=drum)
    for p, s, e in notes:
        inst.notes.append(pretty_midi.Note(90, p, s, e))
    return inst


def _beats(pitches, secs=24.0, step=0.5, dur=0.3):
    out, t, k = [], 0.0, 0
    while t < secs:
        out.append((pitches[k % len(pitches)], t, t + dur))
        t += step
        k += 1
    return out


def _write(path: Path, tracks):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    pm.instruments.extend(tracks)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


# ------------------------------------------------------------------- role / leaf mapping
def test_every_role_leaf_is_a_taxonomy_leaf_and_hints_stay_in_role(tax):
    leaves = set(tax.leaves())
    for role, ls in M.ROLE_LEAVES.items():
        assert set(ls) <= leaves, role
    for prog, hint in M.PROGRAM_HINTS.items():
        role = M.role_of(prog, False)
        assert role is not None and set(hint) <= set(M.ROLE_LEAVES[role]), prog


@pytest.mark.parametrize("prog,leaf", [
    (40, "strings.bowed.violin"), (41, "strings.bowed.viola"), (42, "strings.bowed.cello"),
    (44, "strings.bowed.section"), (48, "strings.bowed.section"), (49, "strings.bowed.section"),
    (46, "strings.plucked.harp"), (105, "strings.plucked.banjo"),
    (56, "brass.trumpet"), (57, "brass.trombone"), (58, "brass.tuba"), (59, "brass.trumpet"),
    (60, "brass.french_horn"), (61, "brass.section"),
    (64, "woodwinds.saxophone.soprano"), (65, "woodwinds.saxophone.alto"),
    (66, "woodwinds.saxophone.tenor"), (67, "woodwinds.saxophone.baritone"),
    (68, "woodwinds.oboe"), (69, "woodwinds.oboe"), (70, "woodwinds.bassoon"),
    (71, "woodwinds.clarinet"), (72, "woodwinds.flute"), (73, "woodwinds.flute"),
    (9, "percussion.mallet.glockenspiel"), (11, "percussion.mallet.vibraphone"),
    (12, "percussion.mallet.marimba"), (13, "percussion.mallet.xylophone"),
    (21, "keys.accordion"), (23, "keys.accordion"), (22, "woodwinds.harmonica"),
])
def test_gm_program_pins_exact_leaf(prog, leaf):
    assert M.PROGRAM_HINTS[prog] == (leaf,)


@pytest.mark.parametrize("prog", [8, 10, 14, 15, 43, 45, 47, 50, 51, 52, 53, 54, 55,
                                  74, 75, 76, 77, 78, 79, 104, 106, 120])
def test_programs_without_a_decidable_leaf_are_dropped(prog):
    assert M.role_of(prog, False) is None


def test_synth_brass_keeps_v1_behaviour():
    # 62/63 (synth brass) stay in the brass role, unhinted, exactly as before v8
    assert M.role_of(62, False) == "brass" and 62 not in M.PROGRAM_HINTS


def test_v1_active_roles_are_the_pre_v8_roles():
    v1 = ["guitar.acoustic.nylon", "bass.upright", "keys.piano", "keys.organ.drawbar",
          "brass.trumpet", "woodwinds.saxophone.tenor", "keys.synth.lead", "keys.synth.pad",
          "drums.acoustic_kit"]
    assert M.active_roles(v1) == {"guitar", "bass", "keys_piano", "organ", "brass", "reed",
                                  "synth_lead", "synth_pad", "drums"}


# ------------------------------------------------------------------- drum-channel split
def test_drum_track_splits_into_kit_and_percussion_parts(tmp_path):
    kit = [42, 36, 38]                         # hat, kick, snare
    other = [54, 70, 39, 62, 56, 65]           # tamb, maracas, clap, conga, cowbell, timbale
    f = _write(tmp_path / "A" / "perc.mid", [_track(0, _beats(kit + other, step=0.1), drum=True)])
    wins = M.load_windows(f, tmp_path)
    by_role = {p.role: {n.pitch for n in p.notes} for p in wins[0].parts}
    assert by_role == {"drums": {42, 36, 38}, "perc_tambourine": {54}, "perc_shaker": {70},
                       "perc_claps": {39}, "perc_hand_drums": {62}}
    assert all(p.is_drum for p in wins[0].parts)
    names = {p.role: p.gm_name for p in wins[0].parts}
    assert names["drums"] == "Drums" and names["perc_claps"] == "Drums (claps)"
    # cowbell / timbales: no leaf -> in no part (never rendered unlabelled)
    assert not any({56, 65} & s for s in by_role.values())


def test_v1_roles_keep_only_the_kit(tmp_path):
    f = _write(tmp_path / "A" / "perc.mid", [_track(0, _beats([42, 36, 54, 39], step=0.1), drum=True)])
    wins = M.load_windows(f, tmp_path, roles=M.active_roles(V1))
    assert [p.role for p in wins[0].parts] == ["drums"]
    assert {n.pitch for n in wins[0].parts[0].notes} == {42, 36}


def test_percussion_note_sets_are_disjoint_from_kit_and_cymbals():
    allp = [n for s in M.PERC_ROLE_NOTES.values() for n in s]
    assert len(allp) == len(set(allp))
    assert not set(allp) & (M.KIT_NOTES | M.CYMBAL_NOTES)
    assert M.PERC_ROLE_NOTES["perc_shaker"] == {69, 70, 82}


# ------------------------------------------------------------------- solo polyphony rule
def test_max_polyphony():
    N = M.Note
    assert M.max_polyphony([]) == 0
    assert M.max_polyphony([N(60, 0, 1, 90), N(64, 1, 2, 90)]) == 1      # touching, not overlapping
    assert M.max_polyphony([N(60, 0, 1, 90), N(64, 0.5, 2, 90), N(67, 0.6, 0.9, 90)]) == 3


def test_chordal_string_parts_never_become_solo_instruments():
    leaves = ["strings.bowed.violin", "strings.bowed.viola", "strings.bowed.cello",
              "strings.bowed.section"]
    chords = [M.Note(p, t, t + 0.9, 90) for t in np.arange(0, 9, 1.0) for p in (60, 64, 67)]
    double = [M.Note(p, t, t + 0.9, 90) for t in np.arange(0, 9, 1.0) for p in (60, 67)]
    rng = np.random.default_rng(0)
    bal = M.LeafBalancer(leaves, rng)
    got_chords = {bal.choose(M.Part(0, 40, False, "strings_solo", chords)) for _ in range(40)}
    assert got_chords == {"strings.bowed.section"}
    got_double = {bal.choose(M.Part(0, 40, False, "strings_solo", double)) for _ in range(40)}
    assert got_double & {"strings.bowed.violin", "strings.bowed.viola", "strings.bowed.cello"}


# ------------------------------------------------------------------- registry gate
def _reg_doc(tmp_path, extra):
    base = [{"id": f"fake_{i}", "name": "x", "kind": "synth", "engine": "fake", "leaves": [l],
             "license": "CC0-1.0", "commercial_clean": True} for i, l in enumerate(V1)]
    return {"version": 1, "root": str(tmp_path / "src"), "v1_leaves": V1, "v8_leaves": NEW,
            "sources": base + extra}


def _save(tmp_path, doc, name="reg.yaml"):
    import yaml
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


def _synth(sid, leaves, **kw):
    return {"id": sid, "name": sid, "kind": "synth", "engine": "fake", "leaves": leaves,
            "license": "CC0-1.0", "commercial_clean": True, **kw}


def test_gate_counts_distinct_present_exact_instruments(tmp_path, tax):
    extra = [
        _synth("vln_a", ["strings.bowed.violin"]),
        _synth("vln_a_vib", ["strings.bowed.violin"], group="vln_a"),          # same instrument
        _synth("sec_a", ["strings.bowed.section"]),
        _synth("sec_b", ["strings.bowed.section"]),
        _synth("tbn_a", ["brass.trombone"]),
        _synth("tbn_pending", ["brass.trombone"], status="pending_approval"),  # not approved
        _synth("tbn_coarse", ["brass.trombone"], label_as="brass"),             # coarse: no leaf label
        {"id": "tamb_missing", "name": "t", "kind": "sfz", "engine": "fake", "path": "nope.sfz",
         "leaves": ["percussion.tambourine"], "license": "CC0-1.0", "commercial_clean": True},
        _synth("tamb_a", ["percussion.tambourine"]),
        _synth("clap_a", ["percussion.claps"]), _synth("clap_b", ["percussion.claps"]),
    ]
    reg = load_registry(_save(tmp_path, _reg_doc(tmp_path, extra)), tax)
    gate = leaf_gate(reg, reg.v8_leaves)
    assert gate["strings.bowed.violin"]["instruments"] == ["vln_a"]
    assert not gate["strings.bowed.violin"]["trainable"]
    assert gate["strings.bowed.section"]["trainable"]
    assert gate["brass.trombone"] == {"instruments": ["tbn_a"], "pending": ["tbn_pending"],
                                      "coarse_sources": ["tbn_coarse"], "trainable": False}
    assert gate["percussion.tambourine"]["instruments"] == ["tamb_a"]           # missing file
    assert gate["percussion.tambourine"]["pending"] == ["tamb_missing"]
    leaves, _ = resolve_leaf_set(reg, "v8")
    assert leaves == tuple(V1) + ("strings.bowed.section", "percussion.claps")
    assert resolve_leaf_set(reg, "v1") == (tuple(V1), {})
    assert resolve_leaf_set(reg, "v8", min_instruments=1)[0][len(V1):] == (
        "strings.bowed.violin", "strings.bowed.section", "brass.trombone",
        "percussion.tambourine", "percussion.claps")
    with pytest.raises(ValueError):
        resolve_leaf_set(reg, "v9")


def test_registry_rejects_bad_v8_entries(tmp_path, tax):
    doc = _reg_doc(tmp_path, [_synth("x", ["brass.trombone"], status="maybe")])
    with pytest.raises(SourceError, match="status"):
        load_registry(_save(tmp_path, doc), tax)
    doc = _reg_doc(tmp_path, [])
    doc["v8_leaves"] = ["keys.piano"]
    with pytest.raises(SourceError, match="already a v1 leaf"):
        load_registry(_save(tmp_path, doc), tax)
    doc = _reg_doc(tmp_path, [{"id": "gm", "name": "gm", "kind": "sf2", "path": "gm.sf2",
                               "leaves": ["brass.trombone"], "programs": {},
                               "license": "MIT", "commercial_clean": True}])
    with pytest.raises(SourceError, match="no programs entry"):
        load_registry(_save(tmp_path, doc), tax)


def test_v8_holdout_rule(tmp_path, tax):
    extra = [_synth("fluidr3_gm", ["percussion.claps", "strings.bowed.section"]),
             _synth("clap_b", ["percussion.claps"]), _synth("clap_a", ["percussion.claps"]),
             _synth("sec_a", ["strings.bowed.section"])]
    reg = load_registry(_save(tmp_path, _reg_doc(tmp_path, extra)), tax)
    # claps: 3 instruments -> hold out the first non-GM-bank group; section: 2 -> none
    assert v8_holdout(reg) == ["clap_a:percussion.claps"]


def test_build_refuses_holdout_that_starves_a_new_leaf(tmp_path, tax):
    from disstruments.ml.synth.build import build
    root = tmp_path / "midi"
    _write(root / "A" / "s.mid", [_track(48, _beats([60, 64]))])
    extra = [_synth("sec_a", ["strings.bowed.section"]), _synth("sec_b", ["strings.bowed.section"])]
    p = _save(tmp_path, _reg_doc(tmp_path, extra))
    with pytest.raises(ValueError, match="training instrument"):
        build(root, tmp_path / "out", 2, sr=16000, registry_path=str(p), fake=True,
              leaf_set="v8", holdout=["sec_a:strings.bowed.section"])


# ------------------------------------------------------------------- end to end (fake engine)
def test_v8_build_renders_new_families_and_cymbals_stay_kit_only(tmp_path, tax):
    from disstruments.ml.datasets import load_dataset
    from disstruments.ml.synth.build import build
    root = tmp_path / "midi"
    for k in range(6):
        _write(root / f"Band {k}" / "song.mid", [
            _track(48, _beats([60, 64, 67])),                                    # string ensemble
            _track(57, _beats([48, 50])),                                        # trombone
            _track(0, _beats([42, 36, 38, 54, 39], step=0.1), drum=True)])       # kit + tamb + clap
    _write(root / "Perc Only" / "p.mid", [_track(0, _beats([54, 39, 70], step=0.1), drum=True)])
    extra = [_synth(f"{l}_{i}", [l]) for l in NEW for i in (0, 1)]
    p = _save(tmp_path, _reg_doc(tmp_path, extra))
    out = tmp_path / "out"
    m = build(root, out, 30, seed=0, sr=16000, registry_path=str(p), fake=True,
              leaf_set="v8", windows_per_file=3)
    assert m["leaf_set"] == "v8" and set(NEW) <= set(m["active_leaves"])
    assert set(m["leaf_gate"]) == set(NEW)
    labs = [json.loads(f.read_text()) for f in (out / "clips").glob("*/labels.json")]
    seen = {l for lab in labs for l in lab["positive_leaves"]}
    assert {"strings.bowed.section", "brass.trombone", "percussion.tambourine",
            "percussion.claps"} <= seen
    for lab in labs:
        leaves = set(lab["positive_leaves"])
        kit_hats = any(s["leaf"].startswith("drums") for s in lab["stems"])
        # cymbals only ever come from a rendered kit part with hat/cymbal notes
        assert ("cymbals" in leaves) == kit_hats, lab["clip_id"]
        for s in lab["stems"]:
            if s["leaf"].startswith("percussion."):
                role = s["midi"]["role"]
                assert role.startswith("perc_") and s["midi"]["is_drum"]
    idx = load_dataset("synthetic", out)
    assert all(r.observed == frozenset(tax.nodes) for r in idx.records)


# ------------------------------------------------------------------- notes engine: one-shots
def test_notes_engine_plays_drum_one_shots_key_exact(tmp_path):
    import soundfile as sf

    from disstruments.ml.synth.engines import NotesEngine
    from disstruments.ml.synth.sources import Source
    d = tmp_path / "notes"
    d.mkdir()
    sr = 16000
    hit = (np.random.default_rng(0).standard_normal(int(0.8 * sr)) * np.exp(-np.arange(int(0.8 * sr)) / (0.3 * sr))) * 0.5
    sf.write(d / "tamb-054-100.wav", hit.astype(np.float32), sr)
    src = Source(id="t", name="t", kind="notes", engine="notes", leaves=("percussion.tambourine",),
                 license="CC0-1.0", commercial_clean=True, path="notes", instrument="tamb")
    part = M.Part(0, 0, True, "perc_tambourine",
                  [M.Note(54, 0.1, 0.15, 100), M.Note(55, 0.5, 0.55, 100), M.Note(53, 0.7, 0.75, 100)],
                  leaf="percussion.tambourine")
    y, meta = NotesEngine().render(part, src, tmp_path, sr, 2.0, np.random.default_rng(0))
    assert meta["oneshot"] is True and meta["octave_shift"] == 0
    assert meta["skipped_out_of_range"] == 2                 # 55/53: no neighbour-key resampling
    assert np.abs(y[int(0.1 * sr):int(0.12 * sr)]).max() > 0.01
    assert np.abs(y[int(0.4 * sr):int(0.45 * sr)]).max() > 1e-4      # rings past the 50 ms note


# ------------------------------------------------------------------- note prep (no downloads)
def test_vcsl_tokens_and_velocity_slots():
    from disstruments.ml.synth.notes_prep import free_velocity, vcsl_tokens
    assert vcsl_tokens("glock_loud_C6_01") == (84, 110, 1)
    assert vcsl_tokens("Vibes_hard_A2_v2_rr1_Main") == (45, 74, 1)
    assert vcsl_tokens("KSHarp_A2_mf1") == (45, 80, 1)
    assert vcsl_tokens("Marimba_hit_Outrigger_B2_med_01") == (47, 80, 1)
    assert vcsl_tokens("Hohner-Special20_Normal_C3") == (48, 85, 1)
    assert vcsl_tokens("Clap_rr3") == (None, 80, 3)
    assert vcsl_tokens("Tamb1_Hit_v1_rr2_Mid") == (None, 52, 2)
    taken: set[int] = set()
    assert [free_velocity(taken, 80) for _ in range(3)] == [80, 81, 82]
    full = set(range(1, 128))
    full.discard(5)
    assert free_velocity(full, 127) == 5


def test_pitch_check_octave_tolerance():
    from disstruments.ml.synth.notes_prep import pitch_ok
    assert pitch_ok(69.2, 69) and not pitch_ok(70.0, 69) and not pitch_ok(None, 69)
    assert not pitch_ok(81.1, 69) and pitch_ok(81.1, 69, octave_ok=True)
    assert pitch_ok(56.6, 69, octave_ok=True)                              # an octave + 0.4 st
    assert not pitch_ok(56.2, 69, octave_ok=True)                          # an octave + 0.8 st


def test_iowa_v8_file_filters():
    import re

    from disstruments.ml.synth.notes_prep import IOWA_V8_ONESHOTS, IOWA_V8_PITCH_PAGES

    def picks(inst, name):
        _, inc, exc, _, _ = IOWA_V8_PITCH_PAGES[inst]
        return bool(re.search(inc, name, re.I)) and not (exc and re.search(exc, name, re.I))
    assert picks("iowa_violin_arco", "Violin.arco.ff.sulA.A4.stereo.aif")
    assert not picks("iowa_violin_arco", "Violin.pizz.ff.sulA.A4.stereo.aif")
    assert picks("iowa_xylophone", "Xylophone.hardrubber.ff.A4.stereo.aif")
    assert not picks("iowa_xylophone", "Xylophone.gliss.up.stereo.aif")
    assert picks("iowa_vibraphone", "Vibraphone.sustain.ff.A3.stereo.aif")
    assert not picks("iowa_vibraphone", "Vibraphone.shortsustain.ff.A3.stereo.aif")
    assert not picks("iowa_vibraphone", "Vibraphone.bow.A3.stereo.aif")
    assert picks("iowa_bells", "bells.brass.ff.A5.stereo.aif")
    _, inc, keys = IOWA_V8_ONESHOTS["iowa_tambourine2"]
    assert keys == (54,) and re.search(inc, "tambourine2.normal.ff.stereo.aif")
    assert not re.search(inc, "tambourine2.fingerroll.ff.stereo.aif")
    assert not re.search(inc, "tambourine1.normal.ff.stereo.aif")


def test_nsynth_prep_merges_instrument_metadata(tmp_path):
    from disstruments.ml.synth.notes_prep import nsynth
    src, out = tmp_path / "ns", tmp_path / "notes"
    (src / "nsynth-valid" / "audio").mkdir(parents=True)
    (src / "nsynth-valid" / "examples.json").write_text(json.dumps({
        "brass_acoustic_000-060-100": {"instrument_str": "brass_acoustic_000", "pitch": 60,
                                       "instrument_family_str": "brass", "instrument_source_str": "acoustic"}}))
    (src / "nsynth-valid" / "audio" / "brass_acoustic_000-060-100.wav").write_bytes(b"x")
    out.mkdir()
    (out / "nsynth_instruments.json").write_text(json.dumps({"bass_acoustic_000": {"pitches": [40]}}))
    r = nsynth(src, out, ("brass",))
    meta = json.loads((out / "nsynth_instruments.json").read_text())
    assert set(meta) == {"bass_acoustic_000", "brass_acoustic_000"} and r["instruments_total"] == 2


# ------------------------------------------------------------------- packaged registry
def test_packaged_registry_v8(tax):
    from disstruments.ml.synth.sources import REGISTRY_PATH
    reg = load_registry(REGISTRY_PATH, tax)
    assert len(reg.v8_leaves) == 28 and not set(reg.v8_leaves) & set(reg.v1_leaves)
    reachable = {l for ls in M.ROLE_LEAVES.values() for l in ls}
    assert set(reg.v8_leaves) <= reachable                 # every new leaf has a MIDI role
    gm = reg.by_id("fluidr3_gm")
    for leaf in reg.v8_leaves:
        if leaf.startswith("percussion.") and not leaf.startswith("percussion.mallet"):
            continue                                       # bank-128 kits, not programs
        for prog in gm.programs.get(leaf, ()):
            assert M.PROGRAM_HINTS.get(prog) == (leaf,), (leaf, prog)   # GM bank = GM name
    for s in reg.sources:
        if s.status:
            assert s.download.get("bytes"), s.id           # size known for the approval list
            assert s.commercial_clean, s.id                # nothing NC is proposed for v8
    # Iowa saxes are exact v8 leaves now (no longer coarse tenor stand-ins)
    assert reg.by_id("iowa_altosax").leaves == ("woodwinds.saxophone.alto",)
    assert reg.by_id("iowa_altosax").label_as is None
