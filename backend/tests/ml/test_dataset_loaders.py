"""MedleyDB / OpenMIC / Slakh loaders against real-format metadata fixtures (no audio)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from disstruments.ml.datasets import UnknownLabelsError, load_dataset
from disstruments.ml.datasets import slakh as slakh_mod
from disstruments.ml.datasets.base import artist_leakage
from disstruments.ml.taxonomy import get_taxonomy

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ml"
MDB = FIX / "medleydb"
OM = FIX / "openmic" / "openmic-2018"
SLAKH = FIX / "slakh" / "slakh2100_flac_redux"


# ---------------------------------------------------------------------- MedleyDB
@pytest.fixture(scope="module")
def mdb():
    # fixture set includes 2 non-public tracks, so it cannot use the pinned split
    return load_dataset("medleydb", MDB, generated_split=True)


def test_medleydb_real_track_labels(mdb):
    r = next(x for x in mdb.records if x.item_id == "AClassicEducation_NightOwl")
    assert r.artist == "aclassiceducation" and r.extra["genre"] == "Singer/Songwriter"
    expected = {"bass.electric", "drums.acoustic_kit", "guitar.electric.distorted",
                "guitar.electric.clean", "voice.sung",
                "keys.synth", "percussion.tambourine"}
    assert expected <= r.positive
    assert "cymbals" not in r.observed                                # 3.0.0: OpenMIC-only
    assert r.positive == r.positive | {a for n in r.positive for a in _anc(n)}   # ancestor-closed
    # coarse sources -> unknown, not negative
    for n in ("bass.electric.fingered", "keys.synth.pad", "bass.synth"):
        assert n not in r.observed, n
    # exhaustive annotation -> known negatives
    for n in ("keys.piano", "brass", "strings", "voice.rap"):
        assert n in r.negative, n
    assert "fx/processed sound" in r.source_labels and not r.unknown_labels
    assert len(r.stems) == 13 and r.stems[0].stem_id == "S01"
    assert r.stems[0].positive == {"bass", "bass.electric"}


def _anc(n):
    segs = n.split(".")
    return {".".join(segs[:i]) for i in range(1, len(segs))}


def test_medleydb_list_valued_instruments(mdb):
    r = next(x for x in mdb.records if x.item_id == "AHa_TakeOnMe")
    assert {"voice.sung", "voice"} <= r.positive


def test_medleydb_artist_normalization_and_disjoint_split(mdb):
    by_id = {r.item_id: r for r in mdb.records}
    assert by_id["MusicDelta_Beatles"].artist == by_id["MusicDelta_BebopJazz"].artist == "musicdelta"
    a1 = by_id["DeclareAString_MendelssohnPianoTrio1Movement1"]
    a2 = by_id["DeclareAString_MendelssohnPianoTrio1Movement2"]
    assert a1.artist == a2.artist and a1.split == a2.split
    assert artist_leakage(mdb.records) == {}
    assert set(mdb.split_counts()) == {"train", "val", "test"}
    again = load_dataset("medleydb", MDB, generated_split=True)
    assert [(r.item_id, r.split) for r in again.records] == [(r.item_id, r.split) for r in mdb.records]
    for seed in range(5):
        assert artist_leakage(load_dataset("medleydb", MDB, seed=seed, generated_split=True).records) == {}


def test_medleydb_main_system_ignored_not_unknown(mdb):
    r = next(x for x in mdb.records if x.item_id == "AmadeusRedux_MozartAllegro")
    assert "Main System" in r.source_labels and not r.unknown_labels
    assert "guitar" in r.negative


def _copy_mdb(tmp_path, n=3):
    d = tmp_path / "Metadata"
    d.mkdir()
    for f in sorted(MDB.glob("Metadata/*.yaml"))[:n]:
        shutil.copy(f, d)
    return d


def test_medleydb_strict_raises_with_full_list(tmp_path):
    d = _copy_mdb(tmp_path)
    f = sorted(d.glob("*.yaml"))[0]
    meta = yaml.safe_load(f.read_text())
    stems = list(meta["stems"])
    meta["stems"][stems[0]]["instrument"] = "kazoo"
    meta["stems"][stems[1]]["instrument"] = ["theremin", "hurdy gurdy"]
    f.write_text(yaml.safe_dump(meta))
    with pytest.raises(UnknownLabelsError) as e:
        load_dataset("medleydb", d, generated_split=True)
    assert set(e.value.labels) == {"kazoo", "hurdy gurdy"}           # theremin is ignored, not unknown
    idx = load_dataset("medleydb", d, strict=False, generated_split=True)
    r = next(x for x in idx.records if x.item_id == f.name[:-len("_METADATA.yaml")])
    assert set(r.unknown_labels) == {"kazoo", "hurdy gurdy"}
    assert r.observed == r.positive                                   # everything else masked
    assert idx.stats["unknown_labels"] == {"kazoo": 1, "hurdy gurdy": 1}


def test_medleydb_audio_root_and_split_file(tmp_path):
    d = _copy_mdb(tmp_path, n=3)
    ids = sorted(p.name[:-len("_METADATA.yaml")] for p in d.glob("*.yaml"))
    audio = tmp_path / "Audio"
    meta = yaml.safe_load((d / f"{ids[0]}_METADATA.yaml").read_text())
    tdir = audio / ids[0]
    (tdir / meta["stem_dir"]).mkdir(parents=True)
    (tdir / meta["mix_filename"]).touch()                             # empty placeholder, not audio
    first_stem = sorted(meta["stems"])[0]
    (tdir / meta["stem_dir"] / meta["stems"][first_stem]["filename"]).touch()
    (audio / ids[1]).mkdir()
    idx = load_dataset("medleydb", d, audio_root=audio, generated_split=True)
    assert idx.stats["skipped_no_audio"] == 1 and len(idx.records) == 2
    r = next(x for x in idx.records if x.item_id == ids[0])
    assert r.audio["mix"] == tdir / meta["mix_filename"] and first_stem in r.audio
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({ids[0]: "test", ids[1]: "train", ids[2]: "val"}))
    pinned = load_dataset("medleydb", d, split_file=split_file)
    assert {r.item_id: r.split for r in pinned.records} == {ids[0]: "test", ids[1]: "train", ids[2]: "val"}
    split_file.write_text(json.dumps({ids[0]: "test"}))
    with pytest.raises(ValueError, match="not in split file"):
        load_dataset("medleydb", d, split_file=split_file)


# ---------------------------------------------------------------------- OpenMIC
@pytest.fixture(scope="module")
def om():
    return load_dataset("openmic", OM)


def test_openmic_partial_label_semantics(om):
    r = next(x for x in om.records if x.item_id == "000046_3840")
    # guitar 1.0, voice 0.8 positive; piano 0.0 negative; everything else unannotated
    assert r.positive == {"guitar", "voice"}
    assert "keys.piano" in r.negative                                 # acoustic piano leaf
    # 3.0.0: OpenMIC `piano` is acoustic piano; it says nothing about electric pianos
    assert not set(get_taxonomy().descendants("keys.electric_piano", include_self=True)) & r.observed
    assert "keys" not in r.observed                                   # parent unknown
    assert "guitar.electric" not in r.observed                        # coarse positive: child unknown
    assert "drums" not in r.observed                                  # not annotated
    assert r.artist == "fma:4" and r.split == "train"
    assert r.source_labels == ("guitar=1", "piano=0", "voice=1")


def test_openmic_binarization_boundary(om):
    r = next(x for x in om.records if x.item_id == "000139_119040")
    assert "voice" in r.positive                                      # 0.5 >= 0.5
    r = next(x for x in om.records if x.item_id == "000312_184320")
    assert "keys.piano" in r.negative                                 # 0.4
    idx = load_dataset("openmic", OM, binarize_threshold=0.9)
    r = next(x for x in idx.records if x.item_id == "000139_119040")
    assert "voice" in r.negative


def test_openmic_official_split_is_artist_disjoint(om):
    assert om.split_counts() == {"train": 7, "test": 5}
    assert artist_leakage(om.records) == {}
    flagged = [r.item_id for r in om.records if r.extra["audio_corrupt"]]
    assert flagged == ["071826_3840"]


def test_openmic_val_carveout_is_artist_disjoint():
    idx = load_dataset("openmic", OM, val_fraction=0.4)
    assert set(idx.split_counts()) == {"train", "val", "test"}
    assert artist_leakage(idx.records) == {}


def _copy_om(tmp_path):
    d = tmp_path / "openmic-2018"
    shutil.copytree(OM, d)
    return d


def test_openmic_readme_partition_names_and_bytes_keys(tmp_path):
    d = _copy_om(tmp_path)
    for split, new in (("train", "train01.txt"), ("test", "test01.txt")):
        (d / "partitions" / f"split01_{split}.csv").rename(d / "partitions" / new)
    with np.load(d / "openmic-2018.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["sample_key"] = arrays["sample_key"].astype("S")          # bytes variant
    np.savez(d / "openmic-2018.npz", **arrays)
    idx = load_dataset("openmic", d)
    assert idx.split_counts() == {"train": 7, "test": 5}


def test_openmic_object_dtype_sample_key(tmp_path):
    # official helper_numpy.py builds sample_key as an object array -> pickled in the npz
    d = _copy_om(tmp_path)
    with np.load(d / "openmic-2018.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["sample_key"] = np.array([str(k) for k in arrays["sample_key"]], dtype=object)
    np.savez(d / "openmic-2018.npz", **arrays)
    idx = load_dataset("openmic", d)
    assert idx.stats["sample_key_pickled"] is True
    assert idx.split_counts() == {"train": 7, "test": 5}
    assert load_dataset("openmic", OM).stats["sample_key_pickled"] is False


def test_openmic_rejects_inconsistent_files(tmp_path):
    d = _copy_om(tmp_path)
    (d / "partitions" / "split01_test.csv").write_text("999999_0\n")
    with pytest.raises(ValueError, match="missing from npz"):
        load_dataset("openmic", d)


# ---------------------------------------------------------------------- Slakh
def test_slakh_redux_split_and_labels():
    idx = load_dataset("slakh", SLAKH)
    by = {r.item_id: r for r in idx.records}
    assert "Track01939" not in by                                     # redux-omitted duplicate
    assert {k: r.split for k, r in by.items()} == {
        "Track00001": "train", "Track00003": "train", "Track00004": "train",
        "Track01501": "val", "Track01876": "test"}
    assert all(r.artist is None for r in idx.records)
    r = by["Track00004"]
    assert {"guitar.acoustic.nylon", "bass.electric.slap", "drums.acoustic_kit"} <= r.positive
    assert "strings.bowed" not in r.observed and "bass.upright" not in r.observed   # solo_strings
    assert "keys" in r.negative
    assert len(r.stems) == 4                                          # unrendered S04 skipped
    assert "guitar.acoustic.steel" in by["Track01876"].positive       # full-path AGML2.component
    assert {"keys.electric_piano.wurlitzer", "keys.organ.drawbar"} <= by["Track01501"].positive
    assert "keys.piano" in by["Track01501"].negative                  # EP is not piano (3.0.0)
    assert not any("cymbals" in r.observed for r in idx.records)      # 3.0.0: OpenMIC-only
    # real Track00001: program 22 "Harmonica" rendered by an organ patch -> organ, not harmonica
    assert "keys.organ.combo" in by["Track00001"].positive
    assert "woodwinds.harmonica" in by["Track00001"].negative
    assert idx.stats["midi_md5_cross_split"] == 0
    assert idx.stats["msd_id_cross_split"] == 1                       # same song, different MIDI


def test_slakh_split_modes():
    s2 = load_dataset("slakh", SLAKH, split_mode="split2")
    by = {r.item_id: r.split for r in s2.records}
    assert by["Track01939"] == "train" and by["Track00003"] == "test"
    assert s2.stats["midi_md5_cross_split"] == 0
    orig = load_dataset("slakh", SLAKH, split_mode="orig")
    assert {r.item_id: r.split for r in orig.records}["Track01939"] == "test"
    assert orig.stats["midi_md5_cross_split"] == 1                    # the leak redux removes
    inc = load_dataset("slakh", SLAKH, include_omitted=True)
    assert {r.item_id: r.split for r in inc.records}["Track01939"] == "omitted"


def test_slakh_vendored_tables_match_published_counts():
    from collections import Counter
    ids = [f"Track{i:05d}" for i in range(1, 2101)]
    redux = Counter(slakh_mod.canonical_split(t, "redux") for t in ids)
    assert redux == {"train": 1289, "val": 270, "test": 151, "omitted": 390}
    assert Counter(slakh_mod.canonical_split(t, "split2") for t in ids) == {"train": 1500, "val": 375, "test": 225}
    assert Counter(slakh_mod.canonical_split(t, "orig") for t in ids) == {"train": 1500, "val": 375, "test": 225}


def _copy_slakh(tmp_path):
    d = tmp_path / "slakh"
    shutil.copytree(SLAKH, d)
    return d


def test_slakh_unknown_plugin_strict_and_fallback(tmp_path):
    d = _copy_slakh(tmp_path)
    f = d / "train" / "Track00004" / "metadata.yaml"
    meta = yaml.safe_load(f.read_text())
    meta["stems"]["S00"]["plugin_name"] = "mystery_guitar.nkm"
    f.write_text(yaml.safe_dump(meta))
    with pytest.raises(UnknownLabelsError, match="mystery_guitar"):
        load_dataset("slakh", d)
    idx = load_dataset("slakh", d, strict=False)
    r = next(x for x in idx.records if x.item_id == "Track00004")
    assert r.unknown_labels == ("plugin:mystery_guitar.nkm",)
    assert "guitar" in r.positive and "guitar.acoustic" not in r.observed   # inst_class fallback
    assert idx.stats["n_plugin_fallbacks"] == 1


def test_slakh_babyslakh_rendered_flag_quirk(tmp_path):
    d = tmp_path / "baby" / "Track00002"
    d.mkdir(parents=True)
    meta = yaml.safe_load((SLAKH / "train" / "Track00004" / "metadata.yaml").read_text())
    for s in meta["stems"].values():
        s["audio_rendered"] = False                                   # as shipped in BabySlakh
    (d / "metadata.yaml").write_text(yaml.safe_dump(meta))
    idx = load_dataset("slakh", tmp_path / "baby", split_mode="orig")
    r = idx.records[0]
    assert len(r.stems) == 4 and idx.stats["n_rendered_flag_overrides"] == 4   # S04 (None) still skipped
    assert r.extra["dir_split"] is None and r.split == "train"


def test_slakh_bad_layout_raises(tmp_path):
    d = _copy_slakh(tmp_path)
    (d / "test").mkdir(exist_ok=True)
    shutil.move(str(d / "train" / "Track00001"), str(d / "test" / "Track00001"))
    with pytest.raises(ValueError, match="no known split scheme"):
        load_dataset("slakh", d)
