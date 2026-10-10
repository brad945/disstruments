"""IDMT-SMT loaders (test-only datasets): label semantics on tiny fixtures that mimic the
real filename / annotation formats (no real audio)."""
import pytest

from disstruments.ml.datasets import DATASETS, load_dataset
from disstruments.ml.taxonomy import Taxonomy

TAX = Taxonomy.load()
TECH = {"bass.electric.fingered", "bass.electric.picked", "bass.electric.slap"}


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"RIFF")


def _by(idx):
    return {r.item_id: r for r in idx.records}


def test_registered():
    assert {"idmt_bass", "idmt_bass_tracks", "idmt_fx"} <= set(DATASETS)


# ---------------------------------------------------------------- IDMT-SMT-Bass
@pytest.fixture
def bass_root(tmp_path):
    for rel in ["PS/FS/BS_1_EQ_1_FS_NO_1_0.wav", "PS/PK/BS_2_EQ_3_PK_NO_2_5.wav",
                "PS/SP/BS_3_EQ_2_SP_NO_3_7.wav", "PS/ST/BS_1_EQ_2_ST_NO_4_12.wav",
                "PS/MU/BS_2_EQ_1_MU_NO_1_3.wav", "ES/SLU/BS_4_EQ_1_PS_FS_ES_SLU_1_13.wav",
                "ES/VI/BS_1_EQ_1_PS_FS_ES_VI_1_3.wav"]:          # ES/VI: not loaded
        _touch(tmp_path / rel)
    return tmp_path


def test_bass_semantics(bass_root):
    idx = load_dataset("idmt_bass", bass_root)
    r = _by(idx)
    assert len(r) == 6 and all(x.split == "test" for x in r.values())
    fs = r["PS/FS/BS_1_EQ_1_FS_NO_1_0"]
    assert fs.positive == {"bass", "bass.electric", "bass.electric.fingered"}
    assert fs.observed == frozenset(TAX.nodes)                  # solo note: fully observed
    assert {"bass.electric.picked", "bass.electric.fretless", "cymbals", "guitar"} <= fs.negative
    assert fs.artist == "fame_baphomet_4_ntb" and fs.extra["midi_pitch"] == 28
    assert "bass.electric.slap" in r["PS/ST/BS_1_EQ_2_ST_NO_4_12"].positive
    assert "bass.electric.slap" in r["PS/SP/BS_3_EQ_2_SP_NO_3_7"].positive
    assert "bass.electric.picked" in r["PS/PK/BS_2_EQ_3_PK_NO_2_5"].positive
    mu = r["PS/MU/BS_2_EQ_1_MU_NO_1_3"]
    assert mu.positive == {"bass", "bass.electric"}
    assert not (TECH & mu.observed)                             # technique unknown
    assert "bass.electric.fretless" in mu.negative and "guitar" in mu.negative
    fl = r["ES/SLU/BS_4_EQ_1_PS_FS_ES_SLU_1_13"]
    assert "bass.electric.fretless" in fl.positive and not (TECH & fl.observed)
    assert idx.stats["n_groups"] == 4
    assert len(load_dataset("idmt_bass", bass_root, include_fretless=False).records) == 5


def test_bass_folder_filename_mismatch(bass_root):
    _touch(bass_root / "PS/FS/BS_1_EQ_1_PK_NO_1_1.wav")
    with pytest.raises(ValueError, match="folder style"):
        load_dataset("idmt_bass", bass_root)
    idx = load_dataset("idmt_bass", bass_root, strict=False)
    assert len(idx.records) == 6 and idx.stats["skipped"]


# ---------------------------------------------------------------- Single-Track
_EVENT = """<event><pitch>{p}</pitch><onsetSec>{on}</onsetSec><offsetSec>{off}</offsetSec>
<fretNumber>3</fretNumber><stringNumber>1</stringNumber><excitationStyle>{s}</excitationStyle>
<expressionStyle>NO</expressionStyle><modulationFrequencyRange>0</modulationFrequencyRange>
<modulationFrequency>0</modulationFrequency></event>"""


def _track(root, tid, styles, model="Fame Baphomet 4 NTB", audio=True):
    ev = "".join(_EVENT.format(p=31, on=i, off=i + .5, s=s) for i, s in enumerate(styles))
    xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><instrumentRecording>'
           f"<globalParameter><audioFileName>{tid}.wav</audioFileName><instrument>EBSS</instrument>"
           f"<instrumentModel>{model}</instrumentModel><recordingArtist>X</recordingArtist>"
           f"</globalParameter><transcription>{ev}</transcription></instrumentRecording>")
    (root / "annotation").mkdir(parents=True, exist_ok=True)
    (root / "annotation" / f"{tid}.xml").write_text(xml)
    if audio:
        _touch(root / "audio" / f"{tid}.wav")


def test_bass_tracks(tmp_path):
    _track(tmp_path, "001", ["MU", "MU"])
    _track(tmp_path, "002", ["FS", "FS"])
    _track(tmp_path, "007", ["SP", "ST", "SP"])
    _track(tmp_path, "009", ["FS", "MU"], audio=False)
    r = _by(load_dataset("idmt_bass_tracks", tmp_path))
    assert r["002"].positive == {"bass", "bass.electric", "bass.electric.fingered"}
    assert {"bass.electric.picked", "bass.electric.slap", "drums"} <= r["002"].negative
    assert "bass.electric.slap" in r["007"].positive and "bass.electric.fingered" in r["007"].negative
    assert not (TECH & r["001"].observed) and "bass.electric.fretless" in r["001"].negative
    mixed = r["009"]                       # FS positive stays; others masked by the MU notes
    assert "bass.electric.fingered" in mixed.positive
    assert not ({"bass.electric.picked", "bass.electric.slap"} & mixed.observed)
    assert "mix" not in mixed.audio and "mix" in r["001"].audio
    assert r["001"].artist == "fame_baphomet_4_ntb"                 # shared with idmt_bass BS_1
    idx = load_dataset("idmt_bass_tracks", tmp_path)
    assert idx.stats["n_groups"] == 1 and "bootstrap_warning" in idx.stats
    assert idx.stats["missing_audio"] == ["009.wav"]
    t = load_dataset("idmt_bass_tracks", tmp_path, group_by="track")
    assert t.stats["n_groups"] == 4


def test_bass_tracks_unknown_code(tmp_path):
    _track(tmp_path, "001", ["FS", "XX"])
    with pytest.raises(ValueError, match="excitationStyle"):
        load_dataset("idmt_bass_tracks", tmp_path)
    r = load_dataset("idmt_bass_tracks", tmp_path, strict=False).records[0]
    assert "bass.electric.fingered" in r.positive and "bass.electric.slap" not in r.observed


# ---------------------------------------------------------------- Audio-Effects
_LIST = """<?xml version="1.0" encoding="utf-8"?><content><listinformation><listID>L</listID>
</listinformation>{files}</content>"""
_AF = ("<audiofile><fileID>{fid}</fileID><instrument>{i}</instrument><instrumentsetting>{s}"
       "</instrumentsetting><playstyle>{p}</playstyle><midinr>{m}</midinr><fxgroup>{g}</fxgroup>"
       "<fxtype>{t}</fxtype><fxsetting>{fs}</fxsetting><filenr>{n}</filenr></audiofile>")


@pytest.fixture
def fx_root(tmp_path):
    files = {"Gitarre monophon/Samples/NoFX/G81-40100-1111-20001.wav",
             "Gitarre monophon/Samples/Distortion/G91-52100-4411-20500.wav",
             "Gitarre monophon/Samples/Overdrive/G82-60100-4421-20900.wav",
             "Gitarre monophon/Samples/Chorus/G81-40100-3311-21000.wav",
             "Gitarre polyphon/Samples/Distortion/P64-43110-4412-42865.wav",
             "Bass monophon/Samples/NoFX/B12-45402-1111-00001.wav",
             "Bass monophon/Samples/Distortion/B41-36203-4411-01057.wav"}
    for f in files:
        _touch(tmp_path / f)
    af = _AF.format(fid="P64-43110-4412-42865", i="P", s=6, p=4, m=43, g=4, t=41, fs=2, n=42865)
    (tmp_path / "Gitarre polyphon/Lists/Distortion").mkdir(parents=True)
    (tmp_path / "Gitarre polyphon/Lists/Distortion/P64-00000-4412-1.xml").write_text(_LIST.format(files=af))
    return tmp_path


def test_fx_semantics(fx_root):
    idx = load_dataset("idmt_fx", fx_root)
    r = _by(idx)
    assert len(r) == 6 and idx.stats["skipped"] == {"Chorus": 1} and idx.stats["xml_checked"] == 1
    clean = r["Gitarre monophon/NoFX/G81-40100-1111-20001"]
    assert clean.positive == {"guitar", "guitar.electric", "guitar.electric.clean"}
    assert {"guitar.electric.distorted", "bass", "cymbals"} <= clean.negative
    for k in ("Gitarre monophon/Distortion/G91-52100-4411-20500",
              "Gitarre monophon/Overdrive/G82-60100-4421-20900",
              "Gitarre polyphon/Distortion/P64-43110-4412-42865"):
        assert "guitar.electric.distorted" in r[k].positive
        assert "guitar.electric.clean" in r[k].negative
    b = r["Bass monophon/Distortion/B41-36203-4411-01057"]
    assert b.positive == {"bass", "bass.electric"}
    assert not (set(TAX.children("bass.electric")) & b.observed)
    assert {"bass.upright", "guitar"} <= b.negative
    assert clean.artist == "guitar_setting:8" and r["Gitarre polyphon/Distortion/P64-43110-4412-42865"].artist == "guitar_setting:6"
    assert b.artist == "bass_setting:4"
    assert len(load_dataset("idmt_fx", fx_root, include_poly=False).records) == 5


def test_fx_xml_mismatch(fx_root):
    # XML says fxtype 42 for a file whose name encodes 41 (4411 = group 4, type 41, setting 1) -> decoding disagreement is an error
    af = _AF.format(fid="G91-52100-4411-20500", i="G", s=9, p=1, m=52, g=4, t=42, fs=1, n=20500)
    (fx_root / "Gitarre monophon/Lists/Distortion").mkdir(parents=True)
    (fx_root / "Gitarre monophon/Lists/Distortion/x.xml").write_text(_LIST.format(files=af))
    with pytest.raises(ValueError, match="mismatch"):
        load_dataset("idmt_fx", fx_root)
    assert len(load_dataset("idmt_fx", fx_root, strict=False).records) == 5


def test_fx_folder_vs_fxtype_and_eq(fx_root):
    _touch(fx_root / "Gitarre monophon/Samples/EQ/G81-40100-1121-20003.wav")
    idx = load_dataset("idmt_fx", fx_root)
    assert idx.stats["skipped"] == {"Chorus": 1, "EQ": 1}
    eq = _by(load_dataset("idmt_fx", fx_root, eq_as_clean=True))["Gitarre monophon/EQ/G81-40100-1121-20003"]
    assert "guitar.electric.clean" in eq.positive
    _touch(fx_root / "Gitarre monophon/Samples/NoFX/G81-41100-4411-20004.wav")   # distortion code
    with pytest.raises(ValueError, match="fxtype"):
        load_dataset("idmt_fx", fx_root)


def test_fx_unknown_effect_folder(fx_root):
    _touch(fx_root / "Gitarre monophon/Samples/Wah/G81-40100-5511-20002.wav")
    with pytest.raises(ValueError, match="unknown effect"):
        load_dataset("idmt_fx", fx_root)


def test_missing_roots(tmp_path):
    for name in ("idmt_bass", "idmt_bass_tracks", "idmt_fx"):
        with pytest.raises(FileNotFoundError):
            load_dataset(name, tmp_path)
