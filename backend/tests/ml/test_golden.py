"""Golden set: labelling page generation and loader semantics."""
import json

from disstruments.ml.datasets import load_dataset
from disstruments.ml.golden.__main__ import build_page
from disstruments.ml.taxonomy import Taxonomy


def _write(root, name, **doc):
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    (root / "audio" / doc["audio_filename"]).write_bytes(b"x")
    (root / "labels" / f"{name}.labels.json").write_text(json.dumps(doc))


def test_page_embeds_taxonomy(tmp_path):
    page = build_page(tmp_path)
    html = page.read_text()
    assert "/*__LEAVES__*/" not in html and "keys.electric_piano.rhodes" in html
    assert (tmp_path / "labels").is_dir() and (tmp_path / "audio").is_dir()


def test_loader_semantics(tmp_path):
    tax = Taxonomy.load()
    _write(tmp_path, "a", audio_filename="a.mp3", artist="Band", title="Song", listened_fully=True,
           labels={"guitar.electric.distorted": {"state": "present", "intervals": [[1.0, 9.5]]},
                   "keys.electric_piano.rhodes": {"state": "unsure", "intervals": []}})
    _write(tmp_path, "b", audio_filename="b.mp3", artist="Other", title="Two", listened_fully=False,
           labels={"bass.electric.fingered": {"state": "present", "intervals": []}})
    idx = load_dataset("golden", tmp_path)
    a = next(r for r in idx.records if r.artist == "band")
    assert {"guitar", "guitar.electric", "guitar.electric.distorted"} <= a.positive
    assert "keys.electric_piano.rhodes" not in a.observed           # unsure -> unknown
    assert "guitar.electric.clean" in a.negative                    # fully listened -> absent is negative
    b = next(r for r in idx.records if r.artist == "other")
    assert "bass.electric.fingered" in b.positive and "bass" in b.observed
    assert "guitar.electric.clean" not in b.observed                # not fully listened -> unknown
    assert "mix" in a.audio and a.split == "test"
