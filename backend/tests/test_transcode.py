"""F2 canonical transcode keeps the upload's level (M5: mono was upmixed at -3 dB)."""
import shutil

import numpy as np
import pytest

from disstruments.pipeline.transcode import to_canonical

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.mark.parametrize("channels", [1, 2])
def test_canonical_keeps_level(tmp_path, channels):
    sf = pytest.importorskip("soundfile")
    t = np.arange(22050) / 22050
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    src = tmp_path / "in.wav"
    sf.write(src, y if channels == 1 else np.stack([y, y], 1), 22050)
    dst = tmp_path / "out.wav"
    to_canonical(src, dst)
    out, sr = sf.read(dst, always_2d=True)
    assert sr == 44100 and out.shape[1] == 2
    rms = lambda a: np.sqrt((a ** 2).mean())  # noqa: E731
    assert 20 * np.log10(rms(out.mean(1)) / rms(y)) == pytest.approx(0.0, abs=0.1)
