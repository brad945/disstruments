"""M5: package a from-scratch CNN checkpoint for the app (ONNX + a JSON sidecar).

The sidecar is the serving contract. It carries everything the app needs to reproduce the
eval-time numbers and to state honestly what the model can and can't say:

- `nodes`: output columns (taxonomy order); `trained`: nodes the render actually had
  positives for (closed upward). Scores on other nodes are meaningless and never shown.
- `temperature`: one global T fit on real held-out clips (OpenMIC train, never trained on;
  M3 C1), applied as sigmoid(logit / T). Only coarse nodes can be fit (OpenMIC has no
  leaf labels), so leaf confidences ship with `leaf_calibrated: false`.
- `routing`: which presence decisions this model owns, and which subtypes it may name
  under each incumbent (PANNs) label. Chosen from the M5 comparison, not by taste.
- `licence`: research-only until a model is trained on commercial_clean clips only.

    python -m disstruments.ml.models export-onnx --model .../scratch_cnn_synthetic_v7s.pt \\
        --manifest .../v7serve/manifest.json --calib-cache .../emb --out ~/.disstruments/models/leaf
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ...pipeline.leaf_tagging import DESCEND_MARGIN
from ..taxonomy import Taxonomy
from . import scratch as S

FORMAT = "disstruments.leafmodel/v1"
ONNX_NAME, SIDECAR_NAME = "model.onnx", "model.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_onnx(model, out: Path, opset: int = 17) -> Path:
    """Logits out (calibration is applied in numpy, so T can change without re-export)."""
    import torch
    model = model.to("cpu").eval()
    x = torch.zeros(1, S.N_MELS, S.FRAMES)
    torch.onnx.export(model, (x,), str(out), input_names=["logmel"], output_names=["logits"],
                      dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return out


def parity(model, onnx_path: Path, X: np.ndarray) -> float:
    """Max |sigmoid(torch) - sigmoid(onnx)| over `X` (B, mels, frames)."""
    import onnxruntime as ort
    import torch
    with torch.no_grad():
        zt = model.to("cpu").eval()(torch.from_numpy(X.astype(np.float32))).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    zo = sess.run(None, {"logmel": X.astype(np.float32)})[0]
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    return float(np.abs(sig(zt) - sig(zo)).max())


def trained_nodes(manifest: dict, tax: Taxonomy) -> list[str]:
    return sorted(tax.close_upward(n for n, c in manifest["leaf_counts"].items() if c > 0))


def fit_global_temperature(scores: np.ndarray, y: np.ndarray, m: np.ndarray, nodes: list[str],
                           trained: set[str]) -> dict:
    from .calibrate import fit_temperatures
    keep = np.array([n in trained for n in nodes])
    t = fit_temperatures(scores[:, keep], y[:, keep], m[:, keep], [0] * int(keep.sum()))[0]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}


# Routing chosen from the M5 comparison (docs/debriefs/m5.md). Keys are the incumbent's
# (PANNs) OpenMIC-20 labels; values are the deepest nodes this model may name under them
# (their ancestors may be named too). Rule: a subtype is listed if the model was trained
# on it, its parent has >= 2 trained children, and it is not measured at chance
# (normalized AP lift < 0.1 on pooled MoisesDB, or flat on unseen instruments). Not yet
# measured on real music = listed, but shown as uncertain until the golden set calibrates
# leaves. The app names at most one subtype per instrument (`leaf_tagging.descend`).
SUBTYPES: dict[str, list[str]] = {
    "guitar": ["guitar.acoustic.nylon", "guitar.acoustic.steel",
               "guitar.electric.clean", "guitar.electric.distorted"],
    "bass": ["bass.upright", "bass.electric", "bass.synth.analog", "bass.synth.sub_808"],
    "synthesizer": ["keys.synth.lead", "keys.synth.pluck"],
    "drums": ["drums.acoustic_kit", "drums.electronic"],   # tr808 is electronic's only trained child
}
SUPPRESSED: dict[str, str] = {
    "bass.electric.fingered": "flat on unseen basses and IDMT (M4 L3c); shown as electric bass",
    "bass.electric.picked": "flat on unseen basses and IDMT (M4 L3c); shown as electric bass",
    "keys.electric_piano": "at chance on real music (MoisesDB AP 0.16 vs 0.11 base rate)",
    "keys.synth.pad": "at chance on real music (MoisesDB AP 0.15 vs 0.15 base rate)",
}
# Presence owned by this model (beats the incumbent on real music, never worse): bass
# (MoisesDB +0.18 [+0.13, +0.24]; OpenMIC tie). Everything else: PANNs keeps presence.
PRESENCE: list[str] = ["bass"]


def sidecar(*, tax: Taxonomy, onnx_path: Path, ckpt_meta: dict, manifest: dict, trained: list[str],
            temperature: dict, parity_max: float, source: str) -> dict:
    return {
        "format": FORMAT,
        "name": "disstruments-cnn",
        "version": f"{Path(source).stem}",
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "onnx": ONNX_NAME,
        "sha256": sha256(onnx_path),
        "taxonomy_version": tax.version,
        "taxonomy_sha256": tax.sha256,
        "nodes": list(tax.nodes),
        "trained": trained,
        "features": {"sr": S.SR, "n_fft": S.N_FFT, "hop": S.HOP, "n_mels": S.N_MELS,
                     "window_s": S.SECONDS, "frames": S.FRAMES, "window_hop_s": S.SECONDS},
        "song_aggregate": "p95 over windows",
        "temperature": temperature,
        "leaf_calibrated": False,
        "onnx_torch_max_abs_diff": parity_max,
        "routing": {"presence": PRESENCE, "subtypes": SUBTYPES, "suppressed": SUPPRESSED,
                    "descend_margin": DESCEND_MARGIN},
        "train": {"n_params": ckpt_meta.get("n_params"), "cfg": ckpt_meta.get("cfg"),
                  "best_val_map": ckpt_meta.get("best_val_map"),
                  "renderer_sha": manifest.get("renderer_sha"), "n_rendered": manifest.get("n_rendered"),
                  "holdout": manifest.get("holdout", []), "fx_profile": manifest.get("fx_profile")},
        "licence": {
            "commercial_clean": bool(ckpt_meta.get("commercial_clean", False)),
            "status": "research-only (renders include non-commercial MIDI); see SYSTEMS_DESIGN §4",
            "attribution": "Trained in part on NSynth (Engel et al. 2017, Google, CC BY 4.0).",
        },
    }
