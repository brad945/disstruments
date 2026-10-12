"""Stage B2 (M5; PRD F29/F30/F33): fine-grained instrument tagging with our own model.

Serves the ONNX export of the from-scratch CNN (`ml/models/export.py`). The features and
windowing match evaluation exactly: 22.05 kHz mono log-mels (`scratch.logmel_array`) over
non-overlapping 10 s windows from t = 0, the same windows as the pooled MoisesDB eval.

`merge()` combines it with the incumbent PANNs tags under the sidecar's routing:
- presence of an instrument comes from whichever model won it in the M5 comparison;
- under a present instrument this model may name a subtype (nylon vs steel, ...): one
  lane per instrument, descending the taxonomy only while the top child clearly beats its
  runner-up (`descend()`). Leaf scores are not calibrated on real music yet, so they are
  flagged `calibrated: false` and the UI always marks them uncertain (F15). With no clear
  winner at the first level the entry is "<Family> (subtype uncertain)" (F33).

The model lives outside the repo (research-only licence): `settings.leaf_model_dir`
holds `model.onnx` + `model.json`; the sha256 in the sidecar is verified on load (F13).
Missing model -> the stage is skipped and the report is PANNs-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import settings

_model = None  # lazy singleton: (onnxruntime session, sidecar dict)


# fake-ML mode has no sidecar; same shape as the shipped routing so tests exercise merge()
FAKE_ROUTING = {"presence": ["bass"],
                "subtypes": {"guitar": ["guitar.acoustic.nylon", "guitar.electric.distorted"],
                             "bass": ["bass.upright", "bass.electric"],
                             "drums": ["drums.acoustic_kit", "drums.electronic"]}}


class LeafModelError(RuntimeError):
    pass


def sidecar_path() -> Path:
    return settings.leaf_model_dir / "model.json"


def available() -> bool:
    return settings.fake_ml or sidecar_path().exists()


def _load():
    global _model
    if _model is None:
        import onnxruntime as ort
        meta = json.loads(sidecar_path().read_text())
        onnx_path = settings.leaf_model_dir / meta["onnx"]
        got = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if got != meta["sha256"]:
            raise LeafModelError(f"leaf model checksum mismatch: {got[:12]} != {meta['sha256'][:12]}")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _model = (sess, meta)
    return _model


def info() -> dict:
    """Provenance for the report (F13): what model, which checksum, what it can't say."""
    if settings.fake_ml:
        return {"name": "fake-leaf", "version": "0", "sha256": "fake", "leaf_calibrated": False,
                "routing": FAKE_ROUTING}
    meta = _load()[1]
    return {k: meta[k] for k in ("name", "version", "sha256", "taxonomy_version", "temperature",
                                 "leaf_calibrated", "routing", "licence", "song_aggregate")}


def trained() -> set[str] | None:
    """Nodes the served model had training positives for (None in fake-ML mode = all)."""
    return None if settings.fake_ml else set(_load()[1]["trained"])


def windows(y: np.ndarray, sr: int, seconds: float, min_s: float = 1.0) -> list[np.ndarray]:
    """Non-overlapping windows from t = 0, the last one zero-padded (as the mel cache pads
    short clips); a tail shorter than `min_s` is dropped unless it is the whole song."""
    n = int(sr * seconds)
    out = []
    for a in range(0, max(1, len(y)), n):
        w = y[a:a + n]
        if len(w) < min_s * sr and out:
            break
        out.append(np.pad(w, (0, n - len(w))) if len(w) < n else w)
    return out


def _sigmoid_t(z: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z / T))


def window_scores(audio_path: Path) -> tuple[list[str], np.ndarray, float]:
    """-> (nodes, calibrated probs (n_windows, n_nodes), window hop seconds)."""
    if settings.fake_ml:
        return _fake_window_scores(audio_path)
    import librosa
    import soundfile as sf

    from ..ml.models.scratch import logmel_array
    sess, meta = _load()
    f = meta["features"]
    data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if file_sr != f["sr"]:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=f["sr"])
    ws = windows(y, f["sr"], f["window_s"])
    X = np.zeros((len(ws), f["n_mels"], f["frames"]), np.float32)
    for k, w in enumerate(ws):
        m = logmel_array(w)
        X[k, :, : m.shape[1]] = m
    z = np.concatenate([sess.run(None, {"logmel": X[i:i + 16]})[0] for i in range(0, len(X), 16)])
    return meta["nodes"], _finalize(_sigmoid_t(z, meta["temperature"]["T"]), meta["nodes"]), \
        float(f["window_hop_s"])


def _finalize(probs: np.ndarray, nodes: list[str]) -> np.ndarray:
    """Parent = max(self, descendants), exactly as the eval harness scores parents. Without
    this, served parent scores differ from every evaluated number (up to 0.5 on `bass`).
    Global T is monotone, so it commutes with the max."""
    from ..ml.taxonomy import get_taxonomy
    return get_taxonomy().max_propagate(probs, nodes)


def _fake_window_scores(audio_path: Path) -> tuple[list[str], np.ndarray, float]:
    """Deterministic, so tests exercise routing: bass -> electric, guitar -> acoustic ->
    nylon, drums too close to call (subtype uncertain)."""
    import soundfile as sf
    scores = {"bass": 0.9, "bass.electric": 0.8, "bass.upright": 0.3,
              "guitar": 0.7, "guitar.acoustic": 0.7, "guitar.electric": 0.2,
              "guitar.acoustic.nylon": 0.7, "guitar.acoustic.steel": 0.3,
              "guitar.electric.distorted": 0.2,
              "drums": 0.4, "drums.acoustic_kit": 0.35, "drums.electronic": 0.3}
    nodes = list(scores)
    n = max(1, int(np.ceil(sf.info(audio_path).duration / 10.0)))
    base = np.array(list(scores.values()))
    return nodes, np.tile(base, (n, 1)), 10.0


# Descend one taxonomy level only if the top child's song score beats the runner-up's by
# this much. Fixed before re-running any song (M5 debrief); not tuned.
DESCEND_MARGIN = 0.2


def _conf(p: np.ndarray) -> float:
    return float(np.percentile(p, 95))


def descend(root: str, p95: dict[str, float], children, allowed: set[str], show: float,
            margin: float) -> str:
    """Walk down from `root` while the best trained child beats its runner-up by `margin`.

    A relative rule, because leaf scores are uncalibrated: one global T is monotone, so it
    can't change how siblings rank, but an absolute cut-off on their scores means nothing
    (on a real acoustic song, "distorted" scored 0.53). Every trained child competes,
    including suppressed ones, but only `allowed` nodes may be named; a node with a single
    trained child can't discriminate, so it is never descended. Returns the deepest node
    reached (`root` = no confident subtype)."""
    node = root
    while True:
        kids = sorted(children(node), key=lambda n: -p95.get(n, 0.0))
        if len(kids) < 2:
            return node
        best, second = kids[0], kids[1]
        if best not in allowed or p95[best] < show or p95[best] - p95[second] < margin:
            return node
        node = best


def merge(panns: list[dict], nodes: list[str], probs: np.ndarray, hop_s: float, routing: dict,
          name_of, panns_node: dict[str, str], show: float, intervals,
          trained: set[str] | None = None) -> list[dict]:
    """PANNs entries (OpenMIC-20 labels) + our window probs -> report instrument entries.

    `name_of(node)` gives display names for our nodes; PANNs entries keep their label text
    (as in schema v1). `panns_node` maps OpenMIC labels to taxonomy nodes;
    `intervals(probs, hop, enter, exit)` is the shared hysteresis helper; `trained` are the
    nodes the model had training positives for (default: all). Every entry gets `node`,
    `model` and `calibrated`. Entries from this model are mix-level: the window scores come
    from the full mix, whatever stem the incumbent found the instrument on.
    """
    col = {n: k for k, n in enumerate(nodes)}
    trained = set(nodes) if trained is None else set(trained) & set(nodes)
    p95 = {n: _conf(probs[:, col[n]]) for n in trained}
    ours = "disstruments-cnn"
    margin = float(routing.get("descend_margin", DESCEND_MARGIN))
    out: list[dict] = []

    def children(n: str) -> list[str]:
        return [c for c in trained if c.rsplit(".", 1)[0] == n and c.count(".") == n.count(".") + 1]

    def ours_entry(node: str, cap: float | None = None, calibrated: bool = True) -> dict | None:
        if node not in col:
            return None
        p = probs[:, col[node]]
        c = _conf(p) if cap is None else min(_conf(p), cap)
        if c < show:
            return None
        return {"label": name_of(node), "node": node, "confidence": round(c, 3), "stem": "mix",
                "activations": intervals(p, hop_s, show, show * 0.7), "model": ours,
                "calibrated": calibrated}

    # presence owned by our model replaces the incumbent's entries for that label
    owned = set(routing.get("presence", []))
    present = [dict(t) for t in panns if t["label"] not in owned]
    for label in owned:
        e = ours_entry(label)
        if e:
            e["panns_label"] = label
            present.append(e)

    subtypes = routing.get("subtypes", {})
    for t in present:
        label = t.get("panns_label", t["label"])
        # PANNs scores were never fit on real data (draft-1 status quo), hence calibrated False
        base = {"node": panns_node.get(label), "model": "panns-sed", "calibrated": False, **t}
        base.pop("panns_label", None)
        root = panns_node.get(label, label)
        if not subtypes.get(label) or root not in trained:
            out.append(base)
            continue
        # allowed = the listed subtypes and their ancestors below the family root
        allowed = {a for n in subtypes[label] for a in _ancestors(n) if a.startswith(root + ".")}
        node = descend(root, p95, children, allowed, show, margin)
        e = ours_entry(node, cap=t["confidence"], calibrated=False) if node != root else None
        if e:
            e["subtype"] = True                      # UI: always uncertain until the golden set
            out.append(e)
        else:
            out.append({**base, "label": f"{base['label']} (subtype uncertain)", "subtype_uncertain": True})
    return sorted(out, key=lambda r: -r["confidence"])


def _ancestors(node: str) -> list[str]:
    parts = node.split(".")
    return [".".join(parts[:k]) for k in range(1, len(parts) + 1)]
