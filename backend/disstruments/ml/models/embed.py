"""Frozen-backbone embedding cache (M2 S5).

Backbones (docs/M2_PLAN.md D1, run side by side):
- `mert95m`: m-a-p/MERT-v1-95M, 24 kHz, 13 hidden layers x 768. **CC-BY-NC-4.0 weights:
  research only** (licence-clean end state, docs/DIRECTION_NOTES.md).
- `clap`: laion/larger_clap_music (Apache-2.0), 48 kHz; hidden layers of the HTSAT audio
  encoder (pooled) + the projected 512-d audio embedding. Production candidate.

Each clip -> (n_layers, dim) float16: per-layer mean over time. The probe learns a softmax
layer weighting, so the cache keeps every layer. Cache file per (backbone, dataset, unit):
`<cache>/<backbone>/<dataset>_<unit>.npz` with `ids`, `emb`, `meta` (json).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

CLIP_SECONDS = 10.0


@dataclass
class Backbone:
    name: str
    hf_id: str
    sr: int
    license: str
    commercial_clean: bool
    embed: Callable[[Sequence[np.ndarray]], np.ndarray]   # list of mono clips -> (B, L, D)
    revision: str = ""


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(name: str, device: str | None = None) -> Backbone:
    """Loads from the local HF cache when the weights are there (`local_files_only`):
    transformers otherwise tries to fetch alternative weight files (e.g. a safetensors
    conversion) before loading, which stalls on a slow link."""
    from pathlib import Path as _P

    import torch

    hf_ids = {"mert95m": "m-a-p/MERT-v1-95M", "clap": "laion/larger_clap_music"}
    cached = name in hf_ids and any(
        (_P.home() / ".cache/huggingface/hub" / ("models--" + hf_ids[name].replace("/", "--"))
         / "snapshots").glob("*/config.json"))
    lfo = {"local_files_only": True} if cached else {}

    dev = device or _device()
    if name == "mert95m":
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        hf = "m-a-p/MERT-v1-95M"
        model = AutoModel.from_pretrained(hf, trust_remote_code=True, **lfo).to(dev).eval()
        fe = Wav2Vec2FeatureExtractor.from_pretrained(hf, trust_remote_code=True, **lfo)

        # Recent transformers no longer return hidden_states for MERT's remote code, so
        # capture them with hooks: encoder input (after pos-conv + layernorm, as HuBERT's
        # hidden_states[0]) plus each of the 12 transformer layers -> 13 x 768.
        captured: list = []
        enc = model.encoder
        enc.layers[0].register_forward_pre_hook(lambda mod, args: captured.append(args[0]))
        for layer in enc.layers:
            layer.register_forward_hook(lambda mod, args, out: captured.append(
                out[0] if isinstance(out, tuple) else out))

        @torch.no_grad()
        def embed(batch):
            captured.clear()
            x = fe([b for b in batch], sampling_rate=24000, return_tensors="pt", padding=True)
            model(**{k: v.to(dev) for k, v in x.items()})
            hs = torch.stack(captured, dim=1)                   # (B, 13, T, D)
            return hs.mean(dim=2).float().cpu().numpy()

        return Backbone("mert95m", hf, 24000, "CC-BY-NC-4.0", False, embed)
    if name == "clap":
        from transformers import ClapModel, ClapProcessor
        hf = "laion/larger_clap_music"
        model = ClapModel.from_pretrained(hf, use_safetensors=False, **lfo).to(dev).eval()
        proc = ClapProcessor.from_pretrained(hf, **lfo)

        @torch.no_grad()
        def embed(batch):
            x = proc(audio=[b for b in batch], sampling_rate=48000, return_tensors="pt")
            x = {k: v.to(dev) for k, v in x.items()}
            a = model.audio_model(**x, output_hidden_states=True)
            layers = [h.flatten(2).mean(dim=-1) if h.dim() == 4 else h.mean(dim=1)
                      for h in a.hidden_states]                 # each (B, C_l)
            proj = model.audio_projection(a.pooler_output)      # (B, 512)
            d = max(max(l.shape[-1] for l in layers), proj.shape[-1])
            pad = lambda t: torch.nn.functional.pad(t, (0, d - t.shape[-1]))  # noqa: E731
            return torch.stack([pad(l) for l in layers] + [pad(proj)], dim=1).float().cpu().numpy()

        return Backbone("clap", hf, 48000, "Apache-2.0", True, embed)
    raise ValueError(f"unknown backbone {name!r} (mert95m | clap)")


def load_clip(path: Path, sr: int, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Mono, resampled, first `seconds` (zero-padded)."""
    import librosa
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    n = int(sr * seconds)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def cache_path(cache_dir: Path, backbone: str, dataset: str, unit: str) -> Path:
    return cache_dir / backbone / f"{dataset}_{unit}.npz"


def build_cache(items: Iterable[tuple[str, Path]], bb: Backbone, out: Path, *,
                batch_size: int = 8, meta: dict | None = None,
                loader: Callable[[Path, int], np.ndarray] = load_clip) -> Path:
    """Embed (item_id, audio_path) pairs; resumable via a `.partial.npz` checkpoint."""
    items = list(items)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")
    done: dict[str, np.ndarray] = {}
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        done = dict(zip(z["ids"].tolist(), z["emb"]))
    todo = [(i, p) for i, p in items if i not in done]
    for k in range(0, len(todo), batch_size):
        chunk = todo[k:k + batch_size]
        embs = bb.embed([loader(p, bb.sr) for _, p in chunk])
        for (i, _), e in zip(chunk, embs):
            done[i] = e.astype(np.float16)
        if (k // batch_size) % 50 == 49:
            np.savez(partial, ids=np.array(list(done)), emb=np.stack(list(done.values())))
    ids = [i for i, _ in items]
    emb = np.stack([done[i] for i in ids])
    m = {"backbone": bb.name, "hf_id": bb.hf_id, "sr": bb.sr, "license": bb.license,
         "commercial_clean": bb.commercial_clean, "n": len(ids), "shape": list(emb.shape),
         "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), **(meta or {})}
    np.savez(out, ids=np.array(ids), emb=emb, meta=np.array(json.dumps(m)))
    partial.unlink(missing_ok=True)
    return out


def load_cache(path: Path) -> tuple[list[str], np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    return z["ids"].tolist(), z["emb"], json.loads(str(z["meta"]))
