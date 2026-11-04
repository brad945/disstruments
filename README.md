# Disstruments (in progress)

A passion project and audio-research project of mine. Give it a song and it gives back
stems and an instrument breakdown, with key, BPM and loudness, in a dark DAW-style report.
The focus is **fine-grained instrument identification**: not just "keys" or "guitar", but
Rhodes vs Wurlitzer, nylon vs steel-string, clean vs distorted, 808 vs acoustic kit.

The instrument model is my own: a 1.2M-parameter CNN on log-mel spectrograms, trained from
scratch on synthetic renders only. On real music (OpenMIC) it ties MERT-95M, a music
foundation model about 80x its size, when both are trained on the same synthetic data
(0.785 vs 0.774 mean AP; 95% CI of the difference [-0.009, +0.027]).

Later I want to turn this into **Instrumake**: software that turns that breakdown into a
starting point in a DAW (matching instruments, key, tempo), so you can build on any song at
any point and improvise on it. That comes from my own background in music production and
composition.

## How it works

1. **Transcode** the upload (ffmpeg) and catch duplicates.
2. **Split stems** with Demucs (vocals, drums, bass, other). Off-the-shelf, not my work.
3. **Which instruments are there?** PANNs on the mix and on each stem, with per-class Platt
   calibration I fit on real data. A stem only vouches for its own instrument.
4. **Which kind?** My CNN, on the full mix, names one subtype per instrument, and only
   when the top subtype clearly beats the runner-up. Otherwise it says "subtype uncertain".
5. **Key / BPM / LUFS** with classic signal processing (librosa, Krumhansl-Schmuckler,
   pyloudnorm).

Confidence is shown honestly: below 0.30 is hidden, 0.30-0.60 shows "?".

## What the research found so far

- **Volume doesn't help; diversity does.** 10x more synthetic data left real-music scores
  flat (0.784 -> 0.785) while synthetic validation kept rising. Adding real one-note
  recordings (NSynth, Iowa) nearly doubled AP on unseen nylon/steel guitar.
- **A loudness shortcut.** Loudness-normalised renders taught the model "louder = present".
  I measured it on real multitracks (MoisesDB) and fixed it with +/-12 dB gain augmentation:
  leaf mAP 0.41 -> 0.49.
- **What licence-clean data costs.** Training only on owned or commercially licensed
  material costs about 0.02 mAP on real music (0.490 -> 0.471).
- **Calibration has to be fit on real data.** Temperature fit on synthetic data is useless
  on real music. Fit on real held-out clips, it partly transfers to full songs (voice ECE
  0.56 -> 0.11), and some apparent calibration failures turned out to be ranking failures.
- **Evaluation discipline.** Artist-disjoint splits, paired bootstrap CIs, hand-labelled
  "golden" sets of full songs (labels frozen before scoring), and adoption rules written
  before each run. One of them rejected my own v8 model.

Details: `docs/debriefs/` (m1-m5, golden, v8), `ML_ENGINEERING.md`,
`docs/research/literature-check.md`.

## Quick start (Mac, Apple Silicon)

```bash
brew install ffmpeg            # if you don't have it
make setup                     # python venv + npm install
make setup-ml                  # torch/demucs/panns/librosa (~several GB, one-time)
make api                       # backend :8642  (terminal 1)
make web                       # frontend :3642 (terminal 2)
```

Ports live in the Makefile (`API_PORT`/`WEB_PORT`, default 8642/3642).

Open http://localhost:3642, drop in an mp3/wav/flac you own, watch the stages stream,
click through to the report.

No models yet? `make api-fake` runs the entire app with deterministic fake ML —
useful for UI work and exactly what the test suite uses.

`make setup-ml` also fetches and sha256-verifies the PANNs checkpoints (2 × 327 MB) via
`backend/scripts/fetch_weights.sh`; Demucs `htdemucs` weights (84 MB) download on first
run. Measured on an M5 (2026-09-27): a 133 s song takes ~20 s end-to-end, of which
separation is ~11 s on MPS (`docs/debriefs/phase-1.md`).

## Features (PRD F-numbers)

- Upload + validation + dedup (F1-F4), job queue with live SSE progress (F22)
- 4-stem Demucs separation with synced solo/mute stem mixer + downloads (F6, F16, F17)
- Instrument tagging with calibrated confidence + when-it-plays lanes (F7, F15)
- Fine-grained subtypes from my own model, served via ONNX (F29, F30, F33)
- Key / BPM / LUFS (F8), waveform, provenance panel (F13)
- Rate limiting fully implemented but dormant (F23): flip `DISS_RATE_LIMITING_ENABLED=1`
- Cost ledger always on (F24)

My instrument model isn't in the repo (it was trained on research-only data). With local
renders, `make leaf-model` exports it to `~/.disstruments/models/leaf`; without it the app
runs PANNs-only. Eval harness: `cd backend && .venv/bin/python -m disstruments.ml.eval --help`.

Deferred: structure (F9), chords (F10), MIDI export (F11), genre profiles (F19).

## Layout

```
backend/disstruments/
  config.py        all limits/config (DISS_* env vars)
  db.py            SQLAlchemy models (SQLite now, Postgres-shaped)
  storage.py       Storage interface (local FS now, S3 later)
  jobqueue.py      queue interface + admission control (Celery later)
  ratelimit.py     token buckets + headers (dormant locally)
  main.py          FastAPI routes (/api/v1)
  pipeline/        transcode → separation → tagging → attributes → runner
  pipeline/leaf_tagging.py   my instrument model (ONNX) + subtype rule
  ml/              taxonomy, dataset loaders, synth renders, models, eval harness, golden sets
frontend/app/      Next.js: upload+library page, /songs/[id] report
```

## Tests

```bash
make test
```

Runs fake-ML end-to-end (upload→report→stems→dedup), rate-limit contract with limits
**on**, and admission checks — per the PRD rule that dormant code is still tested code.

## Legal defaults baked in

Stems are only served to the uploader; canonical processing copies are deleted after
each job; nothing is ever trained on uploads. See PRD §9 before any public deployment.
