# Disstruments — draft 1 MVP

Upload a song → get stems, instrument detection with time lanes, key/BPM/LUFS — in one
dark, DAW-style report. Companion docs: `PRD.md`, `SYSTEMS_DESIGN.md` (scope law lives there).

## Quick start (Mac, Apple Silicon)

```bash
brew install ffmpeg            # if you don't have it
make setup                     # python venv + npm install
make setup-ml                  # torch/demucs/panns/librosa (~several GB, one-time)
make api                       # backend :8000  (terminal 1)
make web                       # frontend :3000 (terminal 2)
```

Open http://localhost:3000, drop in an mp3/wav/flac you own, watch the stages stream,
click through to the report.

No models yet? `make api-fake` runs the entire app with deterministic fake ML —
useful for UI work and exactly what the test suite uses.

First real run downloads Demucs weights (~1 GB) and the PANNs checkpoint (~300 MB)
automatically. On an M-series Mac separation runs on MPS; expect ~1–3 min per song.

## What's in draft 1 (PRD F-numbers)

- Upload + validation + dedup (F1–F4), job queue with live SSE progress (F22)
- 4-stem Demucs separation with synced solo/mute stem mixer + downloads (F6, F16, F17)
- Instrument tagging with confidence + when-it-plays lanes, per-stem and full-mix (F7, F15)
- Key / BPM / LUFS (F8), waveform, provenance panel (F13)
- Rate limiting fully implemented but dormant (F23) — flip `DISS_RATE_LIMITING_ENABLED=1`
- Cost ledger always on (F24)

Deferred to draft 2: structure (F9), chords (F10), MIDI (F11), library filters UI,
genre profiles (F19), eval harness CLI (F25).

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
