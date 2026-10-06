# Disstruments — draft 1 MVP

Upload a song → get stems, instrument detection with time lanes, key/BPM/LUFS — in one
dark, DAW-style report. Companion docs: `PRD.md`, `SYSTEMS_DESIGN.md` (scope law lives there).

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

## What's in draft 1 (PRD F-numbers)

- Upload + validation + dedup (F1–F4), job queue with live SSE progress (F22)
- 4-stem Demucs separation with synced solo/mute stem mixer + downloads (F6, F16, F17)
- Instrument tagging with confidence + when-it-plays lanes, per-stem and full-mix (F7, F15)
- Key / BPM / LUFS (F8), waveform, provenance panel (F13)
- Rate limiting fully implemented but dormant (F23) — flip `DISS_RATE_LIMITING_ENABLED=1`
- Cost ledger always on (F24)

## Draft 2 so far (PRD Amendment A1, the ML track)

- Taxonomy v3.0.0, 65 leaf instruments (F27): `backend/disstruments/ml/taxonomy.yaml`
- MedleyDB / OpenMIC / Slakh loaders + eval harness (F25, instrument stage):
  `cd backend && .venv/bin/python -m disstruments.ml.eval --help`
  (subcommands: `taxonomy`, `baseline`, `run`, `compare`, `coverage`, `make-split`)
- Debrief: `docs/debriefs/m1.md`

Deferred to draft 3: structure (F9), chords (F10), MIDI (F11), library filters UI,
genre profiles (F19).

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
  ml/              taxonomy + mappings, dataset loaders, eval harness (draft 2)
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
