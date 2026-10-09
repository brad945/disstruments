# CLAUDE.md — Disstruments

Song deconstruction app: upload a song → stems + fine-grained instrument identification
+ key/BPM in one DAW-style report. Personal project (Bradley), doubling as an ML
engineering portfolio piece.

## Read these before building anything

- `PRD.md` — product scope. **Scope law: every feature must trace to an F-number.
  Anything else needs Bradley's explicit approval as a PRD amendment first. Never
  silently add features.** Amendment A1 (bottom of PRD) is the current direction.
- `SYSTEMS_DESIGN.md` — architecture, data model, API contracts, licenses.
- `ML_ENGINEERING.md` — the fine-grained instrument recognition plan (the core feature).

## Current state (as of first commits)

Draft 1 is built and runs: upload → transcode/dedup → in-process job queue with SSE →
Demucs 4-stem separation → PANNs coarse tagging → key/BPM/LUFS → report JSON → Next.js
dark DAW UI (waveform, instrument lanes, synced stem mixer). Tests pass in fake-ML mode.
First real run on this Mac done (2026-09-27, `docs/debriefs/phase-1.md` step 3): ~20 s
end-to-end for a 133 s song, Demucs on MPS. Known: PANNs runs over the full song in one tensor (7.8 GB
peak on CPU; now follows `DISS_DEVICE` → MPS, ~7× faster, identical scores), and its scores vs fixed F15 thresholds appear to hide true detections
(e.g. vocals; confounded by a mono test source — confirm on a real master). `make setup-ml` fetches + checksums PANNs weights.

## Next up (draft 2 = PRD Amendment A1)

ML track, in order (milestones in ML_ENGINEERING.md §8):
1. M1 DONE (branch `draft2-m1`, `docs/debriefs/m1.md`): taxonomy v3.0.0 (65 leaves; Bradley's decisions applied 2026-09-28) +
   MedleyDB/OpenMIC/Slakh loaders + `python -m disstruments.ml.eval` harness. Open
   taxonomy decisions for Bradley listed in the debrief.
2. M2 DONE except MedleyDB eval (2026-10-07; `docs/debriefs/m2.md`):
   - render pipeline (`ml/synth`, 10k clips at ~/datasets/disstruments-synth/v1);
   - MERT-95M vs CLAP probes (`ml/models`); OpenMIC trained-9 AP: MERT 0.774, CLAP 0.656,
     prior 0.538;
   - Bradley's listening audit applied.

   MERT is research-only (CC-BY-NC). CLAP is the licence-clean candidate; see
   [DIRECTION_NOTES]. Run GPU jobs one at a time (16 GB RAM).
3. M3 DONE (2026-10-09; `docs/debriefs/m3.md`):
   - a from-scratch 1.2M CNN ties frozen and LoRA MERT on real OpenMIC (coarse classes);
   - calibration = one global temperature fit on real held-out data;
   - 10x more synthetic data doesn't move real-music scores, so M4 = diversity.
4. M4–M5: domain-gap work (source and production diversity, separation-artifact
   augmentation, real co-training), then ONNX serving

Deferred to draft 3: structure (F9), chords (F10), MIDI export (F11), genre profiles (F19).

## Dev commands

- `make api-fake` + `make web` — run app with fake ML (fast, no models)
- `make api` + `make web` — real models (MPS on this M5 Mac)
- `make test` — pytest, fake-ML mode; rate-limit tests run with limits ON
- Env config: `DISS_*` vars, see `backend/disstruments/config.py`

## Hard rules (from PRD — do not violate)

- Stems are only ever served to the uploading user (legal posture, PRD §9 / NG7).
- Never train on user uploads. Never commit audio files (gitignored for a reason).
- No YouTube/link ingestion without Bradley's explicit go-ahead (PRD NG2).
- Rate limiting stays implemented-but-dormant; keep its tests running with limits on.
- Confidence honesty (F15): never present low-confidence detections as fact; UI shows
  "?" for 0.30–0.60, hides below 0.30. Calibration work must keep this meaningful.
- ML models: no new model/checkpoint without (a) an eval-harness run, (b) license
  recorded in SYSTEMS_DESIGN §4 (madmom is BY-NC-SA — do not ship; Essentia is AGPL).
- Interfaces stay swappable (SQLite→Postgres, in-process queue→Celery, local FS→S3);
  don't couple app logic to the local implementations.

## Bradley's working preferences

- Concise and direct; no fluff.
- Grill him with clarifying questions BEFORE building when scope is ambiguous —
  wave-style, one decision at a time, with a recommended answer each. He'd rather be
  asked than have you guess.
- Genres he cares about: rock / indie / pop first.
- Emphasize the ML engineering angle in design decisions — this project should read as
  technically serious (ablations, calibration, measured domain gaps, artist-disjoint
  evals), not just glue code.
