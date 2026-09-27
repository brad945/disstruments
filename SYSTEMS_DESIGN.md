# Disstruments — Systems Design Document

**Version:** 0.1 · Companion to PRD v0.1 · All F-numbers reference the PRD.

---

## 1. Design principles

1. **Local-first, public-ready.** Everything runs on one machine in Phase 0, but every seam where "public" changes things (auth, rate limiting, storage, GPU) is an interface with a local implementation — swap, don't rewrite.
2. **Pipeline as a DAG of replaceable stages.** Each ML stage is a versioned, evaluated plugin behind a stable contract (F12, F13).
3. **Honest outputs.** Confidence propagates from model → DB → UI (F15). No stage may emit unscored claims.
4. **Scope fence.** If it doesn't map to a PRD F-number, it doesn't get built.

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Next.js UI)                                           │
│  upload · job progress (SSE) · song report · library · genre    │
└────────────┬────────────────────────────────────────────────────┘
             │ HTTPS (localhost in Phase 0)
┌────────────▼────────────────────────────────────────────────────┐
│  Next.js server (BFF)                                           │
│  auth stub (F26) · rate-limit middleware (F23, dormant) ·       │
│  presigned upload handling · proxies API to FastAPI             │
└────────────┬────────────────────────────────────────────────────┘
             │ HTTP (internal)
┌────────────▼────────────────────────────────────────────────────┐
│  FastAPI (Python) — control plane                                │
│  jobs API · analysis CRUD · quota/cost ledger (F24) ·           │
│  enqueue to Redis                                                │
└──────┬──────────────────────────────┬────────────────────────────┘
       │                              │
┌──────▼───────┐              ┌───────▼────────────────────────────┐
│ Postgres     │              │ Redis                              │
│ metadata,    │              │ queue (Celery broker) · rate-limit │
│ analyses,    │              │ buckets · job progress pub/sub     │
│ eval runs    │              └───────┬────────────────────────────┘
└──────────────┘                      │
                              ┌───────▼────────────────────────────┐
                              │ Celery worker(s) — data plane      │
                              │ runs pipeline DAG stages           │
                              │ (GPU-bound; 1 worker in Phase 0)   │
                              └───────┬────────────────────────────┘
                                      │
                              ┌───────▼────────────────────────────┐
                              │ Object storage (local FS in P0,    │
                              │ S3-compatible interface):          │
                              │ originals · canonical wav · stems  │
                              │ · MIDI · analysis JSON             │
                              └────────────────────────────────────┘
```

**Why this shape:** Next.js owns UX; FastAPI owns orchestration (Python = native home of every audio ML model); Celery+Redis decouples slow GPU work from the request path (F22); Postgres because genre-profile aggregations (F19) are relational queries; storage behind an S3-style interface so Phase 1 is a config change.

Phase 0 packaging: single `docker-compose up` (next, api, worker, redis, postgres). GPU passthrough to the worker container; CPU fallback config for every stage.

## 3. Analysis pipeline (DAG)

```
upload → validate/transcode (ffmpeg, F2) → content-hash dedup (F3)
   │
   ├─► A. stem separation (F6) ────► per-stem: A1. instrument tagging (F7)
   │                            ├──► A2. MIDI transcription [S] (F11)
   │                            └──► A3. per-stem loudness
   ├─► B. full-mix instrument tagging (F7)
   ├─► C. key + BPM + time sig + LUFS (F8)
   ├─► D. structure segmentation (F9)
   └─► E. chords [S] (F10)
   │
   └─► F. report assembly: merge A–E → analysis JSON (versioned schema)
```

- Stages A–E run as independent Celery tasks with a fan-in assembly task (F). A stage failure marks its section `"status": "failed"` in the report; assembly still completes (F12).
- Instrument timeline (F14 lanes) = union of B (what) and A1 (where/which stem), windowed at 1 s hops.
- Every stage writes a `stage_run` row: model id, version, weights checksum, params, wall time, GPU seconds, peak VRAM (feeds F13 + F24 + eval archive).

## 4. Model selection (initial candidates — eval harness makes the final call)

| Stage | Primary candidate | Fallback / CPU tier | License notes (verify before adoption) |
|---|---|---|---|
| Stem separation | Demucs v4 (`htdemucs_ft`, 4-stem; `htdemucs_6s` for 6-stem) | `mdx_extra_q` (lighter) | Code MIT; check weight terms |
| (stretch) higher-SDR separation | BS/Mel-RoFormer community checkpoints (ZFTurbo MSST repo) | — | **Per-checkpoint licenses vary — audit each** |
| Instrument tagging | PaSST or PANNs (AudioSet/OpenMIC fine-tune) | YAMNet | Apache/MIT typically; verify checkpoint |
| Embedding research spike (electronic subclasses, PRD risk #1) | MERT or CLAP embeddings + lightweight head | — | Check MERT weights license |
| Key/BPM/LUFS | Essentia (+ its TensorFlow models) | librosa (BPM), pyloudnorm (LUFS) | Essentia AGPL-3.0 — **fine locally; revisit if code stays server-side (AGPL triggers on network use of modified versions); prefer librosa/pyloudnorm path if AGPL is unacceptable** |
| Chords | chord-extractor / autochord; madmom as reference | — | **madmom BY-NC-SA — non-commercial; do NOT ship in Phase 1 without replacement** |
| Structure | All-In-One (Kim & Nam) beats+structure model | MSAF (classical) | Check repo license |
| MIDI transcription | Spotify `basic-pitch` | — | Apache-2.0 ✅ |

Rule: no model enters `main` without (a) an eval-harness run archived, (b) license recorded in this table.

### 4.1 Datasets (M1 loaders — licenses as read 2026-09-27)

| Dataset | License | Consequence |
|---|---|---|
| MedleyDB 1+2 (audio) | CC BY-NC-SA 4.0, research/non-commercial; terms ask not to republish "in full or in part" ([downloads page](https://medleydb.weebly.com/downloads.html)) | **Models trained on it are non-commercial.** Fine for eval + portfolio; a commercial Phase must retrain without it or get permission. Never commit its audio. |
| MedleyDB metadata (`marl/medleydb` repo) | MIT | Metadata YAMLs used as test fixtures are OK. |
| OpenMIC-2018 | CC BY 4.0 (Zenodo 1432913); **individual FMA clips carry their own CC licenses incl. NC/ND** (`license_title` in metadata.csv) | Per-clip license must be filtered before any commercial training. |
| Slakh2100 | CC BY 4.0 (Zenodo 4599666); `slakh-utils` MIT | OK. |
| Lakh MIDI, `slakh-generation` patch list, mirdata fixtures | **unverified** | Verify before M2 (render pipeline consumes Lakh MIDI). |

## 5. Data model (Postgres)

```
users            id · email · created_at            (Phase 0: 1 seeded row, F26)
songs            id · user_id · content_hash(uniq) · title · artist ·
                 duration_s · fmt · uploaded_at
genre_tags       song_id · tag (free-form, PRD Q1)
jobs             id · song_id · status(queued|running|done|failed|partial) ·
                 created_at · finished_at
stage_runs       id · job_id · stage · model_name · model_version ·
                 weights_sha · params_json · status · wall_ms ·
                 gpu_s · peak_vram_mb · error
analyses         id · song_id · job_id · schema_version · report_json(jsonb)
instruments      analysis_id · label · confidence · stem · activations_ref
                 (flattened from report for F18/F19/F20 SQL queries)
stems            analysis_id · name · object_key · bytes · expires_at(null=never)
cost_ledger      user_id · day · gpu_s · storage_bytes · jobs_count   (F24)
eval_runs        id · stage · model_name · version · dataset · metrics_json ·
                 latency_ms · vram_mb · created_at · git_sha
```

`report_json` (schema_version'd) is the single source for the report page; `instruments` is a derived index for library/genre queries — rebuildable from JSON.

## 6. API contracts (FastAPI, `/api/v1`)

```
POST   /songs                multipart upload → {song_id, job_id}   (F1–F4)
GET    /jobs/{id}            {status, stages: [{name, status, progress}]}
GET    /jobs/{id}/events     SSE progress stream                    (F22)
GET    /songs                list + filters: instrument, genre, key,
                             bpm_range, q                           (F18, F20)
GET    /songs/{id}/analysis  report JSON                            (F14)
GET    /songs/{id}/stems/{name}  audio (auth: owner only — PRD NG7) (F16)
GET    /songs/{id}/midi/{stem}   .mid                               (F11)
GET    /genres/{tag}/profile aggregates: instrument freq (+n), bpm
                             histogram, key dist, structure stats   (F19)
POST   /songs/{id}/reanalyze re-run with current models             (F13)
```

Every response passes through rate-limit middleware (F23): `X-RateLimit-Limit/-Remaining/-Reset` headers always present; enforcement gated by `RATE_LIMITING_ENABLED`. 429 responses carry a `Retry-After` header plus body `{code: "quota_daily"|"queue_full"|"concurrency", retry_after_s}` per PRD §7.3.

## 7. Rate limiting implementation (dormant but real, F23)

- **Where:** middleware in the Next.js BFF (edge-cheap checks: token bucket) + FastAPI (authoritative: quotas, concurrency, queue depth). Both read the same Redis keys — no drift.
- **Token bucket:** Redis Lua script, sliding window, key `rl:{user}:{endpoint_class}`. Endpoint classes: `general`, `upload`, `download`.
- **Job admission:** on enqueue, transactional check of `concurrent ≤ cap`, `queued ≤ cap`, `daily quota` (from `cost_ledger`), `global queue depth`. Reject → 429 + reason code.
- **Config:** all limits in one `limits.yaml`; local profile sets everything to `unlimited` sentinel. Unit tests run with limits **on** so the dormant path is continuously exercised (PRD "no gaps" requirement).
- **Signals always-on even in Phase 0:** cost ledger writes, anomaly log lines (`burst_upload`, `dup_hash_spam`, `p99_cost_exceeded`) — so Phase 1 tuning starts with real data.

## 8. Evaluation harness (F25, PRD §8)

```
evals/
  datasets/     manifests + download scripts (MUSDB18-HQ, OpenMIC-2018,
                MedleyDB, GiantSteps Key/Tempo, Ballroom, Isophonics,
                Harmonix/SALAMI, Slakh/MAESTRO for MIDI transcription)
  golden/       golden-set labels (yaml) + content hashes — audio never committed
  runners/      one runner per stage, shared scoring lib (mir_eval, museval)
  reports/      archived runs (json + md), keyed by model+version+git_sha
```

- `disstruments eval <stage> [--model M] [--golden]` → scores vs. PRD §8.1 targets, writes `eval_runs` row + markdown report.
- `disstruments eval all --golden` = full-pipeline acceptance test (PRD §8.2).
- CI hook (even local pre-merge): swapping a stage's default model without an attached eval report fails the check.
- Promotion rule enforced in tooling: candidate must beat incumbent on primary metric, ≤ 2% regression on secondaries, and record latency/VRAM.

## 9. Legal enforcement points in the system (PRD §9)

| PRD rule | Where enforced |
|---|---|
| Stems only to uploader (NG7) | `GET /stems` checks `songs.user_id == session.user` — exists even in single-user Phase 0 so it can't be forgotten |
| Transient processing posture | Canonical wav deleted after job completes (config `keep_canonical=false`); original kept only for uploader re-download |
| No training on uploads | No training code paths in repo; golden set = labels + hashes only |
| Analysis facts are shareable, audio is not | Report JSON export contains zero audio data; genre profiles aggregate facts only |
| Upload-only ingestion (PRD §9.2.3, NG2) | API surface has no link/URL ingestion endpoint; adding one requires PRD amendment + legal review |
| Phase 1 gate | Deploy checklist includes: DMCA agent registered, ToS live, repeat-infringer flow, madmom/AGPL deps replaced or cleared |

## 10. Report JSON (abridged shape, `schema_version: 1`)

```json
{
  "schema_version": 1,
  "song": {"id": "…", "duration_s": 213.4},
  "global": {"key": {"value": "F# minor", "confidence": 0.83},
             "bpm": {"value": 128.0, "confidence": 0.97},
             "lufs": -8.9, "time_signature": "4/4"},
  "instruments": [
    {"label": "synthesizer", "confidence": 0.91, "stem": "other",
     "activations": [[0.0, 14.2], [28.5, 213.4]]},
    {"label": "drums", "confidence": 0.98, "stem": "drums",
     "activations": [[14.2, 199.0]]}
  ],
  "stems": [{"name": "vocals", "object_key": "…"}, …],
  "structure": [{"label": "intro", "start": 0.0, "end": 14.2}, …],
  "chords":    {"status": "ok", "timeline": […]},
  "midi":      {"status": "skipped"},
  "provenance": {"stages": [{"stage": "separation", "model": "htdemucs_ft",
                              "version": "4.0.1", "weights_sha": "…"}]}
}
```

## 11. Failure modes & handling

| Failure | Behavior |
|---|---|
| ffmpeg can't decode | Job fails at validate with user-readable error (F2); nothing stored |
| One DAG stage crashes/OOMs | Stage retried once (CPU-tier fallback if OOM); then marked failed; report assembles partial (F12) |
| Worker dies mid-job | Celery visibility timeout re-queues; stages idempotent (keyed by job_id+stage) |
| Duplicate upload | Hash hit → return existing analysis instantly (F3) |
| Disk full | Upload rejected pre-transcode; ledger warning |
| Model download unavailable offline | Weights vendored locally at install; checksummed (F13) |

## 12. Local → public migration map (what changes at Phase 1)

| Concern | Phase 0 | Phase 1 flip |
|---|---|---|
| Auth | seeded single user | real auth provider; schema unchanged (F26) |
| Rate limits | `unlimited` sentinel | `limits.yaml` prod profile; flag on |
| Storage | local FS via S3 interface | S3/R2; stems get `expires_at` |
| GPU | local card | queue → GPU autoscaler (Modal/RunPod class); worker code unchanged |
| Postgres/Redis | docker-compose | managed instances |
| Legal | personal use | DMCA agent, ToS, license audit (madmom/AGPL) |

## 13. Build order (Phase 0)

1. Skeleton: compose stack, upload → transcode → hash → job queue → SSE progress (F1–F3, F22).
2. Stage A separation + stem player UI (F6, F14 partial, F16).
3. Stages B/C (tagging, key/BPM) + full report page (F7, F8, F14, F15).
4. Eval harness for the three live stages + golden set labeling (F25, §8).
5. Stage D structure + library/filters (F9, F18).
6. Genre profiles (F19).
7. Dormant rate limiting + ledger wired end-to-end with tests (F23, F24).
8. Should-haves as time allows: chords, MIDI, A/B listen (F10, F11, F17).

Each step ends with something usable — no big-bang integration.
