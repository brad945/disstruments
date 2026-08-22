# Disstruments — Product Requirements Document

**Version:** 0.1 (MVP scope)
**Owner:** Bradley
**Status:** Draft for review
**Last updated:** 2026-07-15

---

## 1. One-liner

Give Disstruments a song → it identifies and extracts every instrument used to make it, then turns that into (a) a learning tool for understanding songs and genres, and (b) the foundation of a production workflow. Long-term moonshot: also infer synths, presets, mixing decisions, and effects chains.

## 2. Problem statement

Experienced producers can listen to a track and mentally decompose it: "that's an 808, a plucked nylon guitar, a supersaw pad with sidechain." This skill takes years. Disstruments automates that decomposition so that:

1. A producer entering a new genre can immediately see *what instruments define it* and *how they're used in specific songs*.
2. The analysis output feeds directly into production — isolated stems, MIDI, and (later) recreated patches — instead of dying as a static report.

Nothing on the market stitches these together. Instrument taggers exist (coarse), stem splitters exist (mature), and preset reverse-engineering doesn't exist (research problem). The product gap is the **integration + pedagogy layer**.

## 3. Goals / Non-goals

### Goals (MVP, Phase 0 — local)

- G1: Upload an audio file, get a full per-song analysis: detected instruments (with confidence + time regions), separated stems, key, BPM, chords, song structure.
- G2: Analysis is browsable in a clean web UI: waveform + timeline, per-instrument activity lanes, playable stems.
- G3: A song library: every analyzed song is saved locally; genre tags accumulate into a "genre profile" view (instrument frequency across songs of a genre).
- G4: Every ML component sits behind an **evaluation harness** so model choices are data-driven and regressions are caught when swapping models.
- G5: Rate limiting, quota, and abuse-signal architecture is **fully designed and stubbed** in code (interfaces, middleware, config) but dormant in local mode. Going public must not require re-architecture.
- G6: Legal posture is clean from day one (see §9).

### Non-goals (MVP — explicitly out of scope; do NOT build without Bradley's sign-off)

- NG1: Synth/preset/effects-chain inference (Phase 2 moonshot; research track only).
- NG2: YouTube/Spotify/link ingestion (Phase 1+; upload only for now).
- NG3: Public deployment, signups, accounts, billing.
- NG4: A DAW or any audio *editing* capability. We analyze and export; we don't edit.
- NG5: Social features, sharing, comments, playlists.
- NG6: Mobile apps.
- NG7: Redistribution of analyzed audio to anyone other than the uploading user.

> **Anti-hallucination guardrail:** any feature not listed in §6 is out of scope. If during implementation something seems missing, it goes into §12 (Open questions) and gets asked — not silently built.

## 4. Users & personas

- **P1 — Bradley (primary, only user in Phase 0):** producer who wants to deconstruct songs and learn genres fast.
- **P2 — Genre explorer (Phase 1):** producer diving into an unfamiliar genre; wants "the 20 instruments that define UK garage, with song examples."
- **P3 — Learner/student (Phase 1+):** wants to understand how a specific song was constructed.

## 5. Product phases

| Phase | Name | Scope | Gate to next phase |
|---|---|---|---|
| 0 | Local MVP | Upload → analyze → browse → library → genre profiles. Runs on Bradley's machine. | Bradley likes it and it's accurate enough (eval targets §8 met) |
| 1 | Public beta | Auth, rate limiting activated, hosted inference, YouTube-link ingestion decision revisited, DMCA agent registered | Cost model sustainable, abuse controls proven |
| 2 | Production layer | MIDI export polish, patch-recreation assistant, preset/effects inference research track | Research results usable |

## 6. Feature requirements (Phase 0)

Priority: **M** = must-have for MVP, **S** = should-have, **C** = could-have (only if trivial).

### 6.1 Ingestion

- **F1 (M):** Upload audio file via web UI. Formats: mp3, wav, flac, m4a/aac, ogg. Max 200 MB / 15 min duration (config).
- **F2 (M):** Validate + transcode to canonical internal format (44.1 kHz stereo wav) via ffmpeg. Reject corrupt/unsupported files with clear errors.
- **F3 (M):** Deduplicate by audio content hash (re-upload of same song reuses prior analysis).
- **F4 (S):** User-supplied metadata on upload: title, artist, genre tag(s). Optional auto-suggest genre from classifier output.
- **F5 (C):** Batch upload (queue multiple files).

### 6.2 Analysis pipeline (the core)

- **F6 (M):** **Stem separation** into at minimum: vocals, drums, bass, other. Target: 6-stem (vocals, drums, bass, guitar, piano/keys, other) if model quality passes eval.
- **F7 (M):** **Instrument detection/tagging**: multi-label instrument identification over the full mix *and* per-stem, with confidence scores and per-time-window activations (so the UI can show *when* each instrument plays). Taxonomy: OpenMIC-2018's 20 classes as the floor (accordion, banjo, bass, cello, clarinet, cymbals, drums, flute, guitar, mallet percussion, mandolin, organ, piano, saxophone, synthesizer, trombone, trumpet, ukulele, violin, voice), extended with electronic-production classes (e.g., 808/sub bass, pad, pluck, lead, FX/riser) as a Phase 0 stretch **only if** eval data exists for them.
- **F8 (M):** **Global song attributes:** key, BPM (with confidence), time signature (S), duration, loudness (LUFS) — global and per-stem.
- **F9 (M):** **Song structure segmentation:** intro/verse/chorus/bridge/outro boundaries with timestamps.
- **F10 (S):** **Chord progression** extraction with timeline.
- **F11 (S):** **MIDI transcription** of pitched stems (bass line, lead melody, vocal melody) via note-transcription model. Exported as .mid.
- **F12 (M):** Pipeline is a DAG of independent stages; a stage failure degrades gracefully (partial report, failed stage flagged) rather than failing the whole job.
- **F13 (M):** Every model output stored with: model name, version, checksum, parameters, and confidence — so results are reproducible and re-runnable after model upgrades.

### 6.3 Results UI

- **F14 (M):** Song report page: waveform, playback, instrument lanes (which instrument active when), stem player (solo/mute per stem), global attributes card, structure timeline.
- **F15 (M):** Confidence displayed honestly (e.g., "synthesizer — 0.91", "mandolin — 0.34 (uncertain)"). Never present low-confidence detections as fact. UI threshold configurable; default: hide < 0.3, mark 0.3–0.6 as uncertain.
- **F16 (M):** Download: individual stems (wav), full analysis (JSON), MIDI (if F11).
- **F17 (S):** A/B listen: toggle between full mix and any stem subset.

### 6.4 Library & learning layer

- **F18 (M):** Library view: all analyzed songs, searchable/filterable by instrument, genre, key, BPM.
- **F19 (M):** **Genre profile view:** for a genre tag, aggregate across its songs: instrument frequency table ("piano appears in 14/16 analyzed house tracks"), BPM distribution, common keys, typical structure. This is the pedagogical core.
- **F20 (S):** Cross-song instrument query: "show me every song in my library using a saxophone, jump to where it plays."
- **F21 (C):** Song-to-song comparison view (two reports side by side).

### 6.5 Infrastructure features (built now, dormant where noted)

- **F22 (M):** Job queue with status (queued → processing [stage x/y] → done/failed), progress streaming to UI.
- **F23 (M, dormant):** Rate limiting & quota system per §7 — implemented as middleware + config, disabled by a single flag in local mode.
- **F24 (M):** Structured logging + per-job cost/duration accounting (CPU/GPU seconds per stage). This *is* the rate-signaling substrate for Phase 1.
- **F25 (M):** Eval harness per §8, runnable as a CLI (`disstruments eval <stage>`).
- **F26 (S, dormant):** Auth scaffolding: session model with a single hardcoded local user; interface supports real auth later without schema change.

## 7. Rate limiting, quotas, and abuse signaling (design-complete, dormant in Phase 0)

The requirement: **no gaps** — everything below is implemented and unit-tested in Phase 0, gated behind `RATE_LIMITING_ENABLED=false`.

### 7.1 Threat/cost model

Analysis jobs are expensive (GPU-bound stem separation dominates: expect ~10–60 s GPU per song). Abuse vectors when public: bulk-analysis scraping, storage exhaustion via uploads, queue flooding, credential sharing.

### 7.2 Controls

| Layer | Control | Default (Phase 1 values; Phase 0 = unlimited) |
|---|---|---|
| Request | Token bucket per user per endpoint class (sliding window in Redis) | 60 req/min general; 10 req/min upload; 30 req/min download (stems are bandwidth-heavy) |
| Job | Per-user concurrent job cap | 1 concurrent, 3 queued |
| Job | Per-user daily job quota | 10 analyses/day (free tier concept) |
| Upload | Size + duration caps | 200 MB / 15 min |
| Storage | Per-user storage quota; stems auto-expire | 5 GB; stems deleted after 30 days, JSON kept |
| Queue | Global queue depth limit + backpressure (429 + Retry-After) | 100 |
| Signal | Per-user cost ledger (GPU-seconds, storage-bytes) from F24 | always on, even Phase 0 |
| Signal | Anomaly flags: burst uploads, duplicate-hash spam, > p99 job cost | logged Phase 0; alert Phase 1 |

### 7.3 Rate signaling to clients

- All responses carry `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (values = ∞ sentinel in local mode).
- 429 responses include `Retry-After` and a machine-readable reason code (`quota_daily`, `queue_full`, `concurrency`).
- UI surfaces quota state (progress ring on "analyses remaining today") — hidden when unlimited.

## 8. Evaluation harness (requirement, not afterthought)

Every ML stage must be selected and monitored via evals. No model ships or gets swapped without a harness run.

### 8.1 Per-stage evals

| Stage | Dataset(s) | Metric(s) | Phase 0 target |
|---|---|---|---|
| Stem separation | MUSDB18-HQ test split | SDR (per stem) | ≥ published Demucs v4 baseline (~9 dB avg SDR on 4 stems); no silent regressions |
| Instrument tagging | OpenMIC-2018 test split; spot-check MedleyDB | mAP, per-class F1 | mAP ≥ 0.80 on OpenMIC-2018 (competitive with published PaSST/PANNs results — verify current SOTA before locking) |
| Key detection | GiantSteps Key | MIREX weighted score | ≥ 0.70 |
| BPM | GiantSteps Tempo, Ballroom | Acc1 / Acc2 (±4%) | Acc2 ≥ 0.95 |
| Chords | Isophonics (Beatles) subset | WCSR (chord symbol recall) | ≥ 0.75 majmin |
| Structure | Harmonix Set or SALAMI subset | boundary F1 @ 0.5s/3s | F1@3s ≥ 0.60 |
| MIDI transcription | Slakh / MAESTRO subsets (as applicable) | note F1 (onset+pitch) | informational only in Phase 0 |

*(Targets are starting anchors from published literature — the harness's first job is to verify them against current model results and adjust.)*

### 8.2 Golden set

In addition to public benchmarks: a **golden set of 20–30 songs Bradley knows intimately**, hand-labeled (instrument list per song, key, BPM, structure). Full-pipeline runs against the golden set are the acceptance test for "does this feel right," which public benchmarks can't capture. Stored in-repo (labels only + content hashes; audio stays local, never committed).

### 8.3 Harness mechanics

- CLI: `disstruments eval <stage> --model <name>` → scored report (JSON + markdown table), archived with model version.
- Regression rule: a candidate model replaces the incumbent only if it beats it on primary metric without >2% loss on any secondary metric.
- Latency and VRAM recorded per run (cost matters for Phase 1 hosting decisions).

## 9. Legal: copyright & DMCA analysis

**Disclaimer: this is research-informed analysis, not legal advice; consult a lawyer before Phase 1 (public launch).**

### 9.1 Why Phase 0 (local, personal) is low-risk

- You analyze audio you already possess, on your own machine, for private study. No distribution, no public performance, no hosting of copyrighted content. This is comparable to using Melodyne or a stem splitter locally — standard producer practice.

### 9.2 The intuition check ("anyone with a keen ear can do this")

Directionally right but legally incomplete. What's protected is the **recording and composition**, not the *facts about it*. Instrument lists, keys, BPMs, structure labels — factual metadata — are not copyrightable (Feist v. Rural: facts aren't protected). So the *analysis output* (JSON, genre profiles) is clean. The risk concentrates in three other places:

1. **Reproduction during processing.** Transient copies for analysis are generally defensible (fair-use factors favor transformative, non-substitutive analysis — the same theory that supports search-engine indexing, *Authors Guild v. Google*). Keep copies transient where possible; never make the original mix re-downloadable by anyone but the uploader.
2. **Stem redistribution.** A separated stem is a derivative chunk of the recording. Serving stems back **only to the user who uploaded the file** keeps you in personal-use territory. Serving stems of Song X to *other* users = distribution = the actual danger zone. **Hard product rule: stems are only ever accessible to the uploading user (NG7).**
3. **Ingestion source.** File upload of user-possessed audio is the clean path. YouTube ripping violates YouTube ToS (contract issue + weakens fair-use optics) — deferred deliberately.

### 9.3 Phase 1 checklist (before going public)

- Register a DMCA agent with the US Copyright Office (~$6, online) to qualify for §512 safe harbor as a hosting provider for user uploads.
- Terms of Service: users warrant they have the right to upload; repeat-infringer policy; takedown process.
- No public sharing of stems or audio, period (keeps you out of distribution).
- Genre profiles / aggregate stats are factual and fine to show publicly.
- Revisit YouTube ingestion with counsel; if ever added, prefer official APIs/licensed sources.
- Note the EU/UK text-and-data-mining exceptions and the evolving AI-training debate: we do **not** train models on user uploads in Phase 0/1 (if that ever changes, it needs explicit consent flow + fresh legal review).

## 10. Success metrics

- **Phase 0:** Bradley analyzes ≥ 25 songs; golden-set full-pipeline accuracy meets §8 targets; time-to-report < 5 min per song on local hardware; Bradley uses genre-profile view when starting a new track (the honest metric).
- **Phase 1 (later):** activation (first analysis completed), weekly returning producers, cost per analysis vs. willingness to pay.

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Instrument taxonomy too coarse for electronic music (everything is "synthesizer") | High | High — core value prop | Per-stem tagging + electronic-subclass stretch goal; golden set weighted toward Bradley's genres; embeddings (e.g., MERT/CLAP) + few-shot classes as research spike |
| Local GPU can't run best models | Medium | Medium | Eval harness records VRAM/latency; keep a CPU-fallback model tier per stage |
| Scope creep toward the preset moonshot | High | Medium | NG1 + this doc; moonshot lives in a separate research folder, never blocks MVP |
| Model licenses restrict future commercial use | Medium | High (Phase 1) | License column mandatory in design-doc model table; madmom (BY-NC-SA) and similar flagged before adoption |
| "Genre profile" misleads with tiny n | Medium | Low | Show n everywhere ("14/16 songs"); no profile below n=5 without a warning |

## 12. Open questions (ask Bradley — do not decide unilaterally)

1. Genre taxonomy: free-form tags, or a curated genre tree? (MVP default: free-form tags.)
2. Electronic-instrument subclasses (808, pad, pluck, lead): worth a labeling effort in Phase 0, or defer?
3. Should MIDI transcription (F11) be in the first build or fast-follow?
4. Target local hardware spec? (Determines model tier defaults — need GPU/VRAM details.)
5. Stems auto-expiry locally too, or keep forever in local mode? (Default: keep forever locally.)

## 13. Change control

This PRD is the source of truth for scope. Any feature not traceable to an F-number requires an explicit PRD amendment approved by Bradley before implementation.

---

## Amendment A1 — Fine-grained instrument recognition becomes the core (approved by Bradley, 2026-08-21)

**Decision:** the single most important feature is *specific* instrument identification —
leaf-level ("nylon-string acoustic guitar", "Rhodes", "808 sub"), not coarse ("guitar",
"keys"). The project is explicitly positioned as an ML engineering showcase on top of the
existing SWE/system design. Full technical plan: `ML_ENGINEERING.md`.

### New features

- **F27 (M):** Hierarchical instrument taxonomy (~60 leaf classes, 3 levels, versioned
  `taxonomy.yaml`); level 1 stays OpenMIC-compatible so draft-1 results remain valid.
- **F28 (M):** Synthetic training-data pipeline: MIDI → randomized sampler/synth/effects/
  mastering rendering → perfectly labeled clips (target 100k). Provenance logged per clip.
- **F29 (M):** Fine-tuned recognition model: pretrained music foundation backbone
  (MERT primary; CLAP/PaSST baselines) + hierarchical head; ablation ladder from linear
  probe to LoRA; artist-disjoint eval; promotion via existing §8.3 gates.
- **F30 (M):** Calibrated confidences (temperature scaling, ECE reported) so F15's UI
  thresholds correspond to real probabilities.
- **F31 (M):** Synthetic→real domain-gap measurement and mitigation (domain
  randomization, hierarchical co-training with coarse real labels).
- **F32 (S):** Zero-shot CLAP text-prompt tagging as baseline + "search my library by
  sound description" demo.
- **F33 (M):** Report UI upgrade: leaf-level labels with graceful fallback to parent
  ("guitar (subtype uncertain)") per F15 honesty rules.

### Reprioritization

- **Draft 2 = F25 (eval harness) + F27–F31 + F33.** The ML track.
- Structure (F9), chords (F10), MIDI (F11), genre profiles (F19) move to **draft 3**.
  Genre profiles get *better* from this trade: they'll aggregate leaf-level instruments.
- Risk #1 (taxonomy too coarse) is retired — it's now the roadmap, not a risk.
- Open question 2 is resolved (yes, fine-grained subclasses; funded with real effort).
- Budget accepted: ≲$1k total compute (mostly optional), per ML_ENGINEERING.md §8.
