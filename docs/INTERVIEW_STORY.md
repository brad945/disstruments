# Interview Story — Disstruments (living document)

> Updated at the end of every phase. Numbers marked *(pending)* get filled in as
> the milestones land — never cite a pending number in an interview.

## The 30-second pitch

"I built Disstruments — a song-deconstruction app: you upload a track and it gives you
separated stems, fine-grained instrument identification, and key/BPM in one DAW-style
report. The interesting part is the ML engineering: public datasets only label
instruments coarsely — 'guitar', not 'nylon-string acoustic' — so I'm building a
synthetic data pipeline that renders labeled training audio from MIDI, fine-tuning a
music foundation model (MERT) against it, and measuring and closing the synthetic-to-
real domain gap. Everything ships through an eval harness with artist-disjoint splits
and calibrated confidences — the app never claims more certainty than the model has."

## The 2-minute deep dive

Structure it as three layers:

1. **The product problem.** Producers decompose songs by ear to learn genres; that
   takes years. Existing tools are fragmented: stem splitters are mature, instrument
   taggers are coarse, nothing integrates. The gap is fine-grained identification —
   "Rhodes, not keys; 808 sub, not bass" — presented honestly.

2. **The systems layer.** Local-first FastAPI + Next.js app; pipeline is a DAG of
   independent stages (separation → per-stem tagging; full-mix tagging; attributes)
   with graceful degradation — a failed stage flags its section, the report still
   assembles. Every stage records model, version, weights checksum, wall time — full
   provenance, so results are reproducible across model swaps. Scale-sensitive choices
   (SQLite vs Postgres, in-process queue vs Celery, local FS vs S3) all sit behind
   interfaces with the small implementation shipped — swap is config, not rewrite.
   Rate limiting is fully built but dormant, and its tests run with limits ON so the
   OFF-state flag never hides rot.

3. **The ML layer (the core).** Fine-grained instrument recognition is a label-scarcity
   problem: no large dataset says *which kind* of guitar. Plan: ~60-leaf hierarchical
   taxonomy (OpenMIC-compatible level 1); synthetic rendering pipeline — MIDI through
   randomized samplers/synths/effects/mastering chains — manufacturing perfectly
   labeled clips at scale; MERT backbone with an ablation ladder (linear probe → MLP →
   LoRA → full fine-tune); hierarchical head with consistency constraints so the model
   degrades to "guitar (subtype uncertain)" instead of guessing; temperature-scaled
   calibration so a 0.87 means 87%; artist-disjoint eval with a measured synthetic-to-
   real domain gap. Every model swap gates on the eval harness.

## Hardest problem so far + how I debugged it

**Symptom:** first real run — every analysis job failed instantly; tests that mocked
nothing but the models all passed. **Debug path:** job status said `failed` with no
report → worker thread's traceback showed `storage.path` on `None` → but storage *was*
initialized at app startup → diff'd the two call sites: `main.py` accessed
`storage_mod.storage` (late-bound attribute), the pipeline runner did
`from ..storage import storage` (value copied at import time, before init ran).
Python's `from x import y` binds values, not references. Fixed by late-binding;
the durable lesson is that module-global singletons are injection points pretending
they aren't — real DI would make this bug unrepresentable.

Runner-up: the entire Homebrew Python toolchain was broken at the C-extension level
(pyexpat vs system libexpat symbol mismatch) — diagnosing that the *interpreter*, not
the project, was broken, and fixing it with a self-contained uv-managed CPython
without touching system state.

## Concrete numbers to cite

- Draft-1 pipeline: 8 stages per job (transcode, separation, mix tagging, 4 per-stem
  taggings, attributes); 268 tests in fake-ML mode (after M1); rate-limit contract tested ON.
- Models in play: Demucs v4 `htdemucs` (84 MB weights, 4 stems), PANNs Cnn14 (327 MB,
  527 AudioSet classes), librosa key/BPM, pyloudnorm LUFS.
- Real run on M5 (133 s song, warm): separation 11.2 s on MPS (real-time factor
  ≈0.09 → ~20–25 s for a 4-min song, >10× inside the 5-min budget); tagging 8.3 s
  (PANNs, CPU); end-to-end ≈20 s. Peak memory footprint 7.8 GB (full-song SED on CPU
  is ~4 GB of it → chunking follow-up).
- Stock PANNs on an isolated vocal stem: *Singing* scores only 0.14 (95th pct), so the
  0.30 threshold hides it. Consistent with uncalibrated scores meeting fixed thresholds
  (confounded by a 22 kHz mono source; confirm on a real master). It motivates a
  fine-tuned head plus calibration together; temperature scaling alone can't fix one
  class.
- Taxonomy v3.0.0: 65 leaves, 86 nodes, 3 levels, 10 families. Loaders for OpenMIC
  (20k clips/20 classes), MedleyDB (196 public multitracks; all 330 metadata files load
  with 0 unmapped labels), Slakh2100 (redux dedup reproduces published 1289/270/151).
- Harness baselines, MedleyDB pinned test split (29 tracks, 20 artists): prior map_leaf
  0.142, random 0.218. Random "beats" prior (constant scores give AP = prevalence) but
  the gate refuses it on calibration + hierarchical F1. Only 15 leaves have ≥3 real test
  positives, which is why the gate uses an artist-level paired bootstrap.
- Eval-bug war story: a masking choice let a "has drum kit" score reach AP 0.94 on
  "non-kit percussion" (label-dependent missingness). Fixed in taxonomy 2.0.0; an audit
  now catches that class of bug.
- Budget envelope: ≲$1k total; M1 costs $0.

## "What would you do differently?"

- **Dependency injection over module globals.** The storage singleton produced the
  worst bug of phase 1. Constructor-injected dependencies (or FastAPI's `Depends`)
  would have made it structurally impossible.
- **Vendor model weights with checksums from day one** (F13 says so; I initially let
  packages auto-download — one of them via a silent `os.system('wget …')` that macOS
  can't run). Trusting research packages' download paths is a supply-chain and
  reliability mistake.
- **Pin the interpreter in-repo from the start** (a `.python-version`), not after the
  first environment failure.
- **Honest framing for interviews:** draft 1's tagger is a stock AudioSet model — the
  differentiated work (taxonomy, synthetic data, fine-tuning, calibration) is the
  current milestone, and I'd say exactly that rather than oversell.
