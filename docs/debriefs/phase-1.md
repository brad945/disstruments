# Builder's Debrief — Phase 1: Making Draft 1 Actually Run

> Status: complete — steps 1–2 (setup, tests, fake-ML end-to-end), the port move, and
> step 3 (first real-model run, 2026-09-27; see the last section).

The theme of this phase: **code that has only been logic-tested meets a real
machine.** Every failure we hit lived at an integration seam — interpreter ↔ OS,
ORM ↔ database, module ↔ initialization order, package ↔ system tools. None were
in "the logic." That's the phase-1 lesson in one line: unit-tested logic tells you
almost nothing about whether the system runs.

---

## Decision 1 — SQLite + in-process worker queue (over Postgres + Redis + Celery)

**What we chose:** one SQLite file for all metadata; a single Python worker thread
consuming a `queue.Queue` inside the FastAPI process. **What we rejected (for now):**
the full SYSTEMS_DESIGN §2 stack — Postgres, Redis as broker, Celery workers.

**Why:** Phase 0 has exactly one user (me) and one machine. Postgres/Redis/Celery buy
you *concurrent multi-process scale* and *crash isolation* — neither is a Phase-0
problem — and cost you a docker-compose stack, connection management, and serialization
boundaries on day one. The design keeps the *interfaces* (Storage, queue admission,
`database_url` config) shaped like the big stack, so swapping is a config/adapter
change, not a rewrite. What would reverse this decision: a second concurrent user,
GPU work moving to a separate machine, or needing jobs to survive an API-process crash.

**The transferable concept:** *interface seams and swappable adapters.* You don't
avoid over-engineering by never thinking about scale — you avoid it by putting the
scale-sensitive decision behind an interface and shipping the small implementation.
Coupling, not component choice, is what makes migrations expensive.

**Likely interview question:** *"Your job queue is a thread in the web process — what
happens if the process dies mid-job?"*
**Strong answer:** "The job is lost from the in-memory queue, but stages are idempotent
and keyed by job_id, and job state is in SQLite — so a requeue re-runs cleanly. That's
an accepted Phase-0 risk: I'm the only user and I can click reanalyze. The moment jobs
must survive crashes, that's my trigger to move the queue behind the same admission
interface onto Celery with a visibility timeout, which the design already specs. I'd
rather carry a known, bounded risk than run three infrastructure services for one user."

---

## Decision 2 — Pin the interpreter (Python 3.12 via uv) instead of "whatever python3 is"

**What we chose:** a uv-managed, self-contained CPython 3.12, pinned in the Makefile.
**What we rejected:** the system default `python3` (3.14) and the Homebrew 3.11/3.12.

**Why:** two independent failures forced this. (1) The Homebrew Pythons were *broken at
the C level* — their `pyexpat` extension referenced a libexpat symbol
(`XML_SetAllocTrackerActivationThreshold`) newer than the macOS system library provides,
so even `python -m venv` died inside `ensurepip`. (2) Python 3.14 is too new: audio-ML
packages (torch/demucs/panns) publish wheels for new CPython versions ~6–12 months
late; you either compile from source or discover missing wheels at install time. A
uv-managed interpreter is a static, self-contained build — no dependence on Homebrew's
bottle-vs-OS matching, no system mutation to fix it.

**The transferable concept:** *your interpreter is a dependency and "latest" is a
choice, not a default.* ML stacks pin to the trailing-edge CPython (n-1 or n-2) because
the wheel ecosystem, not the language, is the constraint. Same logic as pinning CUDA
versions to match torch builds.

**Likely interview question:** *"Why not just fix Homebrew or use Docker?"*
**Strong answer:** "`brew reinstall` might pull the same mismatched bottle — the root
cause is Homebrew building against a different macOS revision than mine, which I can't
control. Docker solves reproducibility but costs me MPS: Apple-silicon GPU access
doesn't pass through containers, and this project's inference runs on MPS. A standalone
pinned interpreter gets hermetic-enough builds while keeping native GPU access —
that's the actual tradeoff, and MPS wins it."

---

## Decision 3 — Fake-ML mode as a first-class seam (over mocking in tests)

**What it is:** `DISS_FAKE_ML=1` swaps every model stage for a deterministic fake at
the *stage boundary*. The full app — API, queue, DB, SSE, storage, UI — runs identically;
only the model calls are fake. The entire test suite and all UI development run this way.

**Why this beats test-framework mocks:** mocks live in test files and drift from
production wiring; a fake *implementation behind the production interface* exercises the
real wiring every time. It's also a dev-speed feature (frontend work needs no 300 MB
checkpoint) and it proved itself this phase: fake-ML tests caught the cost-ledger and
storage-initialization bugs — pure wiring bugs — before any model ever loaded.

**The transferable concept:** *hexagonal architecture / test doubles at the port.* If
your interface seam is real, a fake adapter gives you a fast, deterministic system test.
If you can't fake a subsystem cleanly, that's a smell that the seam doesn't really exist.

**Likely interview question:** *"What do fake-ML tests NOT catch?"*
**Strong answer:** "Anything about the models themselves: wrong tensor shapes, device
placement, checkpoint loading, output distribution shifts — and integration issues
inside the ML packages, like panns shelling out to wget. That's exactly the boundary I
observed: every fake-mode test passed while the real stack had a broken import. Which
is why phase 1 ends with a timed real-model run, and why the eval harness — not unit
tests — is the quality gate for model behavior."

---

## War stories — the three integration bugs (debugging material)

### 1. `TypeError: NoneType += float` in the cost ledger
A freshly constructed `CostLedger(user_id=…, day=…)` had `None` in its counter columns.
**Root cause:** SQLAlchemy `default=0.0` is applied at **INSERT flush**, not at Python
object construction — ORM defaults are a database-write concept, not a constructor
concept. `row.gpu_s += x` on a not-yet-flushed object hit `None`. **Fix:** initialize
counters explicitly at construction. **Concept:** know *when* your ORM materializes
values (construction vs flush vs refresh); the same class of bug appears with
server-side defaults and autoincrement PKs read before flush.

### 2. Every job failed: `'NoneType' object has no attribute 'path'`
The pipeline runner saw `storage` as `None` even though the app initialized storage at
startup. **Root cause:** `from ..storage import storage` copies the *value at import
time* into the importing module's namespace. The value was `None` (set later by
`init_storage()`), and the runner kept its private `None` forever. `main.py` was
correct because it did `from . import storage as storage_mod` and read the attribute
late. **Fix:** late-bind the same way. **Concept:** Python's `from x import y` binds
values, not references — module-level mutable globals must be accessed through the
module object. The deeper lesson: this is the argument for explicit dependency
injection over module globals; a constructor-injected `Storage` could not have this bug.

### 3. `panns_inference` died on import: missing `~/panns_data/…csv`
**Root cause:** the package auto-downloads its label CSV and 300 MB checkpoint by
calling `os.system('wget …')` — and macOS has no wget. The `os.system` failure is
*silent* (just a nonzero return code nobody checks), then the subsequent `open()`
crashes. **Fix:** pre-fetch both files with curl into the package's expected paths — a
data fix, not a code fix. **Concept:** research-grade ML packages routinely assume
Linux-with-wget; auditing a dependency means auditing its *implicit system
dependencies* too. Long-term, weights should be vendored and checksummed at install
(the design doc already requires this, F13 — this bug is the justification).

---

## Decision 4 — Rate limiting: dormant but continuously tested

Phase 0 runs with `RATE_LIMITING_ENABLED=false`, but the test suite forces limits ON.
**Why:** dormant code that isn't executed rots silently; when Phase 1 flips the flag,
"implemented two months ago" must not mean "broken two months ago." The token-bucket
tests, admission checks, and 429 contract run on every `make test`. **Concept:** *dead
code isn't tested code* — a feature flag's OFF state is only safe if the ON state stays
under test. **Interview question:** *"Why build rate limiting no one uses?"* — "Because
retrofitting admission control means re-architecting request paths under pressure. The
expensive part is the design — where admission happens, what state it reads — and that's
cheap to build alongside the endpoints and nearly free to keep tested. Turning it on is
a config change, not a project."

## Decision 5 — Ports parameterized off 8000/3000 (minor, but a pattern)

Default dev ports collide with half the ecosystem. Moved to 8642/3642 — but the real
decision was making them **variables** (Makefile `API_PORT`/`WEB_PORT` → uvicorn bind,
Next rewrite target, CORS origin) instead of a new pair of hardcodes. Three places had
independently hardcoded port knowledge; now one variable feeds all three. **Concept:**
configuration should have a single source of truth precisely because its consumers
multiply silently.

---

## Step 3 — first real-model run (2026-09-27)

**Setup.** Full stack via the real API (`uvicorn` + `curl` upload, isolated
`DISS_DATA_DIR`), wrapped in macOS `/usr/bin/time -l` for peak memory. Audio: librosa's
Creative-Commons example tracks (no copyrighted audio, nothing committed) —
*Let's Go Fishin'* (Karissa Hobbs, 133.0 s, vocals/guitar/bass/drums) and *Vibe Ace*
(Kevin MacLeod, 61.5 s). Caveat: both are 22.05 kHz mono sources, so they understate
bandwidth and stereo cues a real rock/pop master would give the models.

### War story #4 — every tagging stage failed; the job still "succeeded"
First real run: separation OK, **all five tagging stages failed** with
`FileNotFoundError: ~/panns_data/Cnn14_DecisionLevelMax.pth`. Phase 1.1 had pre-fetched
`Cnn14_mAP=0.431.pth` — the checkpoint for panns' `AudioTagging` class — but the pipeline
uses `SoundEventDetection`, which loads a *different* checkpoint through the same silent
`os.system('wget …')`. The job finished as `partial` and the report assembled without
instruments: graceful degradation (F12) worked exactly as designed, and that is also why
nobody noticed. **Fix:** `backend/scripts/fetch_weights.sh` fetches all three PANNs files
with curl and **sha256-verifies** them; `make setup-ml` runs it. **Concepts:** (1) fix the
*class* of bug, not the instance — the first fix patched one file, the durable fix pins
every artifact by checksum (F13); (2) graceful degradation needs an alarm: a stage that
fails on *every* job is an outage, not degradation. Follow-up: surface stage-failure
rates, don't just record them.

### Numbers (warm run, M5, htdemucs on MPS, PANNs on CPU)

| Stage | *Fishin'* (133 s) | *Vibe Ace* (61.5 s) |
|---|---|---|
| transcode (ffmpeg) | 0.12 s | 0.06 s |
| separation (htdemucs, **MPS confirmed** via `params_json.device`) | 11.2 s | 5.9 s |
| tagging, mix + 4 stems (PANNs SED, **CPU** — hardcoded in `tagging.py`) | 8.3 s | 1.8 s |
| attributes (librosa key/BPM, pyloudnorm) | 0.56 s | 0.24 s |
| **total** | **≈20 s** | **≈8 s** |

- Cold run (first job; model loads): ≈23 s for *Fishin'* — model load costs ~3 s, not
  minutes, once weights are on disk.
- **Separation real-time factor ≈ 0.085–0.096** → a 4-minute song separates in ~20–25 s.
  That's >10× inside the ~5-min budget, so `htdemucs` stays default and there is headroom
  to evaluate `htdemucs_ft` (a 4-model bag, ~4× cost) as a quality tier.
- **Peak memory footprint 7.8 GB** (whole process, all three jobs, includes MPS). With
  tagging broken it was 3.5 GB, so PANNs SED over the *full song in one tensor* on CPU
  is the ~4 GB delta. It scales with duration (the upload cap is 15 min) → follow-up:
  chunk SED inference. Tagging time also scaled superlinearly (2.2× longer audio, ~4.6×
  tagging time), consistent with that.
- Weights on disk: htdemucs **84 MB** (safetensors, HF `adefossez/HTDemucs`); PANNs
  2 × 327 MB.

### Quality — the finding that justifies the ML roadmap
*Fishin'* report: guitar 0.65 (bass stem), bass 0.55, ukulele 0.43 ("?" band); key
A# major (0.99), 117.5 BPM. **No voice and no drums**, on a song with both. Diagnosis on
the isolated stems: PANNs' 95th-percentile *Singing* score on the **vocals stem** is
0.14 (max 0.20); on the drums stem *Drum* is 0.11 and even clip-level *Drum machine* is
only 0.26. The stems are fine — the scores are simply **uncalibrated**, and F15's fixed
0.30/0.60 thresholds treat them as probabilities. A hard threshold on uncalibrated
sigmoid outputs is not "confidence honesty"; it silently hides true detections.
This is the empirical case for F30 (per-level temperature scaling on a real held-out
split) — calibration isn't polish, it's what makes the F15 thresholds mean anything.
Not patched in draft 1 (lowering thresholds by eye would just be uncalibrated in a new
place); it becomes the first before/after ECE story in M3.

**Interview question:** *"Your pipeline reported success but produced no instruments —
how would you have caught that in production?"* **Answer:** "Per-stage failure-rate
metrics with an alert; a stage failing on 100% of jobs is an outage even when the
report degrades gracefully. And a canary: a fixed known song whose expected instrument
set is asserted after every deploy — the golden set (§3.3) doubles as that canary."
