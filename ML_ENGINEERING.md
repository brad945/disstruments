# Disstruments — ML Engineering Plan (Fine-Grained Instrument Recognition)

**Companion to PRD Amendment A1.** This is the technical core of the project: a
fine-grained instrument recognition system built with modern ML engineering practice —
data-centric pipeline design, transfer learning with ablations, synthetic-to-real domain
adaptation, probability calibration, and benchmark-gated model promotion.

---

## 1. Problem statement (the ML framing)

Multi-label, hierarchical audio classification under distribution shift.

Given a polyphonic music recording (full mix and/or separated stems), predict the set of
instruments present at fine granularity — not "guitar" but *nylon-string acoustic /
steel-string acoustic / clean electric / distorted electric* — with per-time-window
activations and **calibrated** confidences.

What makes it non-trivial:

- **Severe label scarcity at fine granularity.** Public datasets top out at coarse labels
  (OpenMIC: 20 classes) or small size (MedleyDB: ~120 multitracks, ~80 instrument labels).
  There is no large "which kind of guitar" dataset. The central engineering contribution
  is *manufacturing* the training distribution (§3).
- **Distribution shift.** Synthetic training audio ≠ real mixed records (different rooms,
  players, mastering chains). Closing and *measuring* this gap is a first-class objective,
  not an afterthought (§5).
- **Hierarchical output space.** Predictions must be taxonomy-consistent (a clip can't be
  "distorted electric guitar: 0.9" and "guitar: 0.1"). §4.3.
- **Compositional inference.** Classification runs on source-separated stems, so upstream
  separation artifacts are part of the input distribution — train-time augmentation must
  simulate them (§3.4).

## 2. Taxonomy (the label space)

Three-level tree, ~60 leaf classes for v1. Level 1 = OpenMIC-compatible (backward
compatible with draft 1 and public benchmarks). Examples:

```
guitar ── acoustic ── nylon | steel
       └─ electric ── clean | crunch/distorted | muted-funk
keys ──── acoustic-piano ── grand | upright
       ├─ electric-piano (rhodes/wurli) | organ (drawbar/church) | clavinet
       └─ synth ── pad | lead | pluck | arp | brass-synth
bass ──── electric-fingered | electric-picked | slap | synth-bass | 808-sub | upright
drums ─── acoustic-kit | electronic-kit | drum-machine-808/909 | percussion-latin
strings ─ solo-violin | solo-cello | section
voice ──── lead | backing/harmony | choir
brass/winds ─ trumpet | trombone | sax | flute | clarinet
```

Design rules: every leaf must be (a) *audibly decidable* by an expert from a stem in
isolation, and (b) *coverable* by at least one data source in §3. Classes failing either
rule stay at the parent level. The taxonomy is versioned (`taxonomy.yaml`, semver);
reports store the version used.

## 3. Data engineering (the centerpiece)

Four sources, combined into one training distribution with per-source provenance:

### 3.1 Public real-world datasets (free labels, small)
| Dataset | What it gives | Role |
|---|---|---|
| MedleyDB 1+2 | ~196 multitracks, per-stem fine instrument labels | fine-grained REAL train + primary domain-gap eval |
| OpenMIC-2018 | 20k clips, 20 coarse classes | level-1 pretrain/regularizer + public benchmark |
| Slakh2100 | 2100 synth-rendered multitracks, 34 classes | mid-grain volume; validates the render approach |
| MoisesDB | 240 multitracks, stem taxonomy | separation-conditioned eval |
| IRMAS | 6705 clips, 11 classes, predominant-instrument | auxiliary eval only |

### 3.2 Synthetic rendering pipeline (unlimited perfect labels)
The Slakh recipe, generalized and owned end-to-end:

```
Lakh MIDI corpus (170k MIDI files)
  → per-track instrument assignment from OUR taxonomy      (label is free & exact)
  → render through sampler/synth pool (sfizz/FluidSynth + commercial libs + synth VSTs
     driven headless via pedalboard/DawDreamer)
  → per-stem augmentation: velocity/articulation jitter, room IRs (convolution reverb),
     amp/cab sims for electric guitar leaves, EQ/compression randomization
  → programmatic mixing: random level balance, panning, bus compression, limiter
     (mastering-chain randomization = the key domain-randomization lever)
  → emit (mix.wav, stems/*.wav, labels.json) triples
```

Target: **~100k labeled 10-second clips across 60 leaves** for a few hundred dollars of
CPU render time (rendering is CPU-bound, embarrassingly parallel). Every hyperparameter
of the renderer is logged per clip → dataset bugs are reproducible and fixable, and
class balance is a *config knob* rather than a crawl problem.

### 3.3 Golden set (human labels, eval-only)
25–30 real songs I know intimately, hand-labeled at leaf level with time regions.
Never trained on. This is the "does it work on actual records" acceptance test.

### 3.4 Separation-artifact augmentation
Because inference runs on Demucs outputs, a fraction of training clips are passed
through Demucs round-trip (mix → separate → take the stem) so the model sees bleed,
phasiness, and spectral holes at train time. Measured as an ablation (§6).

## 4. Modeling

### 4.1 Backbone: pretrained music foundation model
- **Primary: MERT-330M** (music-specific SSL transformer; layer-weighted embeddings).
- **Comparisons: CLAP** (enables zero-shot text-prompt baselines — "nylon guitar" as a
  prompt with no training — which is both a baseline and a demo) and **PaSST** (AudioSet
  supervised transformer; strong tagging prior).

### 4.2 Fine-tuning strategy — run as an explicit ablation ladder
1. Frozen backbone + linear probe (cheapest; the floor)
2. Frozen backbone + 2-layer MLP head
3. **LoRA adapters on top transformer blocks** (expected sweet spot; trains on M5/one A10)
4. Full fine-tune (ceiling; rented GPU, ~$100s)

Each rung reported with mAP / per-leaf F1 / params trained / GPU-hours — the ablation
table *is* the portfolio artifact.

### 4.3 Hierarchical head
Per-node sigmoid outputs over the taxonomy tree with a hierarchy-consistency constraint
(child prob ≤ parent prob, enforced via max-propagation at inference and a hierarchical
BCE penalty at train time). Inference reports the deepest node whose calibrated
probability clears its threshold — so the system says "electric guitar (distorted): 0.87"
when sure, and gracefully degrades to "guitar: 0.93" when not. Unknown instruments fall
out as high-parent/low-children patterns → surfaced as "guitar (subtype uncertain)".

### 4.4 Calibration (confidences that mean something)
Post-hoc temperature scaling per taxonomy level on a held-out real (not synthetic)
split; report Expected Calibration Error before/after. UI thresholds (0.30/0.60 from
PRD F15) then correspond to real probabilities. A "0.87" that means 87% is what
separates this from every hobby classifier.

## 5. Domain adaptation (synthetic → real)

The measurable research-flavored core:

- **Metric:** the *domain gap* = mAP(synthetic eval) − mAP(MedleyDB real eval), tracked
  per class per experiment.
- **Levers, evaluated incrementally:** (1) mastering-chain domain randomization in the
  renderer; (2) mixing real-but-coarsely-labeled OpenMIC data as level-1 supervision
  (coarse labels regularize fine heads via the hierarchy); (3) simple DANN-style
  domain-adversarial loss if 1–2 leave a stubborn gap.
- Deliverable: a plot of real-world mAP vs. % synthetic data and randomization strength —
  the classic data-centric-AI curve, from my own pipeline.

## 6. Evaluation harness (extends PRD §8)

- Splits: artist-disjoint everywhere (no song/artist leakage between train/eval).
- Metrics: mAP (macro), per-leaf F1, hierarchical F1 (partial credit for correct
  parent), ECE, per-class confusion matrices.
- Ablation matrix run and archived by the harness CLI: backbone × tuning rung ×
  {mix-only vs stem-conditioned} × {±separation-artifact aug} × {±domain randomization}.
- Promotion gate unchanged from PRD §8.3: beat incumbent on primary metric, ≤2%
  regression on secondaries, latency/VRAM recorded.
- Regression CI: every model swap requires an attached harness report (already specced,
  now with the fine-grained suite).

## 7. Serving

Fine-tuned model exported to ONNX; benchmarked ONNX-CPU vs PyTorch-MPS on the M5 with a
latency budget of ≤15s per song for the tagging stage. Model weights versioned +
checksummed in the provenance system (PRD F13) — every report says exactly which model
produced it. Training runs tracked (W&B or a local MLflow) with config-hash reproducibility.

## 8. Phasing & budget

| Milestone | What exists at the end | Compute cost |
|---|---|---|
| M1 | Taxonomy v1 + MedleyDB/Slakh/OpenMIC loaders + harness with fine-grained metrics | $0 |
| M2 | Render pipeline v1 (10k clips, 20 leaves) + frozen-MERT linear probe baseline | ~$20 render |
| M3 | Full ablation ladder (LoRA), 100k clips, 60 leaves, calibration | ~$100–400 GPU rent |
| M4 | Domain-adaptation study + separation-artifact aug + golden-set acceptance | ~$100 |
| M5 | ONNX serving swap in the app; draft-2 report shows leaf-level instruments | $0 |

Total: well under $1k, most of it optional. Compute is not the bottleneck; the render
pipeline and evaluation rigor are where the engineering hours go.

## 9. The resume paragraph this project earns

> Built an end-to-end fine-grained instrument recognition system: designed a 60-class
> hierarchical taxonomy; engineered a synthetic data pipeline rendering 100k labeled
> clips from MIDI through randomized sampler/effect/mastering chains; fine-tuned MERT
> with LoRA against linear-probe and zero-shot CLAP baselines; closed a measured
> synthetic-to-real domain gap via domain randomization and hierarchical co-training;
> temperature-calibrated outputs (ECE reported); artist-disjoint benchmark harness with
> promotion gates; served via ONNX inside a FastAPI/Next.js app with full model
> provenance.

Every claim in that paragraph maps to a milestone above and is backed by an archived
eval report — nothing is vibes.
