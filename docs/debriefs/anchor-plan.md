# Anchor plan: pre-registration (written 2026-11-04, before any run)

**Why.** The triangle (`triangle-plan.md`) found two things about fine-tuning the 1.22M CNN
on real coarse labels (OpenMIC): it erases subtype skill (MoisesDB leaf AP 0.461 -> 0.345,
3 seeds, counts), and synthetic pretraining adds nothing out-of-domain beyond 254 real
clips, while a frozen MERT-95M probe keeps both. A literature search
(`docs/research/robust-finetune-search.md`, 174 sources) says both are predicted by known
mechanisms and that the cheapest fixes come first:

1. The leaf rows got no gradient while the shared features drifted under them; forgetting
   concentrates in late layers and BatchNorm statistics; fine-grained classes forget more
   than coarse ones (Ramasesh 2021; Casciotti & Mesaros 2026; PACE 2026).
2. The 11 OpenMIC classes with no synthetic positives have effectively random head rows,
   which distort shared features during fine-tuning (Kumar et al. 2022, LP-FT).
3. The pre-registered fine-tune recipe (peak LR 2e-4, 20 epochs, OneCycle re-warm from 0)
   is 10-20x hotter and 4x shorter than published audio practice (EfficientAT on OpenMIC:
   1e-5 to 2e-5, 80 epochs, wd 0). The notes disagree on which direction to move.
4. Transfer benefit scales as N^0.38, so 78x parameters buys ~5x effective transfer; a 1.8x
   wider CNN buys ~1.25x. Width-vs-forgetting effects need 8-64x width to detect; at 2x
   the prediction is ~0.01 AP, below our 3-seed resolution.

Bradley approved this plan on 2026-11-04 ("yes") with width 2.0x (~4.8M, PANNs CNN10
channels) in place of 3.7-4M.

**Project facts folded in (answered 2026-11-04).**
- The M4 joint run (`c3j`) concatenated 8,216 synthetic + 14,915 OpenMIC clips with no
  weighting: ~36% synthetic / 64% real. That is above the 10-25% replay range the
  literature recommends, and M4 found it hurt MoisesDB (-0.04 to -0.05).
- The scratch LR 2e-3 was an M3 default, not validation-selected. So the 1x width arm goes
  through the same scratch LR grid as 2x (W0 is required).
- Measured wall-clock on this Mac (triangle log): fine-tune, 12,678 clips, 20 epochs =
  **36 min = 1 FT**; real-only from scratch, 40 epochs = 86 min; synthetic scratch at 1x =
  50-67 min; fine-tune at 1,268 clips ~6 min; real-only at 1,268 ~15 min.

## Test-set discipline

MoisesDB and the golden sets are scored once per final seed, after all selection, by the
unchanged harness; nothing in Phases 0-3 reads them. OpenMIC test is likewise scored once.
All selection uses only:
- the OpenMIC validation split (artist-disjoint carve-out of OpenMIC train, 15%, seed 0,
  2,237 clips; the same split every triangle arm used): all-20 mAP, trained-9 mAP,
  untrained-11 mAP;
- the synthetic v1 validation split (892 clips): leaf AP, with the caveat that it rewards
  staying synthetic.

Golden-2 is unlabelled and stays untouched.

## Taxonomy rules enforced in every arm

- Downward propagation of observed negatives to all descendants (target 0) and upward
  propagation of observed positives to ancestors (target 1) are exact under the taxonomy
  and are the only hard label completions used.
- "Coarse positive => at least one child on" is never added: a parent may be positive with
  all children negative (implicit "other", `ML_ENGINEERING.md` S2).
- Soft teacher targets under positive parents are probabilities, never a forced child;
  descendants of negatives stay hard zeros.

## Held fixed across all arms unless the arm says otherwise

Architecture (4 blocks of 2x[3x3 conv, BN, ReLU] + 2x2 avg-pool; channels 32-64-128-256
except Phase 3), depth, kernels, pooling (receptive field unchanged), dropout 0.3, AdamW
weight decay 1e-2, batch 32, OneCycle shape (except the no-re-warm arm), epochs 40 scratch /
20 fine-tune (except the 80-epoch arm), SpecAugment policy, no gain augmentation (as in
v1/M3; the triangle used none), mono upmix, early stopping on validation mAP with patience
8, seeds {0, 1, 2} shared across arms (seed 0 c1 = the M3 checkpoint), masked BCE, the
OpenMIC artist-disjoint split, and the evaluation harness.

Teacher for distillation and replay source in Phases 1-2: the **same-seed c1 checkpoint**
(v1 Lakh renders, research-only). This is a research experiment; a served model built from
the winning recipe must be re-run with the procedural licence-clean renders and teacher
(`preds/m6clean`), and that re-run is a separate, labelled step.

## Arms, by phase

**Phase 0: zero-cost diagnostics on existing checkpoints (~2 h).**
- P0.1 `WISE`: theta(alpha) = (1-alpha) theta_c1 + alpha theta_c3 for each seed, alpha in
  {0, 0.1, ..., 1}, the untrained-11 head rows held at theta_c3 (they were random in c1).
  BatchNorm running statistics are interpolated too. Curves of OpenMIC-val coarse mAP
  (trained-9 and all-20) and synthetic-val leaf AP.
- P0.2 `REFIT`: freeze the c3 trunk, refit the 65 leaf rows on synthetic train (L2
  logistic regression on cached 512-d pooled features), read synthetic-val leaf AP.
- P0.3 `CKA`: per-block linear CKA and pooled-feature L2 change between c1 and c3 on
  held-out synthetic val and OpenMIC val clips.
- Reading: concave `WISE` curves => shared basin, theta_alpha becomes a scored arm; a dip
  => barrier, confirming the hot-LR story. `REFIT` recovering most of the synthetic-val leaf
  AP => head-level cure; little recovery => trunk drift, Phase 2 anchors needed. `CKA`
  blocks 3-4 dropping => a freeze-3-4 surgical variant is justified; block 1 moving =>
  input-level shift.

**Phase 1: fine-tune recipe re-sweep at 1.22M, seed 0, from c1_s0 (~8.5 h).**
Every run logs per-tensor ||g||/||theta|| in epoch 1 (Auto-RGN), relative weight change from
theta_c1 per block, early-stop epoch, train-val gap.
- R1 LR grid at 12,678 clips: {2e-5, 6e-5, 2e-4, 6e-4, 2e-3} (5 FT; 2e-4 is the existing
  c3_s0).
- R2 the same grid at 1,268 clips (5 x ~0.15 FT).
- R3 `DISC`: backbone 2e-5, untrained-11 rows 1e-3, leaf rows 0, BN running stats frozen
  (1 FT).
- R4 `NOWARM+BNFREEZE`: best R1 LR, flat-then-decay from at most the synthetic run's final
  LR, BN running stats frozen (1 FT).
- R5 `AUDIO80`: 2e-5, 80 epochs, wd 0 (the EfficientAT OpenMIC recipe) at 12,678 (4 FT).
- R6 `MASK11`: 2e-4 with the untrained-11 rows masked out of the loss (1 FT): the
  feature-distortion test.
- R7 real-only c2 at 1,268 with scratch grid {1e-3, 2e-3, 4e-3} (3 x ~0.4 FT), because the
  triangle's c2 LR was not validation-selected.
- Selection rule for LR*: highest OpenMIC-val all-20 mAP subject to synthetic-val leaf AP
  >= that of the 2e-4 run; ties within 0.005 go to the smaller LR. The whole curve is
  reported.

**Phase 2: anti-forgetting arms at LR*, 12,678 clips, 3 seeds (~12 h).**
- F0 `BASE*`: plain fine-tune at LR* (control; 3 FT).
- F1 `PROP`: F0 + exact downward/upward label propagation (3 FT).
- F2 `PROP+GATED-LWF`: F1 + hierarchy-gated soft distillation from the same-seed c1
  teacher (cached logits, T = 2) on children of OpenMIC-positive parents, per-leaf quantile
  threshold; lambda in {0.3, 1, 3} on seed 0, then seeds 1-2 at lambda* (~5.5 FT).
- F3 `PROP+REPLAY`: F1 + r in {0.1, 0.25} synthetic clips per batch, fresh clips each
  epoch, on seed 0, then seeds 1-2 at r* (~5 FT).
- Post hoc on every arm: WiSE-FT alpha on validation (evaluation only); the best
  alpha-version of each arm is scored alongside it.
- lambda*, r*, alpha are selected on OpenMIC-val trained-9 and synthetic-val leaf AP
  jointly: maximise leaf AP subject to trained-9 >= F0 - 0.01.

**Phase 3: width, essential arms (~27 h).**
- W0: 1x scratch LR grid {1e-3, 2e-3, 4e-3} x seed 0 (~3 h; 2e-3 exists).
- W1: 2.0x (64-128-256-512, ~4.8M) scratch LR grid x seed 0 (~12 h), then seeds 1-2 at the
  winner = c1(2x) (~8 h). Gate G3 first.
- W2 at 1,268 clips: fine-tune LR grid {LR*/3, LR*, 3 LR*} x seed 0, then c3(2x) x 3 seeds
  and c2(2x) x 3 seeds at their own validated LRs (~3-4 h).
- W3 (optional, gated by G5): c3(2x) at 12,678 clips x 3 seeds (~8 h), to put width on the
  same footing as Phase 2 for leaf forgetting.
- W4 (optional): 1.5x arm (48-96-192-384, ~2.7M) as W1-W2 (~11-14 h); dropped first if the
  budget is cut.

**Final scoring (~2 h).** Every surviving arm x 3 seeds on MoisesDB (trained-9, 7 leaves),
golden-1, OpenMIC test, once.

## What is swept, and on which validation set

| Quantity | Grid | Selected on |
|---|---|---|
| Fine-tune LR at 1x | {2e-5 ... 2e-3} | OpenMIC-val all-20 mAP, constrained by synthetic-val leaf AP |
| Distillation weight lambda | {0.3, 1, 3} | OpenMIC-val trained-9 + synthetic-val leaf AP |
| Replay ratio r | {0.1, 0.25} | same |
| Interpolation alpha | {0, 0.1, ..., 1} | same (plus the full curve reported) |
| Scratch LR per width | {1e-3, 2e-3, 4e-3} | OpenMIC-val trained-9 mAP |
| Fine-tune LR per width | {LR*/3, LR*, 3 LR*} | OpenMIC-val all-20 mAP |

## Decision rules

- Statistic: per seed, 1,000-resample paired bootstrap over MoisesDB artists of
  (arm - control); an effect **counts** only if all three seeds' 95% CIs exclude 0 with the
  same sign (the triangle rule).
- **H-a1 (retention).** For F1-F3 and their alpha-versions: MoisesDB leaf AP (arm - F0)
  counts positive. **Retention achieved** if the arm's leaf AP lies within the paired CI of
  c1's 0.461 in all seeds, and OpenMIC-test trained-9 >= F0 - 0.01. MERT's -0.041 is the
  reference for "retention by a stronger representation".
- **H-a2 (no coarse cost).** MoisesDB trained-9 (arm - F0) does not count negative.
- **H-b1 (LR confound).** At 1,268 clips, c3(LR*) - c2 counts positive on MoisesDB
  trained-9 at 1x. If it does, the OOD-gain question is attributed to the recipe and H-c3
  is pre-declared closed.
- **H-b2 (distortion source).** `MASK11` - c3(2e-4) counts on MoisesDB trained-9 or leaf AP
  => the untrained-11 rows were the distortion source.
- **H-c1 (width, performance).** c1(2x) - c1(1x) counts positive on MoisesDB trained-9;
  secondary: leaves, golden-1. Prior +0.01 to +0.03.
- **H-c2 (width, forgetting).** Difference-in-differences
  [c1(2x) - c3(2x)] - [c1(1x) - c3(1x)] on MoisesDB leaf AP has a paired CI excluding 0 and
  negative in all seeds (W2 at 1,268; W3 at 12,678 if run). Prior: a few thousandths to
  ~0.01 AP. **Declared underpowered at 3 seeds in advance:** a null is "not detected", not
  "no effect". The absolute post-fine-tune leaf AP at 2x is reported as the product-relevant
  quantity.
- **H-c3 (width, pretraining gain).** c3(2x) - c2(2x) at 1,268 counts positive with both at
  their own validated LRs; interpretable only if H-b1 fails.

## Expected outcomes and how each is read

| Observation | Reading | Consequence |
|---|---|---|
| `WISE` concave; some alpha keeps >= 95% of c1's synthetic-val leaf AP and >= 90% of c3's coarse gain | Shared basin; forgetting is a weight-displacement effect | theta_alpha becomes a served candidate; Phase 2 still runs (cheap); W3 deprioritised |
| `WISE` has a dip | Barrier; 2e-4 OneCycle left the basin | Supports "too hot"; Phase 1 low-LR arms are decisive; re-interpolate from the Phase-1 winner |
| `REFIT` recovers most synthetic-val leaf AP | Suppressed, not erased; head-level | Freeze/refit leaf rows is enough; anchors are insurance |
| `REFIT` recovers little; `CKA` drops in blocks 3-4 | Trunk drift in late layers | F2/F3 anchors expected to work; surgical freeze-3-4 is a justified extra arm |
| A higher LR wins Phase 1 and closes untrained-11 under-learning on val | LR was too cold for the new classes | H-b1 likely; width (c) closed; check leaf cost |
| A lower LR or `DISC` wins Phase 1 with better leaf retention | LR was too hot for the trunk | "Two knobs" story confirmed; carry `DISC` into Phase 2 |
| `AUDIO80` beats both | Steps, not peak LR, were the limit | Adopt the long low-LR schedule as the default recipe |
| F2 or F3 reaches retention with no coarse cost | Anchoring suffices at 1.22M | Width not needed for (a); W3 optional |
| No Phase-2 arm reaches retention | Capacity or representation limit | W3 becomes essential; consider label-free routes (AdaBN, domain confusion) |
| H-c1 counts, H-c2 null | Width buys performance, not retention | Matches the literature prior; serve the wider synthetic-only model with anchors |

## Kill rules / phase gates

- **G0 (after Phase 0).** If `WISE` dips for all seeds, mark the 2e-4 run as "barrier" and
  require the Phase-1 winner to be re-interpolated before any Phase-2 arm is promoted.
- **G1 (after Phase 1).** If no grid LR improves OpenMIC-val all-20 mAP over 2e-4 by
  > 0.005 under the leaf constraint, LR* = 2e-4 and the LR-confound hypothesis is rejected
  for under-learning. If an LR >= 6e-4 or `DISC` brings c3 - c2 on OpenMIC-val untrained-11
  to >= -0.02, pre-declare H-c3 closed as LR-explained: W2's c2(2x)/c3(2x) arms drop to 1
  seed each (sanity), and W1 continues for H-c1 only.
- **G2 (after Phase-2 seed-0 sweeps).** Drop any (lambda, r) whose synthetic-val leaf AP is
  below F0's or whose OpenMIC-val trained-9 is > 0.02 below F0 before spending seeds 1-2.
- **G3 (before W1).** Time one epoch at 2x. If a scratch run projects above 6 h, switch to
  1.75x (56-112-224-448, ~3.66M). If batch 32 does not fit, use gradient accumulation to
  keep effective batch and steps fixed and record it.
- **G4 (during W1).** If 2x's early-stop epoch is under half of 1x's or its train-val gap
  exceeds 1x's by > 0.05 mAP, flag overfitting; complete the seeds anyway.
- **G5 (W3 gate).** Run the full-data c3(2x) only if W2's difference-in-differences is
  negative in all 3 seeds at 1,268; otherwise skip.
- **Budget cut order.** W4 (1.5x) first, then W3, then R5 (`AUDIO80`), never Phase 0-1.

## Per-run logging

Chosen LR; early-stop epoch; final train and val mAP and gap; parameter count and pooled
embedding size; wall-clock; relative weight change from init (scratch) or from theta_c1
(fine-tune), overall and per block; per-tensor ||g||/||theta|| in epoch 1; pooled-feature L2
change on synthetic val; validation calibration temperature.

## Compute estimate (one job at a time, 16 GB Apple Silicon; 1 FT = 36 min measured)

| Phase | Content | Hours |
|---|---|---|
| 0 | WiSE sweep, head refit, CKA | ~2 |
| 1 | 5 + 5 LR runs, `DISC`, `NOWARM+BNFREEZE`, `AUDIO80`, `MASK11`, c2 grid | ~8.5 |
| 2 | F0-F3 (~16.5 FT) + evals | ~12 |
| 3 essential | W0 (~3), W1 (~20), W2 (~3.5) | ~27 |
| Final scoring | all arms x 3 seeds, once | ~2 |
| **Essential total** | | **~52 h (~3-4 calendar days)** |
| 3 optional | W3 (+8), W4 (+11-14) | up to ~74 h |

## Fixed now

- Nothing is tuned after seeing a test score. Configs above are final; a failed run is
  re-run with the same config and the failure reported.
- The served model (`~/.disstruments/models/leaf`) is never touched. Outputs go to
  `~/datasets/disstruments-synth/preds/anchor/<arm>_s<seed>` and `eval_runs/` with names
  `anc_<arm>_s<seed>`.
- Code for each phase (interpolation, head refit, CKA, propagation, gated distillation,
  replay, discriminative LR, BN freeze, schedule variants, width) is added with unit tests
  before the phase runs, and smoke-tested on tiny data.
- Phase gates are applied in order; skipping a gate is reported.

## Results

### Phase 0 (2026-11-04, 3 seeds, ~8 min; `preds/anchor/p0/`, `logs/anchor_p0.log`)

Validation metrics only (OpenMIC-val is in-domain for c3; synthetic-val rewards staying
synthetic). MoisesDB untouched.

**P0.1 `WISE`.** Every curve is monotone in alpha for every seed: coarse OpenMIC-val rises,
synthetic-val leaf AP falls, with no dip anywhere. Mean over seeds:

| alpha | OpenMIC-val trained-9 | all-20 | untrained-11 | synthetic-val leaf |
|---|---|---|---|---|
| 0.0 (c1 + c3's untrained-11 rows) | 0.721 | 0.487 | 0.296 | 0.932 |
| 0.2 | 0.763 | 0.572 | 0.417 | 0.921 |
| 0.3 | 0.786 | 0.615 | 0.476 | 0.906 |
| 0.5 | 0.823 | 0.675 | 0.554 | 0.831 |
| 0.7 | 0.848 | 0.710 | 0.596 | 0.653 |
| 1.0 (c3) | 0.867 | 0.731 | 0.620 | 0.395 |

- **G0 does not fire:** no barrier, so c1 and c3 share a basin; 2e-4 did not leave it.
- The pre-registered promotion criterion (>= 95% of c1's leaf AP and >= 90% of c3's coarse
  gain at one alpha) is **not met** at any alpha: the trade-off is smooth. alpha = 0.3 keeps
  97% of the leaf AP for 45% of the coarse gain; alpha = 0.5 keeps 89% for 70%.
- Reading: forgetting here is a continuous weight-displacement effect that interpolation
  can trade but not undo. theta_alpha is kept as a post-hoc option on every Phase-2 arm,
  not promoted on its own.

**P0.3 `CKA`.** Blocks 1-2 barely moved (CKA 0.97 / 0.87-0.96); blocks 3-4 moved a lot
(0.41-0.68 / 0.38-0.44). Relative weight change grows with depth: block 1 ~6%, block 2
~9%, block 3 ~11%, block 4 ~16%, head 37-46%. Pooled 512-d features changed by ~100% of
their norm. Reading per the plan: **trunk drift in late layers**; F2/F3 anchors are the
expected fix, and a freeze-blocks-3-4 surgical arm is justified as an extra in Phase 2.

**P0.2 `REFIT`.** Refitting only the 65 leaf rows on the frozen c3 trunk recovers
synthetic-val leaf AP 0.385 / 0.369 / 0.431 -> 0.753 / 0.749 / 0.783 (seeds 0-2); the same
procedure on c1 costs 0.003 (0.930 -> 0.929), so the refit itself is lossless. About
two-thirds of the lost leaf skill is still readable from c3's features by a new head;
one-third is gone from the trunk. Reading: **partly suppressed, partly erased**. A
post-fine-tune leaf-head refit is added as a cheap post-hoc step on every Phase-2 arm
(alongside WiSE alpha), and the anchors are needed for the remaining third.

**Consequences for Phase 1-2 (decided now, before Phase 1 runs).**
- Phase 1 runs as pre-registered.
- Phase 2 adds one surgical arm, F4 `PROP+FREEZE34` (blocks 3-4 and leaf rows frozen, BN
  running stats frozen), justified by the CKA reading; and two post-hoc steps on every arm:
  WiSE alpha and leaf-head refit on synthetic train, both selected on validation.
- The no-re-warm arm R4 uses a flat LR at LR* for 70% of steps then a linear decay to 0
  (the pre-registered "at most the synthetic run's final LR" is uninformative for OneCycle,
  whose final LR is ~1e-10).

### Phase 1 (2026-11-04 21:13 -> 11-05 15:38, seed 0; `preds/anchor/p1/`, `logs/anchor_p1.log`)

Validation only (OpenMIC-val all-20 / trained-9 / untrained-11 mAP; synthetic-val leaf AP).
Test predictions were written and not read. c1 (synthetic-only) and c2 (real-only, 2e-3 scratch)
are scored by the same code as references. Wall-clock: one fine-tune 36-41 min; `AUDIO80`
2 h 37 min. The Mac was on battery and hibernated 04:45-12:43 at 1% charge; the job paused
and resumed, every arm completed all its epochs (checked per log), nothing was re-run.

| arm (12,678 clips unless noted) | trained-9 | all-20 | untrained-11 | leaf AP | best ep | block-4 / head weight change |
|---|---|---|---|---|---|---|
| c1 reference (no fine-tune) | 0.724 | 0.468 | 0.259 | 0.930 | | |
| c2 reference (real-only scratch) | 0.891 | 0.793 | 0.714 | 0.265 | | |
| R1 2e-5 | 0.745 | 0.568 | 0.424 | 0.395 | 16 | 6% / 17% |
| R1 6e-5 | 0.798 | 0.630 | 0.493 | 0.341 | 17 | 10% / 34% |
| R1 2e-4 (triangle c3) | 0.872 | 0.737 | 0.626 | 0.385 | | 16% / 40% |
| **R1 6e-4 = LR\*** | 0.894 | 0.788 | 0.701 | 0.397 | 16 | 31% / 56% |
| R1 2e-3 | 0.893 | 0.810 | 0.742 | 0.353 | 18 | 75% / 82% |
| R3 `DISC` | 0.848 | 0.738 | 0.648 | **0.840** | 18 | 3% / 52% |
| R4 `NOWARM+BNFREEZE` @6e-4 | 0.895 | 0.792 | 0.708 | 0.339 | 17 | 51% / 90% |
| R5 `AUDIO80` (2e-5, 80 ep) | 0.855 | 0.695 | 0.563 | 0.380 | 76 | 17% / 48% |
| R6 `MASK11` @2e-4 | 0.884 | (0.584) | (0.339) | 0.585 | 14 | 16% / 42% |
| R2 2e-5, 1,268 | 0.690 | 0.505 | 0.354 | 0.852 | 5 | |
| R2 6e-5, 1,268 | 0.677 | 0.503 | 0.360 | 0.816 | 2 | |
| R2 2e-4, 1,268 (triangle) | 0.711 | 0.541 | 0.401 | 0.414 | | |
| R2 6e-4, 1,268 | 0.752 | 0.598 | 0.472 | 0.348 | 19 | |
| R2 2e-3, 1,268 | 0.802 | 0.649 | 0.524 | 0.349 | 16 | |
| R7 c2 1e-3, 1,268 | 0.806 | 0.668 | 0.555 | 0.259 | 34 | |
| R7 c2 2e-3, 1,268 (triangle) | 0.791 | 0.658 | 0.549 | 0.247 | | |
| R7 c2 4e-3, 1,268 | 0.757 | 0.631 | 0.527 | 0.252 | 34 | |

(`MASK11`'s all-20 and untrained-11 are in parentheses: those rows were excluded from its loss.)

**LR\* = 6e-4** by the pre-registered rule: highest all-20 among LRs whose leaf AP is >= the
2e-4 run's 0.385 (6e-4: 0.397 passes; 2e-3: 0.353 fails). It beats 2e-4 by +0.051 all-20 and
+0.075 untrained-11, far above the 0.005 threshold, so **G1's first clause does not fire**: the
triangle's fine-tune LR was 3x too cold.

**G1 second clause fires.** c3 - c2 on OpenMIC-val untrained-11: 2e-4 -0.087 (the triangle's
"under-learning", -0.095 on test); 6e-4 **-0.012**; 2e-3 +0.028; R4 -0.006; `DISC` -0.066.
An LR >= 6e-4 brings it to >= -0.02, so **H-c3 is pre-declared closed as LR-explained**: the
untrained-11 deficit was a recipe artefact, not a pretraining cost. Per the gate, W2's c2(2x)
and c3(2x) arms drop to 1 seed each (sanity) and W1 continues for H-c1 only. Whether the hotter
fine-tune also moves the out-of-domain numbers is a test-time question (H-b1, final scoring).

**At 1,268 clips pretraining still does not win in-domain at any LR.** c3's best all-20 is
0.649 (2e-3) and 0.598 at LR\*; validated c2 reaches 0.668 (1e-3). The leaf constraint keeps
2e-4 at this size (every hotter LR drops leaf AP to ~0.35 vs 0.414), so the Phase 3 W2 grid
{LR\*/3, LR\*, 3 LR\*} = {2e-4, 6e-4, 1.8e-3} brackets it as intended. R7 shows the triangle's
c2 LR (2e-3) was within 0.01 of the validated best, so the triangle's real-only control was
fair.

**Low LRs under-learn and forget anyway.** 2e-5 reaches 0.568 all-20 and still loses leaf AP
to 0.395 with only 6% weight travel in block 4. `AUDIO80` (the EfficientAT recipe) reaches
0.695 after 80 epochs, best epoch 76 and still rising, leaf 0.380: at this model size the
published audio recipe is slow on the coarse axis and no better on the leaf axis.

**Schedule is not a lever.** R4 (flat 6e-4, no warm-up, BN statistics frozen) ties OneCycle
6e-4 on all-20 (0.792 vs 0.788) and loses more leaf AP (0.339 vs 0.397); without warm-up the
weights travel further (block 4 51% vs 31%). OneCycle is kept for Phase 2.

**`DISC` is the only arm that keeps the leaves** (0.840 = 90% of c1's 0.930) and it pays for
it on the coarse axis: 0.738 all-20, the 2e-4 level, because its 2e-5 backbone under-learns
exactly like R1 2e-5; the hot head rows (1e-3) recover most but not all of the gap. Its weight
change is <= 2.7% in every block. Coarse learning and leaf retention are not yet obtained
together by any Phase-1 recipe; that is Phase 2's job.

**`MASK11` (H-b2, validation read).** Removing the 11 never-trained rows from the loss halves
the leaf loss at the same LR (0.585 vs 0.385) and slightly raises trained-9 (0.884 vs 0.872).
The randomly initialised untrained-11 rows are a real source of feature distortion (the LP-FT
mechanism of Kumar et al. 2022: large head gradients from random rows drag the trunk). The
MoisesDB read of H-b2 waits for the final scoring.

**Post hoc: BN-statistics swap** (`anchor bnswap`, evaluation only, `preds/anchor/p1/bnswap.json`).
c1's weights with an arm's BatchNorm running statistics lose 0.23-0.34 leaf AP on their own
(0.930 -> 0.588 at 2e-5, 0.637 at 6e-4, 0.639 at 2e-3, 0.703 after `AUDIO80`, 0.789 after
`MASK11`). Swapping c1's statistics back into a fine-tuned arm recovers nothing and usually
hurts (6e-4: 0.397 -> 0.293; 2e-5: 0.394 -> 0.407), because the weights co-adapted to the
shifted statistics. Reading: **the BN-statistics shift to OpenMIC is a large forgetting
channel by itself, locked in within the first epoch regardless of LR** (running statistics
adapt by momentum, not by the optimizer), which is why 2e-5 forgets with 6% weight travel.
Freezing BN statistics is necessary for retention (`DISC`) but not sufficient once the
weights travel (R4). Leaf AP is therefore not monotone in weight travel: two channels, BN
statistics at any LR and weight displacement at high LR.

**Consequences for Phase 2 (proposed to Bradley 2026-11-05, before any Phase-2 run).**
- LR\* = 6e-4 for F0-F4, OneCycle, 20 epochs, as pre-registered.
- Two cheap seed-0 arms are added, each promoted to seeds 1-2 only if it passes G2:
  F0b `BASE*+BNFREEZE` (6e-4 OneCycle with BN statistics frozen; isolates the BN channel from
  R4's no-warm-up confound) and F5 `LP-FT` (head-only fit of the 20 OpenMIC rows for 3 epochs
  at 1e-3 with the trunk frozen, then the F0 recipe; the `MASK11` reading says the random
  untrained-11 rows distort the trunk, and LP-FT is the published fix). If F0b beats F0 on
  leaf AP at no coarse cost, the anchor arms F1-F3 run on top of BN freeze.
- Everything else in Phase 2 is unchanged.

### Phase 2 amendments (2026-11-06, written before any Phase-2 run)

Bradley approved adding F0b and F5 ("include it too, whatever aids the training results").
The rest below are implementation decisions recorded before the first run.

- **F1 `PROP` is already in force and is not run.** The OpenMIC loader
  (`datasets/openmic.py`) builds `positive = close_upward(positives)` and
  `negative = close_downward(negatives) - positive - masked`, and marks both as observed. Every
  triangle and Phase-1 arm therefore already trained with exact upward/downward propagation;
  F1 is identical to F0 and the "PROP+" prefix in F2-F4 adds nothing. This also sharpens the
  Phase-0 reading: leaf rows did receive gradients during fine-tuning, as hard negatives under
  OpenMIC-negative parents, so the leaf loss is not drift alone.
- **F2 gate = hierarchy gate only (quantile 0).** Distilled entries are the unobserved strict
  descendants of each clip's observed-positive nodes; observed entries (hard labels, including
  propagated negatives) are never distilled. The per-leaf quantile is implemented
  (`--distill-quantile`) but left at 0: with soft targets it would delete the teacher's "low"
  signal, which belongs to hard pseudo-labelling, not to distillation. Smoke on 64 clips: 6.4
  gated entries per clip, 67% of clips have a gate.
- Teacher logits are cached once per run on the full 10 s clips without augmentation; the
  student sees random 8 s crops with SpecAugment (standard offline distillation). T = 2 with the
  T^2 gradient scaling; lambda in {0.3, 1, 3} as pre-registered.
- **Replay appends** round(r x 32) fresh random synthetic train clips to every batch (3 for
  r = 0.1, 8 for r = 0.25) with their full labels; the 32 real clips per step and the number of
  steps are unchanged from F0, and the positive weights come from the real labels only.
- **BN-statistics freeze is crossed with F2 and F3 at seed 0** (arms `F2b`, `F3b`), because
  Phase 1 showed the statistics shift is a forgetting channel of its own; lambda* and r* are
  selected over both variants. F0b (`BASE*+BNFREEZE`) is the plain control for that factor.
- **F5 `LP-FT`**: 3 head-only epochs at a constant 1e-3 (every trunk parameter and all BN
  statistics frozen), then the F0 recipe (OneCycle 6e-4, 20 epochs, all parameters).
- Promotion to seeds 1-2: F0 always; F2* and F3* = highest synthetic-val leaf AP subject to
  OpenMIC-val trained-9 >= F0 - 0.01, among arms that pass G2; F0b, F4, F5 if they pass G2
  (leaf AP >= F0's and trained-9 >= F0 - 0.02). Arms that fail stay at seed 0 and are reported.
- Post hoc on every promoted arm x seed: WiSE alpha curve and leaf-head refit, validation only.
- Budget: 14 seed-0 fine-tunes + up to 12 for seeds 1-2, about 18 h. Script:
  `~/datasets/disstruments-synth/run_anchor_p2.sh`; outputs `preds/anchor/p2/`.

### Phase 2 (2026-11-06 01:45 -> 10:27 seed-0 sweep; seeds 1-2 from 10:35; `preds/anchor/p2/`, `logs/anchor_p2*.log`)

Validation only, as before. All 14 seed-0 arms completed (36-45 min each; the Mac was plugged
in this time). The seed-1/2 promotion step first failed on a filename (the triangle saved
seeds 1-2 as `scratch_cnn_synthetic.pt`, seed 0 as `scratch_cnn.pt`); the launcher was pointed
at both names and resumed with nothing re-run. The pre-registered gate and selection
(`selection.txt`, `promote.txt`) were applied by the script, not by hand.

| arm (seed 0, LR\* = 6e-4, 20 ep) | trained-9 | all-20 | untrained-11 | leaf AP | best ep | block-1 rel. grad norm, ep 1 | block-4 / head weight change | G2 |
|---|---|---|---|---|---|---|---|---|
| c1 reference | 0.724 | 0.468 | 0.259 | 0.930 | | | | |
| F0 `BASE*` | 0.895 | 0.789 | 0.703 | 0.397 | 16 | 0.174 | 31% / 56% | pass |
| F0b `BASE*+BNFREEZE` | 0.890 | 0.778 | 0.687 | 0.485 | 19 | 0.153 | 28% / 45% | pass |
| **F5 `LP-FT`** | 0.884 | **0.803** | **0.736** | 0.552 | 14 | **0.064** | 37% / 81% | pass |
| F4 `FREEZE34` | 0.836 | 0.687 | 0.565 | 0.332 | 17 | 0.267 | 0% / 42% | drop |
| F2 `LWF` lambda 0.3 | 0.876 | 0.780 | 0.701 | 0.758 | 17 | 0.181 | 35% / 58% | pass, not promoted |
| F2 lambda 1 | 0.869 | 0.768 | 0.686 | 0.833 | 19 | 0.213 | 38% / 55% | drop |
| F2 lambda 3 | 0.848 | 0.742 | 0.655 | 0.868 | 18 | 0.303 | 37% / 56% | drop |
| F2b lambda 0.3 + BNFREEZE | 0.877 | 0.774 | 0.690 | 0.753 | 17 | 0.180 | 33% / 60% | pass, not promoted |
| F2b lambda 1 + BNFREEZE | 0.869 | 0.763 | 0.676 | 0.832 | 17 | 0.254 | 36% / 59% | drop |
| F2b lambda 3 + BNFREEZE | 0.852 | 0.735 | 0.640 | 0.870 | 17 | 0.413 | 37% / 59% | drop |
| F3 `REPLAY` r 0.1 | 0.838 | 0.740 | 0.661 | 0.919 | 15 | 0.204 | 30% / 68% | drop |
| F3 r 0.25 | 0.815 | 0.721 | 0.644 | **0.926** | 18 | 0.138 | 29% / 67% | drop |
| F3b r 0.1 + BNFREEZE | 0.832 | 0.731 | 0.649 | 0.914 | 15 | 0.207 | 30% / 55% | drop |
| F3b r 0.25 + BNFREEZE | 0.812 | 0.715 | 0.635 | 0.924 | 19 | 0.156 | 29% / 58% | drop |

(G2: leaf AP >= F0's and trained-9 >= F0 - 0.02 = 0.875. Promotion for F2/F3 additionally needs
trained-9 >= F0 - 0.01 = 0.885. Distillation gate: 7.6% of label entries, 6.5 per clip.)

**F0 reproduces Phase 1.** Same recipe as R1 6e-4: 0.895 / 0.789 / 0.703 / 0.397 vs
0.894 / 0.788 / 0.701 / 0.397 (epoch 0 bit-identical; the rest is MPS non-determinism).

**`LP-FT` is the only arm that improves the new classes and leaf retention at once:** vs F0,
all-20 +0.014, untrained-11 +0.033, leaf AP +0.155, trained-9 -0.011. Three head-only epochs on
the frozen c1 trunk already reach 0.705 all-20 (a linear probe of c1's features; c1 itself scores
0.468 because its 11 untrained rows are random). The mechanism is the one `MASK11` pointed at:
with a fitted head, the first-epoch relative gradient norm on block 1 is 0.064 vs 0.174 for F0
(2.7x smaller), so the trunk is not jolted while the head is random (Kumar et al. 2022). The
total weight travel ends up similar (block 4 37% vs 31%), but the leaf-head refit recovers 0.852
from the LP-FT trunk vs 0.787 from F0's: more leaf information survives in the trunk. It also
converges earliest (best epoch 14, lowest train loss at best).

**BN freeze alone helps (+0.088 leaf AP for -0.005 trained-9), and adds nothing on top of
distillation or replay:** F2b vs F2 within 0.005 on every metric; F3b vs F3 slightly worse. The
statistics channel from Phase 1 matters when nothing else keeps synthetic features in use; once
distillation or replay does, the shift is compensated.

**Distillation trades leaf AP for coarse mAP smoothly:** leaf 0.758 / 0.833 / 0.868 for
trained-9 -0.019 / -0.026 / -0.047 and all-20 -0.009 / -0.021 / -0.047 at lambda 0.3 / 1 / 3.
lambda 0.3 passes G2 but misses the pre-registered promotion constraint by 0.009 on trained-9,
so by the rules **the F2 family stops at seed 0**. It is still the best leaf-per-coarse trade in
the sweep (+0.36 leaf AP for -0.009 all-20) and the obvious partner for `LP-FT` in a combined
arm, which was not pre-registered and is proposed to Bradley below rather than run.

**Replay retains the leaves completely and pays the most for it:** 0.919-0.926 leaf AP
(c1: 0.930) with trained-9 -0.057 (r = 0.1) and -0.080 (r = 0.25), all-20 -0.049 / -0.068. Three
synthetic clips per 32 real ones cost far more coarse accuracy than their share of the batch:
the synthetic clips' coarse labels pull the shared coarse nodes toward synthetic-domain decision
boundaries (the domain gap measured in M3-M4). Replay restricted to leaf nodes (coarse nodes
masked on replay rows) is the natural follow-up; not pre-registered, proposed below.

**Freezing blocks 3-4 is the wrong lever.** With blocks 3-4, the 65 leaf rows and the BN
statistics all frozen, leaf AP still collapses, to the lowest value in the sweep (0.332), and
coarse learning is crippled (0.687 all-20). Only blocks 1-2 could move, and they moved more
(block-1 gradient norm 0.267 vs 0.174): the frozen late layers were co-adapted to the old early
features, so early-layer change alone destroys the leaf readout. The Phase-0 CKA reading ("late
layers drift, so freeze them") identified where the weights move, not what breaks the leaves;
`DISC` (cold everywhere) in Phase 1 and `LP-FT` here both protect the early layers instead. F4
is rejected.

**Post hoc (seed 0; WiSE and refit on the three promoted arms).** The alpha curves are again
smooth and monotone; `LP-FT`'s curve lies above F0's at every alpha on all-20, untrained-11 and
leaf AP (alpha 0.8: 0.871 / 0.793 / 0.729 / 0.705 vs F0 0.882 / 0.774 / 0.685 / 0.577). Refit:
F5 0.852 > F0b 0.824 > F0 0.787 > Phase-0 c3 at 2e-4 0.753.

**Against the pre-registered readings.** No arm reaches "retention with no coarse cost" under
the rules (>= 0.90 x c1's leaf AP within trained-9 -0.01), so by the plan's table **W3 (the
full-data fine-tune at 2x) becomes essential** in Phase 3. `LP-FT` and `BNFREEZE` go to the final
scoring with 3 seeds; F2 lambda 0.3 and F3 r 0.1 are scored once at seed 0 as reported arms.

**Seeds 1-2 (F0, F0b, F5): SEEDS_PENDING**

**Proposed to Bradley (not run; each is a pre-registration amendment):**
1. `P2x LP-FT+LWF0.3(+BNFREEZE)`: the two mechanisms are different (head-first vs gated
   teacher), so the gains should add; 2 seed-0 fine-tunes (~1.5 h), promoted to seeds 1-2 under
   the same G2 + selection rules. Recommended.
2. `F3c LEAF-REPLAY`: replay rows with the coarse nodes masked, r = 0.1; 1 fine-tune (~40 min).
   Tests the contamination reading. Recommended if (1) runs.
3. `W3b`: `LP-FT` at 2x on full data x 3 seeds, the "width + best anchor" product candidate,
   after W3. Decide after W1 shows whether 2x helps at all.

### Phase 3 amendments (2026-11-06, written before any Phase-3 run)

- `--channels` added to `scratch` (the model, checkpoints, diagnostics and `anchor` tools were
  already width-agnostic; parameter counts: 1x 1,216,760; 1.5x 2,702,920; 1.75x 3,665,456;
  2x 4,774,296, with 86 outputs).
- **G3** is applied by the script: one 2x epoch is timed; if 40 epochs project above 6 h the
  width drops to 1.75x, and the choice is written to `preds/anchor/p3/g3.txt`.
- **W0**: 1x scratch grid {1e-3, 4e-3} at seed 0 (2e-3 = c1_s0 exists). Pre-declared: seeds
  1-2 are re-run at a new 1x LR only if it beats 2e-3 by > 0.005 OpenMIC-val trained-9;
  otherwise the triangle's c1 seeds stay the 1x reference.
- **W2 fine-tune LR at 1,268 uses the Phase-1 rule** (highest all-20 subject to leaf AP >= the
  LR\*/3 = 2e-4 run's), not all-20 alone, so the 2x LR is validated the same way the 1x LR was
  (2e-4 at this size) and H-c2's difference-in-differences compares like with like.
- **c3(2x) at 1,268 keeps 3 seeds** (G1 dropped it to 1 for H-c3, but H-c2 needs the three
  paired differences; cost ~0.5 h); c2(2x) at 1,268 runs at 1 seed as G1 says.
- **G5 is not applied as written** (it reads MoisesDB mid-plan, which the test-set discipline
  forbids): W3 runs unconditionally because Phase 2 made it essential, at LR\* = 6e-4 for the
  same footing as F0; WiSE and refit post hoc on each seed.
- **G4** is read from best epoch and train loss at best (train mAP is not logged).
- Script: `~/datasets/disstruments-synth/run_anchor_p3.sh`; outputs `preds/anchor/p3/`.
