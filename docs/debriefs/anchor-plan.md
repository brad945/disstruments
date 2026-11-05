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

*(pending, never cite)*
