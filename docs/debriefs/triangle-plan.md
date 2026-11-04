# Triangle experiment: pre-registration (written 2026-11-04, before any run) and results

**Why.** M3's headline is that a 1.22M CNN ties MERT-95M on real music (OpenMIC trained-9
0.785 vs 0.774) when both are trained on synthetic renders only. "Ties" alone can't say
whether both are good or both hit the same synthetic-to-real wall: there is no real-trained
reference. This was the literature check's top missing experiment
(`docs/research/literature-check.md` §(e)). Bradley approved it on 2026-11-03 ("run that
test extensively").

## Corners (both models)

| corner | trains on | early stopping on |
|---|---|---|
| **c1** synthetic only | v1 renders (8,216 clips) | v1 synthetic val (892) |
| **c2** real only | OpenMIC train minus val (12,678 clips) | OpenMIC val (2,237, artist-disjoint carve-out of OpenMIC train, seed 0) |
| **c3** synthetic, then real fine-tune | c1 checkpoint of the same seed, then the c2 data | OpenMIC val |

Models:
- **cnn**: the M3 1.22M scratch CNN, same config (no gain augmentation, as in v1/M3).
  - c1, c2: 40 epochs, peak LR 2e-3, patience 8.
  - c3: 20 epochs, peak LR 2e-4 (10x lower), patience 8.
- **mert**: the frozen MERT-95M linear probe (layer-weighted).
  - c1, c2: LR 1e-2.
  - c3: LR 1e-3, keeping c1's input normalisation.
  - 200 epochs, patience 15.

Seeds 0, 1, 2 for every arm. c1-cnn seed 0 is the M3 checkpoint itself
(`preds/m3/scratch_cnn.pt`), so the published claim is one of the seeds.

**Real-data curve.** c2 and c3 also at nested subsets of the 12,678 train clips: 254 (2%),
1,268 (10%), 6,339 (50%), 3 seeds each. Same val set, same configs.

**Data choice (Bradley's standing rules).**
- Real training data is OpenMIC train only.
- MoisesDB stays test-only: nothing is trained, tuned or early-stopped on it.
- Golden-2 is untouched (unlabelled, reserved for its own pre-registration).

**Reference rows (not part of the decision rules).**
- **c3j-cnn**: M4 L1, joint synthetic + all-OpenMIC co-training, 1 seed, already measured.
  - OpenMIC trained-9 0.870.
  - MoisesDB −0.04 to −0.05 vs v1.
  - It is a prior: joint real training helped in-domain and hurt out-of-domain.
- **v7sg**: the served model.
  - c1 as already measured (1 seed).
  - plus one c3 fine-tune from it (seed 0), CNN only.

## Tests and metrics

1. **OpenMIC test** (5,085 clips): mean AP over **trained-9** (the 9 OpenMIC classes the v1
   leaves cover; the M3 metric). In-domain for c2/c3 (same source, same labelling), so it
   flatters them.
2. **Pooled MoisesDB** (720 clips, 3 windows per song): mean AP over the trained-9 nodes it
   can score (>= 3 positive and >= 3 negative clips). **Out-of-domain for every corner.
   This is the primary readout.**
3. **MoisesDB leaf retention**: mean AP over v1-trained leaves MoisesDB can score, for c1
   vs c3 (does coarse real fine-tuning erase subtypes?). c2 learns no leaves and is not
   scored here.
4. Secondary: OpenMIC **untrained-11** (only c2/c3 can learn these).

All deltas are paired artist bootstraps (`harness.paired_bootstrap`, 1,000 resamples, boot
seed 0), computed per seed (seed k vs seed k). Scoring code: `ml/eval/triangle.py`.

**A difference counts only if every seed's paired CI excludes 0, with the same sign.**
Otherwise it is "no detectable difference", whatever the point estimate.

## Primary questions and how each outcome is read

- **Q1. How big is the synthetic-to-real gap?** c2 − c1, per model, on MoisesDB (primary)
  and OpenMIC.
  - Gap on OpenMIC but not MoisesDB: the "gap" is mostly OpenMIC in-domain advantage, not
    sim-to-real.
  - Gap on both: a real sim-to-real gap, sized.
- **Q2. Does MERT's advantage only show with real labels?** mert − cnn within c1 and within
  c2, on both tests.
  - "Only with real labels" requires: c2 mert − cnn > 0 (counts, by the rule above) on
    MoisesDB, while c1 mert − cnn shows no detectable difference.
  - If c2 also ties, the small CNN matches the foundation model with real labels too.
- **Q3. Is synthetic pretraining worth anything once real data exists?** c3 − c2 at each
  curve point, per model.
  - The expected pattern is a benefit that shrinks as real data grows.
  - Report the smallest real-data size where c3 − c2 stops counting.
- **Q4. Does fine-tuning forget subtypes?** c3 − c1 on MoisesDB leaf retention (CNN, and
  MERT probe).

## Fixed now

- Nothing is tuned after seeing a test score. Configs above are final; a failed run is
  re-run with the same config, and the failure is reported.
- If an arm diverges or stops at epoch <= 2, it is reported as-is, not re-tuned.
- The served model (`~/.disstruments/models/leaf`) is never touched. Outputs go to
  `~/datasets/disstruments-synth/preds/tri/` and `eval_runs/` with names `tri_<arm>_s<seed>`.
- One GPU job at a time (16 GB RAM). Estimated ~18 h of compute.

## Results (2026-11-04, run complete)

All arms ran to completion with the configs above. Nothing was re-tuned. Tables:
`~/datasets/disstruments-synth/preds/tri/tables/*.md` (1,000-resample paired artist bootstrap).
Mean AP over 3 seeds unless noted; "counts" = every seed's paired CI excludes 0, same sign.

**Incident (scoring, not training).** `scratch` tagged its pooled-MoisesDB predictions
`moisesdb_all`, which the harness rejects, and the run script's error filter hid the message,
so every CNN arm trained in this run had no MoisesDB score. Fixed in code
(`models/__main__._pred_dataset`, shared with `scratch-predict`, with a test). The label was
corrected on the existing files (ids, nodes and scores verified bit-identical), and they were
scored with the unchanged harness. No predictions were regenerated.

### Main table

| | OpenMIC trained-9 (in-domain for c2/c3) | MoisesDB trained-9 (primary, 6 nodes) | MoisesDB leaves (7) |
|---|---|---|---|
| c1 cnn | 0.778 | 0.656 | 0.461 |
| c1 mert | 0.774 | 0.670 | 0.485 |
| c2 cnn | 0.901 | 0.652 | (no leaves trained) |
| c2 mert | 0.867 | 0.666 | (no leaves trained) |
| c3 cnn | 0.898 | 0.654 | 0.345 |
| c3 mert | 0.874 | **0.708** | 0.444 |
| ref: c1 v7sg (served, 1 seed) | 0.805 | 0.703 | 0.566 |
| ref: c3 v7sg (1 seed) | 0.893 | 0.662 | 0.355 |
| ref: c3j cnn (joint, 1 seed) | 0.870 | 0.659 | 0.402 |

### Q1. Synthetic-to-real gap (c2 − c1)

| | OpenMIC | MoisesDB |
|---|---|---|
| cnn | +0.123, counts | −0.004, no detectable difference |
| mert | +0.093, counts | −0.004, no detectable difference |

**Gap on OpenMIC but not MoisesDB, for both models.** By the pre-registered reading, the
OpenMIC "gap" is mostly in-domain advantage, not sim-to-real. On independent real music,
training only on our synthetic renders matches training on 12,678 human-labelled real clips.

### Q2. Does MERT's advantage only show with real labels? (mert − cnn)

| | OpenMIC | MoisesDB |
|---|---|---|
| c1 | −0.004, no difference | +0.014, no difference |
| c2 | **−0.033 (CNN better)**, counts | +0.014, no difference |

**No.** The 1.22M CNN ties MERT-95M with synthetic labels and with real labels on the primary
test, and beats it in-domain with real labels. The M3 tie now holds on 3 seeds and both tests.

### Q3. Is synthetic pretraining worth anything once real data exists? (c3 − c2)

| real clips | cnn OpenMIC | cnn MoisesDB | mert OpenMIC | mert MoisesDB |
|---|---|---|---|---|
| 254 | −0.012, no | **+0.076, counts** | +0.069, counts | +0.066, counts |
| 1,268 | −0.046, no | −0.014, no | +0.044, counts | +0.070, counts |
| 6,339 | −0.004, no | +0.015, no | +0.019, counts | +0.053, counts |
| 12,678 (all) | −0.003, no | +0.003, no | +0.007, no | **+0.042, counts** |

- **MERT:** the benefit shrinks in-domain as expected (stops counting on OpenMIC only at full
  data), but on MoisesDB it **never stops counting**. Synthetic-then-real is the best recipe
  measured (0.708).
- **CNN:** the benefit counts only at 254 clips and stops counting at 1,268.
- Likely mechanism (secondary test, not a decision rule): on the OpenMIC untrained-11 classes
  (learnable only from real data), c3 cnn − c2 cnn = **−0.095 (counts)**, while c3 mert (0.723)
  is above c2 mert (0.689). The pre-registered CNN fine-tune (peak LR 2e-4, 20 epochs) under-learns
  the new classes. So the CNN result says "this fine-tune recipe adds nothing", not "synthetic
  pretraining is worthless for the CNN". Testing a higher LR is a separate, labelled follow-up.

### Q4. Does fine-tuning forget subtypes? (c3 − c1, MoisesDB leaves)

- cnn: **−0.116**, counts.
- mert: **−0.041**, counts.
- served v7sg: −0.211 (1 seed, CI excludes 0); its fine-tune also drops MoisesDB trained-9
  0.703 → 0.662.

**Yes.** Coarse real labels ("piano") overwrite subtype knowledge ("Rhodes"), more so for the
CNN. The served model stays the synthetic-only v7sg.

### Post hoc (not pre-registered; 300-resample bootstrap)

On MoisesDB, c3 mert − c3 cnn = +0.054 and counts. With this fine-tune recipe, MERT gets more
out of the synthetic-then-real combination than the CNN does.

### What can be cited

1. The CNN ties MERT-95M on real music when both train on the same synthetic data:
   - 3 seeds;
   - both OpenMIC and MoisesDB;
   - it also ties with real labels on MoisesDB.
2. On out-of-domain real music (MoisesDB), synthetic-only training matches real-only training
   on OpenMIC, for both models. The large OpenMIC gap is in-domain advantage.
3. For MERT, synthetic pretraining then real fine-tuning beats real-only out-of-domain, at
   every real-data size tested, including all 12,678 clips.
4. Fine-tuning on coarse real labels erases fine-grained subtype skill (both models).

Caveat on scope:
- MoisesDB trained-9 is 6 nodes over 720 clips (240 songs).
- Real training data is one source (OpenMIC / FMA).
