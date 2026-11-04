# Triangle experiment: pre-registration (written 2026-11-04, before any run)

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

## Results

*(pending — never cite)*
