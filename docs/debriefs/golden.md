# Golden set: pre-registration and results

## Pre-registration (written 2026-10-16, before any model was run on these songs)

**Question.** Does instrument identification work on real, full, mastered songs, labelled
by a musician by ear, with an answer key no code of ours wrote?

**Test set (frozen).**
- 25 songs, 24 artists (Josh Woodward has 2), Jamendo, CC-licensed, downloaded via
  Jamendo's free-download links. Audio stays in `~/datasets/golden/audio/`, never committed.
- Chosen by Claude Code from MTG-Jamendo uploader tags for instrument breadth (rock / indie /
  pop / funk / folk / jazz / hip-hop). Bradley did not pre-screen them; he labelled what he
  heard. Selection bias to note: tag-picked songs may have more clearly audible instruments.
- Labels: Bradley, full listen, `~/datasets/golden/labels/` (local; not committed).
  Two corrections on Bradley's instruction before any scoring: "Spoken word" removed from
  the 18 songs where he had ticked it to mean "has vocals" (he meant singing), and
  `listened_fully` set on 3 songs he confirmed he listened to in full. Originals kept in
  `labels_original/`. Frozen label hash (sha256 of per-file sha256s): `8573c426904fab5b`.

**What is scored. Nothing is tuned or chosen on this set.**
1. Served leaf model `v7sg` (our CNN, Lakh renders, research-only). Song score = **p95 over
   10 s windows**, as the app serves it.
2. Licence-clean model (procedural MIDI, MoisesDB 0.471). Same aggregation.
3. PANNs, Platt-calibrated as served, on the full mix: coarse instruments only.
4. Chance reference: per-node prevalence on this set.
5. The app end-to-end: the real pipeline (Demucs → PANNs presence → leaf routing → report).
   Scored as what a user sees: named instruments and subtypes, confident (≥ 0.6) vs "?"
   (0.3–0.6), against Bradley's labels.

Excluded: **Essentia** (all 25 songs come from MTG-Jamendo, its training source, and its
training split is undocumented, so its scores would be contaminated).

**Metrics.**
- Primary: mean AP over nodes the model was trained on with ≥ 3 positive and ≥ 3 negative
  observed songs, reported separately for coarse (level 1) and fine (levels 2–3) nodes.
- Uncertainty: artist-bootstrap 95% CIs (resample artists). Paired bootstrap for
  v7sg vs PANNs (shared coarse nodes) and v7sg vs licence-clean.
- Nodes with < 3 positives: counts and per-song hits only, no AP.
- Sensitivity (not primary): mean and max over windows instead of p95.
- End-to-end: per family, how many of Bradley's present instruments the app shows
  (recall), how many shown instruments he did not mark (false alarms), and whether a named
  subtype matches his label when he gave one.

**Decision rules fixed now.**
- This run changes nothing in the served app by itself.
- Leaf calibration, the descent margin and p95-vs-max are follow-ups. If done, they use
  leave-one-artist-out cross-validation on this set and are reported as such, never as the
  test score.
- 25 songs is small: per-node results are evidence, not verdicts. A difference counts only
  if its paired CI excludes 0.

## Analysis decisions (2026-10-16, before any model was run; label hash re-verified `8573c426904fab5b`)

The pre-registration leaves some details open. They are fixed here, from the labels and the
models' trained-node lists only (no model output exists yet), and committed before scoring.

**Inputs.** Every model scores the app's canonical transcode (`transcode.to_canonical`:
44.1 kHz stereo PCM), not the mp3, because that is what the app scores. Item ids and audio
come from the `golden` loader.
- v7sg: `leaf_tagging.window_scores`, the served ONNX path (sidecar sha `0a1df16e1827`,
  T = 2.0342, max-propagation). One pass gives the per-window scores; p95, mean and max all
  come from them.
- Licence-clean: no sidecar exists, so `export-onnx` on `m6clean/scratch_cnn_synthetic_vproc.pt`
  (manifest `vproc/manifest.json`) into a scratch dir. It fits T on the commercially licensed
  OpenMIC-train clips, as in the M5 debrief (expected T ≈ 3.75; if not, stop). Then scored
  through the same `window_scores` code. T is fit on OpenMIC, never on this set.
- PANNs: `tagging.panns_framewise` → `tagging.calibrate` (framewise Platt, as served) → p95
  over frames, mix only. Not `panns-predict`, which takes p95 of *raw* scores.

**"Trained on".** One rule for every model: trained = the model's output nodes closed
upward, with parents max-propagated (the rule behind the v7sg sidecar's `trained` list).
- v7sg and licence-clean: the sidecar/manifest `trained` lists. They are identical (36 nodes).
- PANNs: its 20 OpenMIC-label nodes closed upward. A sensitivity uses direct outputs only.

**Eligible nodes.** ≥ 3 observed positives and ≥ 3 observed negatives, on the full set:

| model | coarse (level 1) | fine (levels 2–3) |
|---|---|---|
| v7sg, licence-clean | keys, cymbals, brass, woodwinds | guitar.acoustic, guitar.electric, guitar.electric.clean, guitar.electric.distorted, bass.electric, bass.electric.fingered, bass.electric.picked, bass.upright, bass.synth, keys.piano, keys.electric_piano, keys.organ, keys.synth, keys.synth.lead, drums.acoustic_kit, drums.electronic (16) |
| PANNs (primary, closed upward) | keys, cymbals, percussion, strings, voice, brass, woodwinds | not scored (pre-registered: coarse only) |
| PANNs (sensitivity, direct outputs) | cymbals, voice | – |

- **guitar, bass and drums have no coarse AP:** 2, 1 and 0 observed negatives. That follows
  the rule as written. For them I report recall at 0.3/0.6 and the scores on the few negative
  songs.
- **Paired comparisons use the intersection** of the two models' eligible nodes. v7sg vs
  PANNs coarse: keys, cymbals, brass, woodwinds (direct-output sensitivity: cymbals alone).
  v7sg vs clean: the full lists above. Each model's own mean is reported too.

**Mean AP and CIs.** Per-node AP comes from the harness (`eval run --dataset golden`,
archived runs). The subset mean is the harness `map_macro` over the frozen node list.
- Artist bootstrap: 24 groups, 1,000 resamples, seed 0 (harness `paired_bootstrap` and the
  same resampling for single-model CIs).
- In a resample, a node with no positives or no negatives is skipped from the mean (the
  harness rule). n_valid is reported.

**Chance.**
- Per-node prevalence, as pre-registered.
- Also the exact expected AP of a random ranking:
  E[AP] = (k−1)/(n−1) + H_n·(n−k)/(n(n−1)), with k positives among n observed songs. At small
  n this sits well above prevalence, and any "worse than chance" claim must beat it too.

**Nodes with < 3 positives.** Counts, then for each positive song the model's score and rank
among the observed songs.

**End-to-end.**
- **Run:** the real pipeline in-process (`runner.run_job`, real models, MPS) on an isolated
  data dir, one song at a time. Ingest mirrors the upload handler.
- **Shown:** every report entry is shown, because the backend drops < 0.30. "Confident"
  means ≥ 0.60 and not a subtype entry; subtype entries are always "?", as in the frontend
  `isUncertain`.
- **Mapping:** each entry maps to the taxonomy node in its own `node` field (the app's
  OpenMIC-label → node mapping; subtype entries name their node). "X (subtype uncertain)"
  lanes map to the family node.
- **Scoring an entry:**
  - hit if the node is in Bradley's positives (ancestor-closed);
  - false alarm only if the node is observed-negative;
  - unsure or unknown never counts.
- **Family = level-1 node.**
  - Recall: songs where Bradley marks the family present and the app shows any entry in it.
  - False alarm: songs where the family is observed-negative and the app shows an entry in it.
  - Families the app can't express are reported as not expressible, e.g. percussion other
    than mallets: shaker, tambourine, claps, hand drums.
- **Known label mismatches, reported separately:**
  - PANNs "piano" is fed by AudioSet "Piano" *and* "Electric piano" but maps to keys.piano.
    It is scored strictly at instrument level (a Rhodes-only song showing "piano" is a
    false alarm for keys.piano) and is fine at family level (keys). The piano/e-piano
    confusions are counted.
  - "synthesizer" maps to keys.synth. On a song whose only synth is synth bass it counts
    under keys and is noted.
- **Subtype agreement:**
  - a named subtype agrees if Bradley marked that node or a descendant present;
  - disagrees if it is observed-negative;
  - unlabelled if he said "type not sure".
  - "Subtype uncertain" lanes are counted separately.

**Out of scope for this run:** leaf calibration, descent margin and p95-vs-max selection
(the pre-registered follow-ups, which need leave-one-artist-out CV). Essentia is not run.

## Results (run 2026-10-16)

**Provenance.**
- **Labels:** frozen hash `8573c426904fab5b`, verified before the run and again at analysis
  time. The harness's own label-matrix hash is `3c3201b05da05cc1`.
- **Commits:** analysis decisions `ae280db` (committed before any model ran); scoring code
  `79de938`. Every harness run records git `79de938`, clean.
- **Harness runs:** `~/datasets/golden/eval/runs/20261016T0949*_golden_{v7sg,clean,panns}_{p95,mean,max}`.
  Prediction sha256 prefixes (p95): v7sg `2bdec33b615b`, licence-clean `9ef8b8e7bd7e`,
  PANNs `a2698a3a6d25`.
- **Glue and outputs:** in `~/datasets/golden/eval/`, outside the repo.
  - Scripts: `predict.py`, `run_app.py`, `analyze.py`, `vs_random.py`.
  - Outputs: `results.json`, `vs_random.json`, per-song reports in `reports/`.
  - Reusable code with tests: `ml/eval/subsets.py`, `ml/eval/endtoend.py`.
- **Models as scored:**
  - v7sg: served ONNX `0a1df16e1827`, T = 2.0342.
  - Licence-clean: scratch export `d049fce38c1d`, T = 3.748, fit on 1,461 commercially
    licensed OpenMIC-train clips (expected ≈ 3.75 ✓).
  - PANNs: `cnn14+platt@b8a3369efc68`.
- **End to end:** 25/25 jobs `done`, every stage `ok`.
- **Serving-path checks:**
  - All 47 PANNs confidences that came from the mix match my model-level calibrated p95.
  - All 45 subtype confidences are ≤ our p95.
  - So the model-level numbers describe what the app computes.

### Headline: mean AP over the frozen node lists (p95, artist bootstrap 95% CI, 24 artists, 1,000 resamples)

| model | nodes | mean AP | chance = prevalence | random-ranking AP | AP − prevalence | AP − random-ranking AP |
|---|---|---|---|---|---|---|
| v7sg, coarse | 4 | **0.567** [0.433, 0.740] | 0.404 [0.330, 0.505] | 0.476 | +0.163 [+0.055, +0.336] | **+0.091 [−0.019, +0.262]** |
| v7sg, fine | 16 | **0.693** [0.626, 0.794] | 0.398 [0.354, 0.458] | 0.478 | +0.295 [+0.232, +0.392] | +0.215 [+0.156, +0.308] |
| licence-clean, coarse | 4 | **0.623** [0.458, 0.786] | 0.404 | 0.476 | +0.219 [+0.066, +0.379] | +0.146 [−0.005, +0.311] |
| licence-clean, fine | 16 | **0.629** [0.592, 0.725] | 0.398 | 0.478 | +0.231 [+0.186, +0.335] | +0.151 [+0.108, +0.255] |
| PANNs, coarse (closed upward) | 7 | **0.754** [0.704, 0.854] | 0.485 [0.432, 0.553] | 0.548 | +0.269 [+0.219, +0.372] | +0.206 [+0.157, +0.303] |
| PANNs, coarse (direct outputs, sensitivity) | 2 | 0.914 [0.826, 0.978] | 0.640 [0.540, 0.731] | 0.682 | +0.274 [+0.153, +0.410] | +0.231 [+0.121, +0.356] |

**Paired deltas** (same resamples; a difference counts only if its CI excludes 0):

| comparison | nodes | Δ mean AP [95% CI] | verdict |
|---|---|---|---|
| v7sg − licence-clean, coarse | keys, cymbals, brass, woodwinds | −0.055 [−0.164, +0.075] | tie |
| v7sg − licence-clean, fine | the 16 fine nodes | +0.064 [−0.014, +0.127] | tie |
| **v7sg − PANNs, coarse** | keys, cymbals, brass, woodwinds | **−0.155 [−0.326, −0.029]** | **PANNs better** |
| v7sg − PANNs, coarse (direct-output sensitivity) | cymbals | −0.254 [−0.465, −0.025] | PANNs better (one node, labels doubtful) |

**Sensitivity: window aggregation (not primary).**

| model, level | p95 | mean | max | v7sg − clean at that aggregation |
|---|---|---|---|---|
| v7sg, fine | 0.693 | 0.660 | 0.727 | p95 +0.064 [−0.014, +0.127]; mean +0.041 [−0.023, +0.101]; max +0.100 [+0.017, +0.152] |
| clean, fine | 0.629 | 0.619 | 0.628 | – |
| v7sg, coarse | 0.567 | 0.599 | 0.552 | p95 −0.055; mean −0.013; max −0.055 (all CIs span 0) |

The single clear v7sg-over-clean delta appears only under max, a sensitivity. It is
reported, not adopted.

**What the headline says:**
- **Fine-grained, both our models clearly beat chance** under both chance bars. v7sg is at
  0.69 against a random-ranking 0.48. This is the first real, full-song,
  independently labelled evidence that the subtype scores carry signal.
- **Coarse, v7sg is not distinguishable from a random ranking** (+0.09 [−0.02, +0.26]).
  - Its coarse list has only 4 nodes. guitar, bass and drums are excluded: they have 2, 1
    and 0 negatives.
  - One of the 4 nodes is cymbals, whose negatives are doubtful (below).
  - Dropping cymbals by hand (post hoc, arithmetic only): v7sg 0.565 vs PANNs 0.688 on
    keys/brass/woodwinds. So PANNs' coarse lead isn't only cymbals.
- **PANNs is the better coarse ranker** on the shared nodes, consistent with routing
  presence to PANNs.
- **The licence cost is again not measurable:** −0.06 coarse and +0.06 fine, both ties.
  MoisesDB said the same: 0.490 vs 0.471.

### Per node (p95 AP; chance = prevalence / random-ranking AP)

Nodes the models were trained on, plus PANNs' own. Eligible (≥ 3 positives and ≥ 3 negatives) in **bold**.

| node | pos/obs | prev | rand AP | v7sg | clean | PANNs |
|---|---|---|---|---|---|---|
| guitar | 23/25 | 0.92 | 0.93 | 0.87 ↓ | 0.88 ↓ | 0.99 |
| **guitar.acoustic** | 12/24 | 0.50 | 0.56 | 0.78 | 0.69 | – |
| **guitar.electric** | 21/24 | 0.88 | 0.89 | 0.83 ↓ | 0.83 ↓ | – |
| **guitar.electric.clean** | 10/17 | 0.59 | 0.65 | 0.76 | 0.68 | – |
| **guitar.electric.distorted** | 6/17 | 0.35 | 0.45 | 0.41 ↓ | 0.43 ↓ | – |
| bass | 23/24 | 0.96 | 0.96 | 1.00 | 0.90 ↓ | 0.95 ↓ |
| **bass.electric** | 15/23 | 0.65 | 0.70 | 0.94 | 0.94 | – |
| **bass.electric.fingered** (suppressed in app) | 4/14 | 0.29 | 0.41 | 0.73 | 0.34 ↓ | – |
| **bass.electric.picked** (suppressed in app) | 3/14 | 0.21 | 0.35 | 0.87 | 0.48 | – |
| **bass.upright** | 3/25 | 0.12 | 0.22 | 0.54 | 0.75 | – |
| **bass.synth** | 7/25 | 0.28 | 0.36 | 0.44 | 0.52 | – |
| **keys** | 19/25 | 0.76 | 0.79 | 0.83 | 0.81 | 0.88 |
| **keys.piano** | 5/25 | 0.20 | 0.29 | **0.80** | 0.50 | 0.48 |
| **keys.electric_piano** | 9/25 | 0.36 | 0.44 | 0.40 ↓ | 0.47 | – |
| **keys.organ** | 3/25 | 0.12 | 0.22 | **0.75** | 0.70 | 0.22 ↓ |
| **keys.synth** | 10/24 | 0.42 | 0.49 | 0.70 | 0.63 | 0.80 |
| **keys.synth.lead** | 7/21 | 0.33 | 0.42 | 0.53 | 0.62 | – |
| **drums.acoustic_kit** | 15/21 | 0.71 | 0.75 | 0.89 | 0.87 | – |
| **drums.electronic** | 7/20 | 0.35 | 0.44 | 0.72 | 0.60 | – |
| **cymbals** | 14/25 | 0.56 | 0.61 | 0.57 ↓ | 0.74 | 0.83 |
| **brass** | 4/25 | 0.16 | 0.26 | 0.31 | 0.42 | 0.42 |
| **woodwinds** | 3/22 | 0.14 | 0.25 | 0.55 | 0.54 | 0.76 |
| **voice** (PANNs only) | 18/25 | 0.72 | 0.75 | (untrained) | (untrained) | 1.00 |
| **percussion** (PANNs only) | 20/24 | 0.83 | 0.85 | – | – | 0.93 |
| **strings** (PANNs only) | 5/22 | 0.23 | 0.33 | – | – | 0.46 |

↓ = at or below the random-ranking AP. Rows with no AP are untrained or unscored nodes.

**Where models do no better than chance (honest list):**
- **Distorted electric guitar** (both ours) is below a random ranking: 0.41 and 0.43 vs 0.45.
  On MoisesDB it was the strongest subtype (lift +0.67). That is the biggest surprise here.
- **Electric piano** (v7sg) is below random: 0.40 vs 0.44. It is suppressed in the app anyway.
- **Cymbals** (v7sg) is below random, with the label caveat below.
- **Guitar/electric guitar** (both ours, 2–3 negatives): v7sg scores *Whippy* (an EDM track
  with no guitar) 0.992 and *Je l'aime* 0.764, so it can't reject guitar on non-guitar songs.
- **PANNs organ** is at chance: 0.22 vs random 0.22. It scored 0.025–0.045 on the three
  organ songs.
- Per-node evidence from 3–5 positives, not verdicts: **v7sg ranks piano (0.80 vs PANNs 0.48)
  and organ (0.75 vs 0.22) far better than PANNs.** The bass technique nodes the app
  suppresses look strong for v7sg (0.73 / 0.87), but not for the clean model (0.34 / 0.48),
  so the instability that justified suppressing them is still there.

**Coarse nodes with too few negatives** (recall at the F15 thresholds on positives; scores on the negatives):

| node | model | recall@0.3 | recall@0.6 | scores on Bradley's negatives |
|---|---|---|---|---|
| guitar (23+, 2−) | v7sg / clean / PANNs | 1.00 / 1.00 / 0.96 | 1.00 / 0.96 / 0.78 | *Whippy* 0.99 / 0.99 / 0.04; *Je l'aime* 0.76 / 0.57 / 0.50 |
| bass (23+, 1−) | v7sg / clean / PANNs (mix only) | 1.00 / 1.00 / **0.52** | 1.00 / 1.00 / **0.00** | *Beautiful World* 0.50 / 1.00 / 0.39 |
| drums (24+, 0−) | v7sg / clean / PANNs (mix only) | 1.00 / 1.00 / 0.96 | 1.00 / 1.00 / 0.67 | – |

PANNs on the mix alone shows bass at ≥ 0.6 on **no** song. The app's bass *stem* rescues
it: end-to-end bass recall is 23/23 (below).

**Rare nodes (< 3 positives): score and rank among observed songs** (1 = top):
- **v7sg:**
  - nylon guitar (*Sonic Mystery*): 0.77, rank 3/13.
  - synth pad: *Whippy* 0.88 (2/22), *Knowing She Is Happy* 0.68 (9/22).
  - 808: *Whippy* 1/15, *Old school Jazz Hip Hop* 5/15.
  - tenor sax (*Friday Night*): rank 1/24.
  - trumpet: 0.97 and 0.96, but ranks only 5/23 and 6/23, because it fires on many songs.
  - analog synth bass: ranks 4/22 and 12/22.
- **PANNs:**
  - banjo (*Les prénoms ensablés*): 1/23.
  - saxophone: 1/24.
  - flute: 3/25.
  - clarinet: 2/24.
  - violin: 14/24.
  - cello: 12/24.
  - trombone: 19/24.
  - trumpet: 8/23 and 2/23.
  - accordion: 4/25.

### End to end: what a user sees (25 songs, real pipeline)

**153 instrument entries shown.**
- **76 confident (≥ 0.6, not a subtype):** 64 hits, 12 false alarms, so precision 0.84. Of
  the 12 false alarms:
  - 8 are cymbals (see the label caveat);
  - 2 are voice on songs Bradley marked voiceless: *Whippy* 0.98 and *Friday Night in the
    Lounge* 0.88, both from the vocals stem;
  - 1 is saxophone on *In Control* (0.61): its brass section is real, trumpet unsure;
  - 1 is "piano" on *Beautiful World* (0.98), where Bradley heard Rhodes.

  Excluding cymbals, confident precision is 64/68 = 0.94.
- **77 "?" entries:** 41 hits, 30 false alarms, 6 unlabelled.

**Per family (level 1):** recall = songs where Bradley marks the family and the app shows
it. Wilson 95% CIs.

| family | present | shown (≥ 0.3) | recall [CI] | shown confident | absent | false alarms (confident) |
|---|---|---|---|---|---|---|
| guitar | 23 | 22 | 0.96 [0.79, 0.99] | 9 | 2 | 1 (0) |
| bass | 23 | 23 | 1.00 [0.86, 1.00] | 8 | 1 | 1 (0) |
| drums | 24 | 24 | 1.00 [0.86, 1.00] | 10 | 0 | – |
| voice | 18 | 18 | 1.00 [0.82, 1.00] | 18 | 7 | 2 (2) |
| keys | 19 | 13 | **0.68 [0.46, 0.85]** | 5 | 6 | 2 (0) |
| cymbals | 14 | 14 | 1.00 [0.78, 1.00] | 13 | 11 | **11 (8)** |
| strings | 5 | 2 | 0.40 [0.12, 0.77] | 0 | 17 | 4 (0) |
| brass | 4 | 0 | **0.00 [0.00, 0.49]** | 0 | 21 | 1 (0) |
| woodwinds | 3 | 1 | 0.33 [0.06, 0.79] | 1 | 19 | 1 (1) |
| percussion | 20 | 1 | 0.05 [0.01, 0.24] | 1 | 4 | 0 |

**Per instrument the app can name:**
- **organ: 0/3.** PANNs scores 0.025–0.045 on the organ songs; v7sg's organ scores there are
  0.71, 0.81 and 0.48, but v7sg doesn't own presence.
- **piano: 2/5,** plus 1 false alarm (a Rhodes).
- **synthesizer: 8/10** shown, with **6 false alarms** in 14 synth-free songs, all "?". One
  is *silence*, whose only synth is synth bass.
- **trumpet: 0/2. trombone: 0/1. violin: 0/1. cello: 0/1.**
- **banjo: 1/1** ("?"). **saxophone: 1/1,** plus 2 false alarms.
- **mallets: 1/3.**
- **Percussion is mostly not expressible:** the app has no label for shaker (13 songs),
  tambourine, claps or hand drums. So its 1/20 percussion recall is a coverage gap, not a
  detection failure.
- **The "?" band floods on jazz.** *Friday Night in the Lounge* (swing) shows 12 entries,
  including trumpet, trombone, mandolin, cello and violin at 0.31–0.49, none of which
  Bradley heard.

**Named subtypes** (always "?"):
- **45 named:** 32 agree, 10 disagree, 3 unlabelled, so **76% agree** (Wilson [0.62, 0.87]) of
  the 42 labelled.
- **Reference:** naming those same nodes at their base rate among songs with the parent
  instrument would agree 62% of the time. That reference was not pre-registered.
- **By level:**
  - Level 2: 29/34 agree. Most named level-2 subtypes are acoustic kit (12) and electric
    bass (11), the common cases.
  - Level 3: **3/8**. Synth pad is 0/3 labelled, and nylon, on *Beautiful World*, is wrong.
- **By family:** drums 13/14, bass 11/14, guitar 7/10, synth 1/4 (labelled).
- **Disagreements:**
  - "upright bass" on the EDM track and on *North Star*;
  - "acoustic kit" on *silence*, which has electronic drums;
  - "electric bass" on *Miracle*, which has synth bass;
  - "distorted guitar" on *Came A Miracle*, which Bradley hears as clean.
- **43 lanes said "subtype uncertain".** In 30 of them Bradley did name a subtype, so the
  0.2 descent margin declines more often than it names.

### Per-song examples

1. ***Friday Night in the Lounge* (swing, instrumental).**
   - Confident "voice" (0.88, vocals stem): sax bleeding into the vocals stem is the likely
     cause.
   - Five "?" brass and orchestral entries (trumpet, trombone, mandolin, cello, violin) and a
     "?" synthesizer; acoustic guitar named where Bradley hears clean electric.
   - The worst song for false alarms: 8 of 12 entries are wrong.
   - The real instruments (kit, cymbals, sax, electric bass) are all found.
2. ***Cheeky red shoes* (blues rock, 12 labelled instruments).**
   - The app shows 5 entries. It misses guitar (PANNs 0.296, just under 0.3; v7sg guitar is
     0.93), trumpet (PANNs 0.04), clarinet and accordion.
   - Bradley's note: an unlisted organ-like instrument with tremolo. The app's
     "synthesizer ?" (0.37) counts as a false alarm, but it is plausibly that instrument.
     PANNs organ is 0.04, v7sg organ 0.36.
3. ***Morelite 13*, *I Want To Tell You*, *dark passenger* (organ songs).**
   - Keys is never shown on the first two, and organ on none of the three.
   - This is the clearest "app misses an obvious instrument" case. The model that hears
     organ (v7sg, AP 0.75) isn't the one deciding presence.
4. ***North Star* (pop).**
   - Bradley's note: a chime/thumb-piano (kalimba-like) sound throughout, which he labelled
     vibraphone. PANNs shows "mallet_percussion" confidently (0.72), counted a hit; it is
     plausibly the kalimba.
   - Piano is missed: PANNs piano 0.13, v7sg piano 0.57.
   - "Upright bass ?" (0.83) is wrong: Bradley hears electric.
5. ***Whippy* (EDM).**
   - Bradley's note: "all synth… maybe some electric piano?". Synth is found (0.84
     confident) and drum machine is named.
   - Three wrong entries: confident "voice" (0.98, vocals stem), "upright bass ?" and
     "cymbals ?". At model level, v7sg also gives guitar 0.99 here.
   - Bradley may want to re-listen for vocal chops. The label stands as frozen.

Bradley's other note: ***Not Around You*** has a shamisen-like pluck, which he labelled
"plucked strings, type not sure". The app shows no plucked strings: PANNs banjo, mandolin
and ukulele score 0.12–0.23. Our model has no such class, and its nylon score is 0.76,
possibly hearing that pluck.

### What didn't go as pre-registered, and caveats

- **Coarse AP is thin by construction.** guitar, bass and drums are excluded by the
  ≥ 3-negatives rule (2, 1 and 0 negatives), so v7sg's coarse mean rests on 4 nodes. The
  shared v7sg-vs-PANNs coarse set depends on the "trained on" rule (fixed before scoring in
  `ae280db`): 4 nodes closed upward, 1 (cymbals) for direct outputs.
- **Cymbals labels are probably a convention, not the audio.** Bradley marked cymbals
  absent on 11 songs. 9 of them have a drum kit marked (6 acoustic kit, 3 "type not sure"),
  and hi-hats and rides are near universal on a kit. The other two are *Whippy* (drum
  machine) and *Knowing She Is Happy* (drums unsure). The app showed cymbals on all 25. Per the rules these count as false alarms
  (8 confident) and as negatives in AP. Labels were not edited. Bradley should decide
  whether "drum kit" implied cymbals for him.
- **The licence-clean model had no sidecar.** It was exported to a scratch dir (not
  served) with its T fit on OpenMIC commercial clips, exactly as in the M5 debrief.
- **Added beyond the pre-registration:**
  - the exact random-ranking AP as a second chance bar (fixed in `ae280db`);
  - PANNs mean/max aggregations;
  - the subtype base-rate reference.

  None of these changes a pre-registered number.
- **Not done (out of scope, per the decision rules):** leaf calibration, the descent
  margin, p95 vs max. Any of these must use leave-one-artist-out CV on this set and be
  reported as such. Essentia was not run.
- **Small n and bias:**
  - 25 songs, 24 artists. Per-node results rest on 3–15 positives, and CIs are wide and
    asymmetric (a node with no positives in a resample drops out of the mean).
  - Songs were picked from MTG-Jamendo uploader tags for instrument breadth, so instruments
    are probably more audible than in a random song.
  - One labeller, by ear, with no second opinion. His "unsure" and "type not sure" marks
    are respected (unknown, never negative).
- **App finding:** `runner.run_job` never removes its `mkdtemp` work dir (`diss-job*`, ~150 MB
  of stems per job). It is a temp-dir leak; the eval driver cleaned it up per song. Not fixed here.

**Next (proposals, not decided here):**
1. The organ/piano presence gap (v7sg vs PANNs) is the strongest pointer for a routing
   re-check. It needs leave-one-artist-out evaluation, not this test score.
2. Distorted guitar at chance on real songs contradicts MoisesDB; look at why before
   showing that subtype.
3. Bradley: rule on the cymbals convention and the *Whippy* voice question for future
   label sets (this set stays frozen).
