# Fixes after golden-1: pre-registration and results

## Pre-registration (written 2026-10-16, before any model was run on a gate set for these fixes)

**Why.** Golden-1 (`docs/debriefs/golden.md`) found three accuracy problems in the app:
1. **Organ is never shown** (0/3). PANNs scores every AudioSet organ class near 0 on full
   mixes; our model (v7sg) hears organ, but it doesn't own presence.
2. **Percussion other than mallets can't be expressed** (shaker, tambourine, claps, hand
   drums), and neither can generic brass, string sections or harp, although PANNs has
   AudioSet classes for all of them that the app ignores.
3. **Instrumentals get confident voice** (0.88, 0.98) because the Demucs vocals stem
   vouches for voice on its own, and separation bleed makes that stem non-silent.

Bradley approved working on all three.

**The rigor rule.** Golden-1 was used to *find* these problems, so it must not choose or
gate any fix. Every choice below is made on other real data only:
- **OpenMIC-2018 test** (5,085 clips; human labels; bootstrap groups = artists);
- **MoisesDB pooled test** (720 × 10 s windows, 240 songs, 43 artists; real multitracks);
- **OpenMIC train** for fitting calibration only (as the PANNs Platt fit and our global T
  already were). MoisesDB, MedleyDB and IDMT are never fitted on.

Golden-1 is re-run afterwards and reported **only as a dev-set number, not a test**. A
fresh golden-2 will be the real test. Golden-1 labels stay frozen (hash `8573c426904fab5b`,
checked before and after); cymbals are unobserved on golden per the golden.md addendum.

**What is already known (disclosure).** These gate sets are not fresh:
- M5 chose the routing and adopted v7sg on these same MoisesDB windows and OpenMIC test.
- Already measured on MoisesDB against the app-faithful incumbent, v7sg − PANNs AP:
  piano +0.036 [−0.023, +0.114] and bass +0.006 [−0.008, +0.021], both ties.
- The Essentia comparison reported v7sg `keys.organ` AP 0.190 on MoisesDB. PANNs organ AP
  on these sets has not been looked at for v7sg.
- The m5 stem-rule table gave voice recall/precision at 0.3 for mix only (0.96 / 0.98) and
  mix + dedicated stem (1.00 / 0.96). Nothing at 0.6 was reported for voice by rule.

So these sets are independent of golden-1, but not of M5's earlier decisions.

**Label counts used to design the rules** (labels only, no model output):

| node | MoisesDB pos / neg | OpenMIC test pos / neg |
|---|---|---|
| guitar | 512 / 129 | 286 / 150 |
| bass | 514 / 180 | 134 / 329 |
| drums | 514 / 146 | 278 / 146 |
| cymbals | unobserved | 297 / 139 |
| keys.piano | 212 / 424 | 285 / 130 |
| keys.organ | 57 / 639 | 121 / 310 |
| keys.synth | 137 / 517 | 268 / 112 |
| brass.trumpet | 0 / 708 | 318 / 467 |
| woodwinds.saxophone | 0 / 702 | 305 / 324 |
| voice | 268 / 162 | 224 / 150 |
| percussion | 104 / 530 | 211 / 0 |
| percussion.{tambourine, shaker, claps, hand_drums} | 0 / 545 each (175 windows with a-tonal percussion are unobserved) | unobserved |
| brass / brass.french_horn | 9 / 708, 0 / 708 | – |
| strings.bowed / .section | 23 / 685, 17 / 685 | – |
| strings.plucked.harp | 0 / 718 | – |
| woodwinds.harmonica | 0 / 702 | – |
| keys.harpsichord | 0 / 692 | – |

**Common to every gate.**
- **Bootstrap:** the harness `paired_bootstrap` resampling: artist groups, 1,000 resamples,
  seed 0, 95% percentile CI. A difference counts only if its CI excludes 0.
- **Unit:** one 10 s clip (OpenMIC) or window (MoisesDB), scored as the app scores a song:
  p95 over PANNs frames; our model's single window score. This is the same clip-vs-song
  caveat as M5.
- **The app path for PANNs:** canonical transcode (`to_canonical`) → htdemucs → PANNs on the
  mix and on each stem, exactly as `runner` runs it. One GPU job at a time.
- **A dataset "has data" for a node** if it has ≥ 10 observed positives and ≥ 10 observed
  negatives there.

### Fix 1: who owns presence (organ, and every other node our model was trained on)

**Candidates.** Every PANNs (OpenMIC-20) label whose node v7sg was trained on: guitar,
bass, drums, cymbals, piano (`keys.piano`), organ (`keys.organ`), synthesizer
(`keys.synth`), trumpet (`brass.trumpet`), saxophone (`woodwinds.saxophone`).

**Ours.** The served v7sg (ONNX sha `0a1df16e1827`; checkpoint sha `a99ae621…`, the same
weights as the `m6gain` predictions), max-propagated as served and evaluated.

**Incumbent: PANNs as the app runs it.**
- Platt-calibrated (`panns_calibration.json`, the fit behind provenance `cnn14+platt@b8a3369efc68`).
- p95 over frames; a label's framewise score is the max of its AudioSet classes.
- For voice, drums, cymbals and bass, the max of the mix and the dedicated Demucs stem
  (`STEM_ROLES`), both calibrated.
- MoisesDB: all 720 windows through the full app path.
- OpenMIC test:
  - labels no stem vouches for: the archived mix-only PANNs predictions, which equal the
    app's incumbent up to the decode path (Platt is monotone, so AP is unchanged);
  - bass, drums and cymbals (and voice, for Fix 3): a new Demucs + PANNs run over every test
    clip observed for any of them (1,481 clips).

**Statistic.** ΔAP = ours − incumbent at the label's node, per dataset with data.

**Rule (the bass rule from M5).** Our model owns presence for a label iff both hold:
1. **beats:** on at least one dataset with data, the ΔAP CI is entirely above 0;
2. **never worse:** on no dataset with data is the ΔAP CI entirely below 0.

**Where only one dataset has data** (cymbals, trumpet and saxophone: OpenMIC only), both
conditions are judged on that dataset alone; such a pass is reported as weaker evidence.

**Multiplicity.** Nine labels × up to two datasets is up to 18 CIs with no correction,
the same as when bass was decided. A pass resting on one CI just above 0 is reported as
such.

**Calibration (only for labels that pass).** Owned presence must have calibrated
confidences, or F15's 0.30 / 0.60 mean nothing.
- **Map:** per-node Platt, p' = sigmoid(a · logit(p) + b), on top of the served score (our
  global T, then max-propagation).
- **Fit:** OpenMIC train, all 14,915 clips (the same set as T and the PANNs Platt fit),
  each node's observed clips only.
- **Applied:** per window before the p95, as PANNs' fit is applied framewise.
- **Reported, not gated:** ECE (15 bins) and recall / precision at 0.3 and 0.6, T-only vs
  Platt, on OpenMIC test and MoisesDB, next to the incumbent's.
- **Confident-precision guard:** if the calibrated owned node's precision at ≥ 0.6 is below
  0.5 on either test set (with ≥ 10 clips at ≥ 0.6), its entries ship `calibrated: false`
  (always "?") instead.

**Shipping.** Passing labels go into `export.PRESENCE`, the Platt parameters into a new
sidecar field, written by `export-onnx` (`make leaf-model`). The old sidecar is kept at
`~/.disstruments/models/leaf.v7sg-20261016`. If no label passes, Fix 1 ships nothing.

### Fix 2: connect AudioSet classes the app ignores

**Candidate map (frozen).** Each row becomes a new PANNs label: the framewise max of its
AudioSet classes, p95, from the mix only (no stem vouches for it).

| AudioSet class(es) | label | taxonomy node |
|---|---|---|
| Tambourine | tambourine | percussion.tambourine |
| Maraca, Rattle (instrument) | shaker | percussion.shaker |
| Clapping | claps | percussion.claps |
| Tabla | hand_drums | percussion.hand_drums |
| Cowbell | cowbell | percussion |
| Percussion | percussion | percussion |
| Brass instrument | brass | brass |
| French horn | french_horn | brass.french_horn |
| String section | string_section | strings.bowed.section |
| Bowed string instrument | bowed_strings | strings.bowed |
| Harp | harp | strings.plucked.harp |
| Harmonica | harmonica | woodwinds.harmonica |
| Harpsichord | harpsichord | keys.harpsichord |

**Checked and not added:**
- **Would feed an existing calibrated label** and silently invalidate its Platt fit:
  Electronic organ, Vibraphone, Double bass, Drum machine, Steel guitar, Rapping,
  Pizzicato. These need a refit, not a mapping change.
- **AudioSet parents that cover instruments the app already names**, so they would fire
  on every guitar or wind part: Plucked string instrument, Wind instrument.
- **Rare in the target genres, or no taxonomy node:** Sitar, Zither, Timpani, Gong, Wood
  block, Tubular bells, Steelpan, Orchestra, Finger snapping, and plain "Rattle" (an
  animal sound).
- **Conga and Bongo** don't exist in AudioSet 527; Cello is already mapped.

**Honesty (F15).** No clean real calibration data exists for these labels: OpenMIC lacks
them, and MoisesDB / MedleyDB are test-only. So they ship raw, flagged `calibrated: false`,
and the UI always marks them "?", never as fact. No per-class threshold is chosen on
MoisesDB.

**False-alarm check on MoisesDB** (720 windows, raw p95 from the mix, show = ≥ 0.30):
- **Without:** windows where the label's node is observed-negative.
- **With-evidence:** the node is positive, or it is unobserved and its nearest observed
  ancestor is positive. For tambourine, shaker, claps and hand drums that is the 175
  windows with a-tonal percussion: an upper bound on "with".
- **FA rate:** shows on "without" / number of "without" windows.
- **Show precision:** shows on "with-evidence" / (shows on "with-evidence" + shows on
  "without").

**Bar.** A label is kept iff:
1. its FA rate is ≤ 2%; and
2. if it shows on ≥ 5 windows in total, its show precision is ≥ 0.5 (it doesn't mostly
   fire falsely).

A label with < 5 shows that passes (1) is kept and reported as "no recall evidence".
Labels with no positives anywhere in MoisesDB (French horn, harp, harmonica, harpsichord)
can only be checked for false alarms: any show there is false.

**Scope.** Each kept label names a node of the approved taxonomy v3 (F27), shown as an
instrument detection (F7) with leaf-level names (F33) under the F15 honesty rule. A label
that can't be traced that way is not shipped; it is reported.

### Fix 3: the vocals stem vouching for voice

Notation: m = calibrated voice p95 on the mix, s = calibrated voice p95 on the vocals stem.

**Candidates (frozen, none with a free parameter).**

| rule | voice score |
|---|---|
| R0 (current) | max(m, s) |
| R1 (gate at the show threshold) | m if m < 0.30, else max(m, s) |
| R2 (mean) | (m + s) / 2 |
| R3 (min) | min(m, s) |
| R4 (mix only) | m |

The stem-energy-floor idea was dropped before measuring: it needs a threshold that would
have to be chosen on a test set.

**Data.** MoisesDB pooled (voice 268 positive / 162 negative windows) and OpenMIC test
voice (224 / 150 clips), both through the app path with Demucs stems.

**Metrics.** Per rule, for each candidate vs R0, paired artist bootstrap:
- false alarms on voice-negative clips at ≥ 0.6 (FA@0.6) and at ≥ 0.3 (FA@0.3);
- recall on voice-positive clips at ≥ 0.6 and ≥ 0.3;
- AP.

**A candidate passes iff, on both datasets:**
1. **Recall guard:** ΔR@0.3 ≥ −0.02 and ΔR@0.6 ≥ −0.02 (points), and both CI lower bounds
   ≥ −0.05.
2. **AP guard:** the ΔAP CI is not entirely below −0.01.
3. **False alarms drop:** ΔFA@0.6 ≤ 0 and ΔFA@0.3 ≤ 0 (points), and on at least one
   dataset the CI of ΔFA@0.6 or ΔFA@0.3 is entirely below 0.

**Choice.** Among passing candidates, the largest mean reduction in FA@0.6 over the two
datasets wins. Within 0.005, the smaller change wins, in the order R1, R4, R3, R2. If none
passes, R0 stays and the doc says so.

### Golden-1 re-run (dev set, not a test)

- **Run:** the same in-process driver as golden.md (`runner.run_job`, real models, isolated
  data dir), on the shipped code and sidecar. One song at a time.
- **Scoring:** the same scorer (`ml/eval/endtoend.py`) for the archived "before" reports and
  the new "after" reports, with cymbals unobserved on both (the addendum).
- **Reported before vs after:**
  - organ recall (3 songs);
  - voice false alarms on the 7 songs Bradley marked voiceless;
  - new percussion / brass / strings / woodwind detections (hits, false alarms, unlabelled);
  - confident precision.
- **Nothing is changed in response** to these numbers in this round.
- **Label hash** checked before and after.

## Results (run 2026-10-16, after the pre-registration commit `51b6f93`)

**Provenance.**
- **App-path scores:** `~/datasets/fixes_g1/collect.py` (canonical transcode → htdemucs →
  PANNs, MPS, one job at a time).
  - `moisesdb_app.npz`: 720 windows × mix + 4 stems, 25 min.
  - `openmic_app.npz`: 1,481 OpenMIC test clips × mix + vocals/drums/bass stems, 43 min.
  - Both keep raw p95 for all 527 AudioSet classes, raw and calibrated p95 per app label,
    and RMS per source. Calibration sha `b8a3369efc68`.
- **Gate scripts and outputs:** `fix1_gate.py` / `.json` (harness `paired_bootstrap`),
  `fix2_gate.py` / `.json`, `fix3_gate.py` / `.json`.
  - Fix 3 uses `fastboot.py`, a per-statistic copy of `paired_bootstrap` (same item order,
    groups and RNG draws). It reproduces the harness delta and CI exactly on a check case.
- **Ours:** the `m6gain` predictions (v7sg weights, checkpoint sha `a99ae621`, = served ONNX
  `0a1df16e1827`), harness max-propagation.
- **Sanity check against M5's own stem run** (same 720 ids): mix scores agree to ≤ 0.06
  (raw), the bass stem to 0.006. The vocals stem differs by up to 0.30 on one window,
  because Demucs on MPS is not bit-reproducible. The gates use only the new run.

### Fix 1: no label passes. Presence stays with PANNs; nothing ships.

ΔAP = v7sg − PANNs as the app runs it; paired artist bootstrap, 95% CI.

| label (node) | OpenMIC test: pos/neg, ours vs PANNs, Δ [CI] | MoisesDB: pos/neg, ours vs PANNs, Δ [CI] | verdict |
|---|---|---|---|
| guitar | 286/150, 0.942 vs 0.986, **−0.044 [−0.072, −0.021]** | 512/129, 0.945 vs 0.926, +0.018 [−0.035, +0.076] | fails: worse on OpenMIC |
| bass (incl. bass stem) | 134/329, 0.586 vs 0.721, **−0.135 [−0.214, −0.056]** | 514/180, 0.984 vs 0.977, +0.007 [−0.005, +0.022] | fails: worse on OpenMIC |
| drums (incl. drums stem) | 278/146, 0.904 vs 0.966, **−0.062 [−0.106, −0.021]** | 514/146, 0.980 vs 0.995, −0.015 [−0.030, +0.001] | fails: worse on OpenMIC |
| cymbals (incl. drums stem) | 297/139, 0.976 vs 0.984, −0.008 [−0.035, +0.013] | unobserved | fails: no beat |
| piano | 285/130, 0.985 vs 0.980, +0.005 [−0.010, +0.021] | 212/424, 0.677 vs 0.642, +0.034 [−0.025, +0.132] | fails: tie on both |
| **organ** | 121/310, 0.556 vs 0.686, **−0.130 [−0.226, −0.036]** | 57/639, 0.190 vs 0.252, −0.062 [−0.160, +0.053] | **fails: worse on OpenMIC** |
| synthesizer | 268/112, 0.788 vs 0.979, **−0.191 [−0.258, −0.128]** | 137/517, 0.441 vs 0.360, +0.081 [−0.103, +0.212] | fails: worse on OpenMIC |
| trumpet | 318/467, 0.715 vs 0.836, **−0.122 [−0.184, −0.062]** | 0 positives | fails |
| saxophone | 305/324, 0.795 vs 0.906, **−0.111 [−0.151, −0.065]** | 0 positives | fails |

- **Organ, the reason for this fix, fails clearly.** As a *ranker*, PANNs is better at
  organ on OpenMIC and level on MoisesDB. Our model's golden-1 organ AP (0.75 vs 0.22) does
  not replicate on either independent set.
- **No calibration was fitted and nothing was re-exported.** The pre-registration only
  calibrates labels that pass. `export.PRESENCE` stays `[]`, and the served sidecar is
  unchanged (sha `0a1df16e1827`; backup at `leaf.v7sg-20261016`).
- **Repeating M5's bass and piano checks** on this new app-path run gives the same ties:
  bass +0.007 vs +0.006, piano +0.034 vs +0.036.

**Why organ is missed anyway: an operating-point problem, not a ranking one.** This is
descriptive and was not pre-registered (`organ_operating_point.json`). It uses calibrated
PANNs and v7sg with its served T.

| organ positives | PANNs recall@0.3 / @0.6 | v7sg recall@0.3 / @0.6 | precision@0.3: PANNs / v7sg |
|---|---|---|---|
| OpenMIC test (121) | 0.73 / 0.52 | 0.85 / 0.49 | 0.52 / 0.36 |
| MoisesDB (57) | **0.05 / 0.05** | 0.49 / 0.12 | 1.00 / 0.12 |

PANNs ranks organ about as well as ours on MoisesDB. But its calibrated scores on real
multitracks sit far below 0.3: median 0.02 on organ windows. The Platt fit was made on
OpenMIC, where organ clips score high, and it doesn't transfer. That is the golden-1
symptom ("0.025–0.045 on the organ songs"). Our model reaches the threshold more often,
but at 12% precision it would mostly show organ where there is none. **Neither model gives
a usable organ detector on real multitracks today.** A fix would be a different
calibration or an operating-point rule. Either would need a new pre-registration and data
that isn't a test set (see "For Bradley").

### Fix 2: 12 of 13 labels kept, but they almost never fire. Shipped (`7a353e0`).

MoisesDB, 720 windows, raw p95 on the mix, show = ≥ 0.30.

| label → node | windows without / with-evidence | false shows (FA rate, Wilson 95%) | true shows | max score without | verdict |
|---|---|---|---|---|---|
| tambourine → percussion.tambourine | 545 / 98 | 0 (0.0%, ≤ 0.7%) | 0 | 0.005 | kept, no recall evidence |
| shaker → percussion.shaker | 545 / 98 | 0 | 0 | 0.005 | kept, no recall evidence |
| claps → percussion.claps | 545 / 98 | 0 | 0 | 0.001 | kept, no recall evidence |
| hand_drums → percussion.hand_drums | 545 / 98 | 0 | 0 | 0.015 | kept, no recall evidence |
| cowbell → percussion | 530 / 104 | 0 | 0 | 0.005 | kept, no recall evidence |
| **percussion → percussion** | 530 / 104 | **14 (2.6%, [1.6%, 4.4%])** | 1 | 0.49 | **dropped: FA > 2%, show precision 0.07** |
| brass → brass | 708 / 9 | 0 | 0 | 0.077 | kept, no recall evidence |
| french_horn → brass.french_horn | 708 / 9 | 0 | 0 | 0.005 | kept, no recall evidence |
| string_section → strings.bowed.section | 685 / 23 | 0 | 0 | 0.013 | kept, no recall evidence |
| bowed_strings → strings.bowed | 685 / 23 | 0 | 0 | 0.080 | kept, no recall evidence |
| harp → strings.plucked.harp | 718 / 1 | 0 | 0 | 0.245 | kept, no recall evidence |
| harmonica → woodwinds.harmonica | 702 / 14 | 0 | 0 | 0.014 | kept, no recall evidence |
| harpsichord → keys.harpsichord | 692 / 19 | 0 | 2 | 0.069 | kept, no recall evidence |

- **What it means:** the 12 kept labels are safe (no false shows in 530–718 windows each),
  but on real multitracks they almost never reach 0.30. There were 2 true shows in total, both
  harpsichord. They fix the *coverage* gap (the app can now say "tambourine ?"), not the
  *detection* gap. Raw PANNs scores sit far below the F15 thresholds (m5 story 4) and there
  is no non-test data to calibrate them on.
- **Generic "Percussion"** is the AudioSet parent of drum kits, and it fired on kit-only
  windows, as expected. It is the only label dropped.
- **Scope:** every kept label names a node of the approved taxonomy v3 (F27), is shown as a
  detection (F7) with its own name (F33), and is always "?" (F15). Nothing needed a PRD
  amendment.
- **Code (`7a353e0`):**
  - `tagging.EXTRA_AUDIOSET_MAP` / `EXTRA_LABEL_NODES` / `label_nodes()`;
  - `calibrated` is now per label (true only for labels the Platt fit covers);
  - every PANNs entry carries its node;
  - the stage version names the label map (`+map@<sha8>`);
  - the UI and `endtoend.shown_as` treat `calibrated: false` as "?";
  - the library filter finds new labels by name and by family.

### Fix 3: no candidate passes. The vocals stem keeps vouching (R0); nothing ships.

Voice on calibrated PANNs; Δ vs R0 with paired artist bootstrap CIs. MoisesDB: 268 positive
and 162 negative windows, 43 artists. OpenMIC test: 224 / 150 clips, 794 artist groups in
the Demucs subset.

| rule | MoisesDB FA@0.6 / FA@0.3 | MoisesDB R@0.6 / R@0.3 | OpenMIC FA@0.6 / FA@0.3 | OpenMIC R@0.6 / R@0.3 | fails |
|---|---|---|---|---|---|
| R0 max (current) | 0.031 / 0.074 | 0.978 / 0.996 | 0.153 / 0.247 | 0.991 / 0.996 | – |
| R1 gate at 0.30 | 0.000 (−0.031 [−0.071, 0.000]) / 0.025 (−0.049 [−0.089, −0.015]) | 0.959 (−0.019 [−0.041, −0.005]) / **0.963 (−0.034 [−0.074, −0.008])** | 0.087 (−0.067 [−0.113, −0.030]) / 0.120 (−0.127) | 0.973 (−0.018) / 0.978 (−0.018) | recall guard: MoisesDB R@0.3 −0.034 |
| R2 mean | 0.000 / 0.037 | **0.944 (−0.034 [−0.068, −0.012])** / 0.985 (−0.011) | 0.067 (−0.087 [−0.139, −0.042]) / 0.167 | **0.964 (−0.027)** / 0.996 | recall guard: R@0.6 on both |
| R3 min | 0.000 / 0.000 | **0.840 (−0.138)** / **0.959 (−0.037)** | 0.047 / 0.067 | **0.915 (−0.076)** / **0.964 (−0.031)** | recall guard |
| R4 mix only | 0.000 / 0.025 | **0.847 (−0.131)** / **0.963 (−0.034)** | 0.080 / 0.120 | **0.955 (−0.036)** / 0.978 | recall guard |

AP guard and false-alarm conditions: every candidate passes them. **Every candidate fails
the recall guard** (a point drop larger than 0.02 somewhere). By the pre-registered rule
R0 stays.

**What it shows:**
- **The vocals stem is a real source of confident false voice.** On OpenMIC, 15% of
  voiceless clips show confident voice under the current rule, against 8% from the mix
  alone.
- **R1 comes closest:**
  - it halves those OpenMIC false alarms (0.153 → 0.087) and removes all of MoisesDB's;
  - it costs about 2 recall points on OpenMIC and 2–3.4 on MoisesDB;
  - the bar was 2.
- **Shipping R1 anyway would be choosing on test data after seeing it.** The bar stays as
  written. Whether a ~3-point recall cost is worth halving confident false voice is a
  product trade-off for Bradley (see below).
