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

## Results

(pending)
