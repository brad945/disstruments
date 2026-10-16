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

## Results

(pending)
