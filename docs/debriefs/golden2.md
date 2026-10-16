# Golden set 2: pre-registration (written 2026-10-16, before any labels exist)

**Why a second set.** Golden-1 (25 songs) was used to *find* the problems that the
post-golden-1 fixes target (`docs/debriefs/fixes-after-golden1.md`). Testing those fixes on
golden-1 would be test reuse. Golden-2 is the fresh test; golden-1 becomes a dev set.

**Songs (frozen).** 30 Jamendo tracks in `~/datasets/golden2/audio/` (CC, Jamendo's free
download links), chosen by Claude Code from MTG-Jamendo uploader tags to target the weak
spots golden-1 exposed: organ (7), brass (5), strings (6), hand percussion (6),
instrumentals (3), Rhodes (3). The target group per song is recorded in
`~/datasets/golden2/selection.json` and is **not shown to the labeller**. Song order is
shuffled; the page shows no tags. Artist-disjoint from golden-1 (one candidate dropped
because it was Josh Woodward under another account). Bradley may skip songs that sound bad;
the analysis uses whatever he labels (target >= 25).

**Labels.** Bradley, by ear, same page and rules as golden-1 ("type not sure" rows
available). **Cymbals are unobserved on golden-2** (golden-1 showed a drum kit implied
cymbals to him). Labels are hashed at collection time and frozen before scoring; any
post-hoc edit needs Bradley's explicit instruction and is logged, as on golden-1.

**What is scored.**
1. **The app before vs after the fixes** (before = commit `4d24033` with its model sidecar;
   after = the app as shipped by the fixes doc). Same songs, paired by song.
2. Model level, as on golden-1: served leaf model, licence-clean model, PANNs; plus any new
   leaf model (v8) **only if** it was adopted on other data before golden-2 scoring starts.
3. Chance references (prevalence and exact random-ranking AP).
Essentia excluded (Jamendo source = its training data).

**Primary questions and metrics.**
- Q1 (the fixes): per family, recall and false alarms of what the app shows (confident
  >= 0.6 / "?" 0.3-0.6), before vs after; organ recall; voice false alarms on songs Bradley
  marked without voice; detections of the newly connected percussion/brass/strings labels;
  precision of confident entries. Paired by song; artist-bootstrap / Wilson CIs.
- Q2 (replication): mean AP over trained nodes with >= 3 positive and >= 3 negative songs,
  coarse vs fine, against random-ranking AP, as pre-registered for golden-1.
- A difference counts only if its paired CI excludes 0. Per-node results with < 3
  positives are counts, not APs.

**Decision rules fixed now.**
- Nothing is tuned on golden-2. It is the test.
- Leaf calibration / thresholds learned from human labels use golden-1 (dev) only, or are
  cross-validated by artist and reported as such; golden-2 reports their effect once.
- If a fix makes golden-2 worse, it is reported, not quietly reverted; reverting needs a
  decision recorded here.

## Amendment (2026-10-16, before any golden-2 labels exist)

The fixes round shipped Fix 2 (new "?" labels) and, by Bradley's decision, R1 for voice
(`fixes-after-golden1.md`); Fix 1 (organ) failed its gate and did not ship. So **"after" =
the app at the commit that adds R1**, plus v8 only if v8 passes its own pre-registered
adoption rule (`v8-plan.md`) and Bradley promotes it before golden-2 scoring starts.
"Before" stays commit `4d24033`. The R1 comparison (voice false alarms on songs Bradley
marks voiceless, and voice recall) is the primary Q1 item.
