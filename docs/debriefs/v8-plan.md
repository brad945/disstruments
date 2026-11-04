# v8 plan: teach the leaf model the families it has never heard (pre-registration)

> Written 2026-10-16, **before any v8 render or training**. Branch
> `worktree-agent-a8260c3606686e629` (from `m5`). Nothing here has been run except unit tests
> and a 5-clip CPU smoke render. The adoption rule below is fixed now; results get appended
> under "Results" without changing it.

## Why

The served model (`v7sg`) was trained on rock/pop roles only: piano, organ, guitar, bass,
trumpet, tenor sax, synths, drums. GM strings, most brass, woodwinds, mallets,
accordion/harmonica and non-kit percussion were *dropped* from every render. On the golden set
that showed up as a coverage gap, not a model error: strings 2/5, brass 0/4, percussion 1/20
(shaker, tambourine, claps, hand drums have no label at all).

The lever is the one M4 proved: **distinct instruments per leaf**, not volume. A leaf trained
on one sampler learns that sampler (leave-source-out: 1-2 instruments collapse, 3+ generalize).

## What the code now does

- **GM → leaf** (`ml/synth/midi.py`). New roles, each with GM program hints that pin the exact
  leaf (label exactness comes from rendering the chosen leaf's own source, as before):

  | role | GM programs (0-based) | leaves |
  |---|---|---|
  | strings_solo | 40 violin, 41 viola, 42 cello | violin, viola, cello, section |
  | strings_section | 44 tremolo, 48/49 ensembles | section |
  | harp / plucked | 46 / 105 banjo | harp / banjo, mandolin |
  | brass | 56 trumpet, 57 trombone, 58 tuba, 59 muted tpt, 60 horn, 61 section (62/63 synth brass, unhinted, as in v7) | trumpet, trombone, French horn, tuba, section |
  | reed | 64-67 saxes, 68/69 oboe + English horn, 70 bassoon, 71 clarinet | 4 sax types, oboe, bassoon, clarinet |
  | flute | 72 piccolo, 73 flute | flute ("Flute / piccolo") |
  | mallet | 9 glock, 11 vibes, 12 marimba, 13 xylophone | the 4 mallet leaves |
  | accordion / harmonica | 21, 23 / 22 | keys.accordion / woodwinds.harmonica |
  | perc_* (drum channel, by note) | 54 / 69,70,82 / 39 / 60-64 | tambourine / shaker / claps / hand drums |

  Dropped (never rendered, so never unlabelled): celesta, music box, tubular bells, dulcimer,
  bowed contrabass (43), pizzicato strings (45), timpani, synth strings, choir, recorder/pan
  flute/whistle/ocarina, sitar/koto, and drum notes with no leaf (cowbell, timbales,
  agogo, guiro, claves, wood blocks, cuica, triangle).
- **Solo rule.** A part with more than 2 simultaneous notes never becomes solo
  violin/viola/cello (a solo sampler playing 4-note chords sounds like a section). It goes to
  `strings.bowed.section` instead.
- **Cymbals stay kit-only.** Percussion parts are split off the drum channel by note; they
  never contain hat/cymbal notes, so only a rendered kit part can imply `cymbals` (tested end
  to end).
- **v1 builds are unchanged in what they parse** (`roles=` filter); `--leaf-set v1` is still
  the default.
- **Distinct-instrument gate** (`sources.leaf_gate`, `build --leaf-set v8`): a new leaf is used
  only if ≥ 2 *distinct instruments* (`group`) render it with an *exact* label from files that
  are on disk and approved. Coarse sources (`label_as`) don't count. Articulations of one
  recording share a group. FluidR3 counts as one instrument (its kits even share the
  percussion samples: identical peaks across 5 kits in the smoke check). The manifest records
  the gate.
- **Sources** (`sources.yaml`): FluidR3 GM programs for all 28 new leaves (low weight 0.15,
  all audible in a CPU smoke check, **not yet audited by ear**); Iowa alto/soprano sax are now
  exact leaves; the NSynth entry covers the new leaves; an 808 clap; **52 pending entries**
  (VSCO-2 CE, Iowa MIS, VCSL) marked `status: pending_approval` until Bradley approves.
- **Note engine**: drum one-shots play key-exact (no resampling) and ring out.
  `notes_prep iowa --set v8` and `notes_prep vcsl` prepare the pending sources (not run).
- **Adoption check is code** (`python -m disstruments.ml.eval.adopt`), so the rule below is
  executed, not interpreted. Smoke-checked on archived runs: v7sg vs v7s on pooled MoisesDB
  passes `old` (+0.051 over 21 nodes; CI [+0.034, +0.065] from only 50 resamples), matching
  M5's direction.

## Per-leaf sources (trainable = ≥ 2 distinct instruments, exact labels)

"Today" = on disk now. "After approval" = plus the pending downloads (sizes below). NSynth
instruments are counted only where Bradley has already labelled them.

| leaf | today | after approval | instruments after approval |
|---|---|---|---|
| strings.bowed.violin | 1 ✗ | **3** ✓ | FluidR3, Iowa violin (arco), VSCO solo violin |
| strings.bowed.viola | 1 ✗ | **2** ✓ thin | FluidR3, Iowa viola |
| strings.bowed.cello | 1 ✗ | **2** ✓ thin | FluidR3, Iowa cello |
| strings.bowed.section | 1 ✗ | **4** ✓ | FluidR3, VSCO violin / viola / cello sections |
| strings.plucked.harp | 1 ✗ | **4** ✓ | FluidR3, VSCO harp, VCSL concert + folk harp |
| strings.plucked.banjo | 1 ✗ | 1 ✗ | FluidR3 only; no second source found |
| strings.plucked.mandolin | 0 ✗ | 0 ✗ | no GM program and no source |
| brass.trombone | 1 ✗ | **3** ✓ | FluidR3, Iowa tenor trombone, VSCO trombone |
| brass.french_horn | 1 ✗ | **3** ✓ | FluidR3, Iowa horn, VSCO horn |
| brass.tuba | 1 ✗ | **3** ✓ | FluidR3, Iowa tuba, VSCO tuba |
| brass.section | 1 ✗ | 1 ✗ | FluidR3 "Brass Section" only |
| woodwinds.saxophone.soprano | **3** ✓ | 3 ✓ | FluidR3, Iowa soprano, NSynth reed_025 |
| woodwinds.saxophone.alto | **3** ✓ | 3 ✓ | FluidR3, Iowa alto, NSynth reed_001 |
| woodwinds.saxophone.baritone | **4** ✓ | 4 ✓ | FluidR3, NSynth reed_026/046/050 |
| woodwinds.clarinet | **5** ✓ | 7 ✓ | FluidR3, NSynth ×4, + Iowa, VSCO |
| woodwinds.flute | 1 ✗ | **4** ✓ | FluidR3, Iowa flute, VSCO flute, VSCO piccolo |
| woodwinds.oboe | 1 ✗ | **3** ✓ | FluidR3, Iowa, VSCO |
| woodwinds.bassoon | 1 ✗ | **3** ✓ | FluidR3, Iowa, VSCO |
| woodwinds.harmonica | 1 ✗ | **3** ✓ | FluidR3, VCSL Hohner Special 20 (C+F), VCSL Super 64 |
| keys.accordion | 1 ✗ | 1 ✗ | FluidR3 only (unless NSynth reeds get accordion labels) |
| percussion.mallet.glockenspiel | 1 ✗ | **4** ✓ | FluidR3, Iowa bells, VCSL, VSCO |
| percussion.mallet.vibraphone | 1 ✗ | **3** ✓ | FluidR3, Iowa, VCSL |
| percussion.mallet.marimba | 1 ✗ | **4** ✓ | FluidR3, Iowa (yarn), VCSL, VSCO |
| percussion.mallet.xylophone | 1 ✗ | **4** ✓ | FluidR3, Iowa, VCSL, VSCO |
| percussion.tambourine | 1 ✗ | **6** ✓ | FluidR3, Iowa ×3, VCSL tamb. 2, VSCO/VCSL tamb. 1 (one group: same Versilian instrument) |
| percussion.shaker | 1 ✗ | **4** ✓ | FluidR3, VCSL small + large shaker, VCSL cabasa |
| percussion.claps | **2** ✓ | 3 ✓ | FluidR3, our 808 clap (synthetic), + VCSL real claps |
| percussion.hand_drums | 1 ✗ | **5** ✓ | FluidR3, VCSL bongos / conga / darbuka, VSCO congas |

- **Today: 5 of 28 new leaves are trainable** (soprano, alto, baritone, clarinet, claps).
  That is not worth a training round; `run_v8.sh` refuses to train with fewer than 15 new
  leaves unless `ALLOW_PARTIAL=1`.
- **After approval: 24 of 28.** Not trainable fairly: banjo, mandolin, brass section,
  accordion. Their MIDI parts stay unrendered (no unlabelled audio).
- **Thin (exactly 2): viola, cello.** M4's LSO says 1-2 instruments memorize the sampler.
  NSynth `string_acoustic` (85 instruments) would fix this but needs a download and labels.
- **Labelling work that adds instruments for free:** 67 NSynth brass instruments are already
  on disk, unlabelled (trumpet / trombone / horn / tuba); 46 NSynth reeds are unlabelled.
  `nsynth_label page` now offers these families.
- Real-recorded (not GM bank) instruments are the majority for every trainable leaf except
  claps (one real, one synthetic 808, one GM bank).

## Missing sources: need Bradley's approval

None downloaded. All commercial-clean; nothing NC is proposed.

| source | what | licence | size (download) | how |
|---|---|---|---|---|
| VSCO-2 CE (sparse paths) | solo violin, 3 string sections, harp, horn, trombone, tuba, flute, piccolo, oboe, clarinet, bassoon, glock, marimba, xylophone, GM-style tambourine + congas | CC0 1.0 (asks: don't sell the samples themselves) | ~1.2 GB WAV | add paths to the existing `sources/vsco2_ce` sparse checkout (ref `SFZ`) |
| Iowa MIS 2012 pitch pages | violin/viola/cello arco, flute, oboe, Bb clarinet, bassoon, horn, tenor trombone, tuba, marimba (yarn), vibraphone, xylophone, bells; 3 tambourines | any use, no restrictions | ~0.9-1.2 GB AIFF (rough: HEAD on sampled files); ~0.4 GB after mono/trim; cache deletable | `notes_prep iowa --set v8` |
| VCSL (sparse paths) | small + large shaker, cabasa, claps, tambourines 1-2, bongos, conga, darbuka, glock, marimba, vibes, xylophone, 3 harmonicas, concert + folk harp | CC0 1.0 | ~0.45 GB WAV | sparse checkout to `~/datasets/vcsl`, then `notes_prep vcsl` |
| NSynth `string`/`flute` (optional) | 85 string + 35 flute instruments (unlabelled) | CC BY 4.0 (already attributed) | ~3.4 GB extracted; streams the ~22 GB train archive | `fetch2.sh` with `FAMS=(string flute)`, then `notes_prep nsynth --families string flute` |
| NSynth `mallet` (not recommended) | 78 mallet instruments | CC BY 4.0 | ~3.5 GB | mallets are already 3-4 instruments without it |

Not proposed: Sonatina Symphonic Orchestra (CC Sampling Plus, not in the reviewed licence
list); GeneralUser GS (custom licence, would need review); Philharmonia samples (not CC).
Licence rows are in SYSTEMS_DESIGN §4.1.

## Real held-out checks, per family (counted with the loaders, 2026-10-16)

Test-only sets, never trained on. Counts are positives / observed items (MoisesDB pooled: 720
clips, 3 windows × 240 songs, 43 artists).

| family / node | OpenMIC test (5,085, 1,709 artists) | MoisesDB pooled | golden (25 songs) |
|---|---|---|---|
| violin | 394 / 631 (149 artists) | inside `strings.bowed` (23 pos, 7 artists) | 1 |
| cello | 226 / 485 | 0 (solo cello) | 1 |
| viola | – | 0 | – |
| string section | – | 17 (5 artists) | – |
| harp | – | – | – |
| trombone | 228 / 720 | `brass` only: 9 pos, 3 artists | 1 |
| horn, tuba | – | (inside `brass`) | – |
| saxophone subtypes | sax parent 305 / 629; subtypes unobserved | – | tenor 1 |
| clarinet | 137 / 640 | `woodwinds`: 14 pos, **1 artist** | 1 |
| flute | 175 / 562 | 0 | 1 |
| oboe, bassoon | – | – | – |
| mallets | `percussion.mallet` 211 / 478; subtypes unobserved | 9 (4 artists) | marimba 2, vibes 1 |
| tambourine, shaker, claps, hand drums | – | `percussion` parent only: 104 pos, 17 artists | **shaker 13, tamb. 4, hand drums 4, claps 4** |
| accordion / banjo / mandolin | 115 / 193 / 140 | – | 1 / 1 / 0 |

- **OpenMIC test is the main real check for new families** (violin, cello, trombone,
  clarinet, flute, mallets). It is independent for v8: we never co-train on OpenMIC (only the
  temperature is fit on OpenMIC *train*).
- **MoisesDB** only sees new families at parent level with 1-17 artists. Directional evidence.
- **Percussion leaves have no public real test here.** The golden set is the only one (and it
  was already looked at for M5). MedleyDB has fine labels for oboe, bassoon, horn, tuba,
  harp, tambourine, shaker, etc., but **MedleyDB audio is not on this machine** (licence
  CC BY-NC-SA, access by request; test-only if obtained).
- Nobody labels viola, horn, tuba, oboe, bassoon, harp or sax subtypes in a real set we have:
  for those the evidence is synthetic LSO only, and the UI must keep them as "?".

## Adoption rule (pre-registered; fixed before any v8 result)

**Candidate** `v8sg`: `run_v8.sh` stage 2, i.e. the v8serve render (`--leaf-set v8`, no
holdouts, 16k clips, drive2, bandlimit 0.3, 24 kHz, seed 0) + ±12 dB gain augmentation, same
CNN, hyper-parameters and seed as v7sg. **Incumbent** `v7sg` (sha 0a1df16e1827), scored from
its existing prediction files (`preds/m6gain`) with the same harness commit. All deltas are
candidate − incumbent, paired artist bootstrap, 1,000 resamples, seed 0. Node lists are
computed mechanically by `eval.adopt`: trained nodes from each render manifest (as the ONNX
sidecar does), keeping nodes with ≥ 3 observed positives and ≥ 3 observed negatives.

Pre-training guard (not a result): every v1 leaf keeps ≥ min(0.85 × its v7serve positives,
1,000) positives in the v8 render. If not, raise `N_CLIPS` and re-render; never train on it.

**v8sg replaces v7sg only if all four gates pass:**

| gate | test set | nodes | passes if |
|---|---|---|---|
| A1 no forgetting, overall | MoisesDB pooled | trained by both, scoreable | subset mean AP Δ ≥ −2% × incumbent (the harness tolerance, `REL_TOL`) |
| A2 no forgetting, per node | MoisesDB pooled | same | no node's paired CI lies entirely below −0.01 (M5 F2 run-noise rule) |
| A3 clean vs distorted | IDMT-SMT-Audio-Effects | same | A1 + A2 on this set |
| A4 old coarse classes | OpenMIC test | same (≈ the trained-9) | A1 + A2 on this set |
| B1 new families learned | OpenMIC test | trained only by v8, scoreable (violin, cello, trombone, clarinet, flute, mallets, …) | subset mean AP beats the incumbent **and** chance (constant scores), both paired CIs' lower bound > 0 |

Reported, not gating (decided now, with reasons):
- **B2** new families on MoisesDB pooled (`--min-artists 3`): parent nodes, 3-17 artists, too
  few for a CI to mean much. Reported with CIs and direction.
- **IDMT-SMT-Bass**: bass technique is suppressed in the app and flipped 0.51 → 0.34 from gain
  augmentation alone (M5), so it can't arbitrate. Reported.
- **Synthetic test** (seen sources) and **LSO** (new leaves on held-out instruments, from the
  separate v8 LSO model; held-out pick fixed by rule: first non-GM-bank instrument, by name,
  of every new leaf with ≥ 3 instruments). LSO decides routing, not adoption.
- **Golden set percussion** (shaker 13, tambourine/claps/hand drums 4 each): AP vs chance with
  artist CIs, report only (25 songs, already used once).

**If a gate fails:** v7sg stays. No re-tuning against these test sets. Diagnosis happens on
synthetic val / LSO only; any fix is a new pre-registered round, reported next to this one.

**If adopted (swap checklist, as M5):** ONNX export to a staging dir (the script does this,
the served model is untouched) → ONNX/torch parity and app-vs-harness gap → temperature refit
on OpenMIC train → routing re-derived mechanically: a new subtype is shown only if its
normalized lift on real data ≥ 0.1 (OpenMIC where labelled) and it isn't flat on unseen
instruments (LSO); leaves no real set labels (viola, horn, tuba, oboe, bassoon, harp, sax
subtypes, percussion leaves) are listed only under the M5 rule "not yet measured = shown as
uncertain". Presence stays with PANNs unless ours beats the app-faithful PANNs (M5 presence
rule). Routing lives in `pipeline/` and `export.py`; that work waits for the agent currently
changing `pipeline/`.

## Run plan (`~/datasets/disstruments-synth/run_v8.sh`, not run)

1. Preflight: free disk ≥ 16 GB (9 with `SKIP_LSO=1`), no other training job, registry valid,
   gate logged, ≥ 15 new leaves trainable.
2. LSO model `v8` (diagnostic): render with the rule-picked holdouts → mels → train (gain
   aug) → synthetic `test` / `test_lso` runs.
3. Candidate `v8sg`: v8serve render → old-leaf guard → mels → train → predict MoisesDB pooled,
   OpenMIC, IDMT → harness runs for candidate and incumbent.
4. `eval.adopt` gates A1-A4, B1 (+ reports) → staging export only if all pass.

| step | estimate (from v7serve / m6_gain logs on this Mac) |
|---|---|
| render 16k clips (6 workers) | ~50 min per dataset (LSO dataset ~55 min) |
| mels | ~10 min per dataset |
| train 40 epochs, ~13.1k clips, MPS | ~75 min per model |
| predict + harness runs | ~10 min |
| adoption checks (1,000 resamples) | ~75 min CPU (MoisesDB 6, OpenMIC 15, IDMT-FX ~30 min each) |
| **total** | **~3.5 h** serve model only; **~6 h** with the LSO model |
| disk | ~7 GB per render + mels (≈ 14.5 GB both) on top of ~3.5 GB of new sources; 24 GB free today |

16k clips (v7: 10k) because the new roles compete for the 5 parts per clip; the guard checks
the old leaves kept their data. Trumpet will lose its stand-in data (v7 rendered every brass
part, including trombone/horn/section MIDI, as trumpet), which is why the guard floor is
min(85%, 1,000) rather than 85% alone.

## Before running (checklist)

1. Bradley approves (or trims) the downloads above.
2. Fetch: VSCO sparse paths; `notes_prep iowa --set v8`; VCSL sparse checkout +
   `notes_prep vcsl`. Listen to `audition` pages for the new sources and the FluidR3 v8
   programs (M2's audit removed several FluidR3 patches; these haven't been heard).
   Check the Iowa bells' octave by ear.
3. Delete the `status: pending_approval` lines of what was fetched; `sources --leaf-set v8`
   should show ~24 trainable.
4. Label NSynth brass (67) and the remaining reeds (46) with `nsynth_label page`; optionally
   NSynth strings/flute after their download.
5. Free GPU; then `run_v8.sh` (or `SKIP_LSO=1 run_v8.sh`).

## Decisions for Bradley

1. **Downloads**: approve VSCO-2 CE (~1.2 GB), Iowa v8 (~0.9-1.2 GB raw), VCSL (~0.45 GB)?
   NSynth strings/flute (~3.4 GB)? (Recommend: yes, yes, yes, and strings/flute only if
   viola/cello should be more than thin.)
2. **Timbales** (GM 65/66): dropped, a deviation from the brief ("bongo/conga/timbale").
   Main reason: only FluidR3 has timbale samples (VCSL hand drums are written to keys 60-64,
   the VSCO note map has no 65/66), so timbale notes inside `hand_drums` would be a
   one-instrument shortcut within the leaf. Secondary: they're played with sticks, and the
   leaf says "hand drums". Say so and it's a one-line change (`PERC_ROLE_NOTES`).
3. **808 clap** as a claps instrument: synthetic, but it's the clap most pop records use.
   Keep, or real claps only (then claps needs VCSL + one more)?
4. **Viola / cello** with 2 instruments: train them thin, wait for NSynth strings, or label
   them only at `strings.bowed` for now?
5. **Not trainable** (brass section, accordion, banjo, mandolin): fine to leave out of v8?
6. **Iowa alto/soprano** are exact leaves now. Any future `--leaf-set v1` build no longer
   uses them as coarse tenor stand-ins (reproducing v7 needs its git SHA anyway).
7. **Synth brass** (GM 62/63) still renders as acoustic brass leaves, as in v7. Keep for
   continuity, or drop (cleaner labels, less trumpet data)?
8. **Disk**: 24 GB free. Both models need ~16 GB plus sources; run `SKIP_LSO=1` first, or
   clear old renders (v1, v4lso, v5div: your call).

## Decisions applied before running (2026-10-16)

Bradley chose "Yes, use defaults". Applied before any v8 render; the adoption rule above is
unchanged.

1. **Downloads approved and fetched**: VSCO-2 CE sparse paths, Iowa MIS v8 pages and VCSL
   sparse paths. Each licence was re-checked at download time:
   - VSCO-2 CE `LICENSE`/`Readme.txt` are identical to the pinned checkout: CC0, credit
     requested, don't resell the samples. Same commit `6dd651d`, merged into `sources/vsco2_ce`.
     Every new `.sfz` sample reference resolves to a file.
   - VCSL `LICENSE` is CC0 1.0, commit `c1ea7bc`, at `~/datasets/vcsl`.
   - Iowa `MIS.html` says "used for any projects, without restrictions".
   
   The 52 `status: pending_approval` markers are removed, and the git commits are pinned in
   `sources.yaml`.
2. **NSynth strings/flute declined** (not downloaded). Viola and cello train thin, with 2
   instruments each (decision 4: accepted). There's no real calibration data for them, so
   they stay "?" in the UI.
3. **Kept out** (one source each): timbales, brass section, accordion, banjo, mandolin.
4. **808 clap kept** as a claps instrument.
5. **Synth brass (GM 62/63) dropped**: `ROLE_OF_PROGRAM` brass is now 56-61, so those parts are
   never rendered. Before, v1-v7 rendered them as trumpet. This also changes what a
   `--leaf-set v1` build parses: reproducing v7 needs its git SHA (as already noted for the
   Iowa saxes).
6. **Skipped**: NSynth brass/reed labelling and the FluidR3/new-source listening pass. In their
   place, an automated sanity render (below).
7. **`SKIP_LSO=1`** (disk). There is **no LSO model** this round, so there are no
   unseen-instrument numbers for the new leaves. Routing decisions that need LSO wait for it.

**Data-prep fix found while preparing VCSL**: the folk harp and both Hohner harmonicas name
their notes in the "C3 = middle C" convention. pYIN reads every file exactly +12.0 semitones
above its name, so the pitch check rejected all 183 files. `notes_prep.VCSL_NAME_OCTAVE_SHIFT`
(+12 for those 4 folders) fixes the names. The check itself is unchanged. Without the fix,
harmonica would have had one instrument (FluidR3) and failed the gate.

**Old-leaf guard previewed before rendering, so `N_CLIPS` = 30,000 (the pre-registered fix).**
The planner is deterministic, so I ran it alone, with no audio (same seed, leaf set and MIDI),
and counted planned parts against the guard floor min(0.85 × v7serve, 1,000):

| leaf | v7serve | floor | v8 plan at 16k | v8 plan at 28k |
|---|---|---|---|---|
| brass.trumpet | 1,813 | 1,000 | **601** (under) | 1,039 |
| woodwinds.saxophone.tenor | 606 | 515 | **451** (under) | 778 |
| every other v1 leaf | | | above floor | above floor |

- **Why trumpet fell.** v7 rendered every brass part as trumpet (trombone, horn and section
  MIDI, plus synth brass 62/63). v8 splits brass four ways, and Bradley's synth-brass drop
  removes 62/63 entirely.
- **Why tenor fell.** The four sax types and clarinet/oboe/bassoon now share the reed role.
- **Remedy.** The plan's rule is "raise `N_CLIPS`, not the threshold". Trumpet grew almost
  linearly (601 → 1,039 from 16k to 28k), so 30k gives about 1,110, roughly 10% headroom over
  the floor. v7serve lost only 13 silent stems in 10k clips between planning and rendering.
- **Decided before any v8 render.** The only change to the candidate definition is "16k clips"
  → 30k. The recipe, the gates and the incumbent are unchanged; the script's own guard still
  runs on the rendered manifest.
- **Cost.** About 6 h instead of 3.5 h. Disk: about 9.6 GB of renders plus about 3.4 GB of mels,
  against 23 GB free.
- **Still thin even at 30k** (planned parts): violin, viola and cello about 200 each, harp
  about 330, glockenspiel/marimba/xylophone about 300.

**Automated sanity render (replaces the listening pass).** 108 renders of 3 s each:
- every approved source of every gated new leaf;
- every FluidR3 program of those leaves, including each percussion kit.

All 108 are finite and audible (peak > −40 dBFS, RMS > −55 dBFS, before the renderer's loudness
normalization), with 0 unmapped or out-of-range notes.

**Prepared note counts** (`notes_prep`, after pitch checks):
- Iowa: violin 46, viola 46, cello 44, flute 39, oboe 33, clarinet 45, bassoon 37, horn 44,
  trombone 33, tuba 36, marimba 59, vibraphone 40, xylophone 42, bells 41; tambourines 1-3
  have 2 one-shots each.
- VCSL: shakers 42/18, cabasa 6, claps 10, tambourines 2+2, bongos/conga/darbuka 120/170/100,
  glock 18, marimba 20, vibes 36, xylophone 47, harmonicas 57/39, harps 37/57.
- The Iowa AIFF cache was deleted after prep, as planned.

## Results

> Run 2026-10-16 15:34-22:25 at `fe9ff6f`: `N_CLIPS=30000 SKIP_LSO=1 run_v8.sh`.
> - **Render**: 30,000 clips (train 24,670 / val 2,742 / test 2,588), 54 silent stems.
> - **Old-leaf guard**: passed (trumpet 1,123, tenor 709; nothing under the floor).
> - **Training**: 40 epochs, best synthetic val mAP 0.807, 1.22M params.
>
> All numbers come from `preds/v8adopt/*.json`: paired artist bootstrap, 1,000 resamples,
> seed 0, deltas are candidate − incumbent. Nothing was re-tuned after seeing them.

### Verdict: ADOPTION RULE FAILED, v7sg stays served

A1/A2 failed, so nothing was exported. The served model is byte-identical before and after:
sha256 `0a1df16e…` (onnx) and `acb34374…` (json).

| gate | test set | nodes | result | pass? |
|---|---|---|---|---|
| A1 no forgetting, overall | MoisesDB pooled | 21 | subset ΔmAP **−0.0227**, CI [−0.0412, +0.0090]; allowed −0.0119 (2% × incumbent 0.595) | **FAIL** |
| A2 no forgetting, per node | MoisesDB pooled | 21 | `keys.synth.lead` Δ −0.159, CI [−0.301, −0.026], entirely below −0.01 | **FAIL** |
| A3 clean vs distorted | IDMT-SMT-FX | 6 | ΔmAP **+0.073**, CI [+0.031, +0.096]; no node below the floor | pass |
| A4 old coarse classes | OpenMIC test | 9 | ΔmAP **+0.013**, CI [−0.000, +0.025]; no node below the floor (organ −0.034, CI [−0.100, +0.029]) | pass |
| B1 new families learned | OpenMIC test | 6 | vs v7sg **+0.351**, CI [+0.310, +0.387]; vs chance **+0.275**, CI [+0.243, +0.310] | pass |
| B2 (report) | MoisesDB pooled, ≥ 3 artists | 5 | vs v7sg +0.064, CI [+0.002, +0.271]; vs chance +0.056, CI [+0.020, +0.265] | (reported) |
| IDMT-Bass (report) | IDMT-SMT-Bass | 2 | ΔmAP +0.088, CI [+0.061, +0.120] (fingered +0.158) | (reported) |

**What sank A1.**
- Two nodes carry the whole shortfall:
  - `woodwinds`: AP 0.649 → 0.334, Δ −0.315, CI [−0.446, +0.039]. Only 14 positives, from
    **1 artist**.
  - `keys.synth.lead`: AP 0.325 → 0.165.
- Without `woodwinds`, the 20-node mean is −0.008, inside the tolerance. Without both, it is
  −0.000.
- Read as a diagnosis, not a re-scoring:
  - On one artist's woodwind parts, v7sg's "everything reedy is tenor sax" scores ranked
    better than v8's split across 9 woodwind leaves.
  - Synth lead most likely lost ground to the new sustained timbres it now competes with
    (harmonica, flute, oboe, bowed strings). This is unconfirmed: it needs synthetic val/LSO
    confusion analysis, not test-set probing.
- The rule was fixed before the run, and v8 fails it as written.

**Other old nodes on MoisesDB** that moved without crossing the floor:
- `guitar.electric.clean` −0.045, CI [−0.107, +0.008];
- `drums.electronic` −0.054, CI [−0.113, +0.016];
- `guitar.electric.distorted` +0.035, CI [+0.006, +0.073];
- `drums` +0.013, CI [+0.002, +0.023].

### New families on real music (vs chance = constant scores = prevalence AP)

**OpenMIC test (B1 nodes).** Every node's AP beats chance, with the CI lower bound > 0. v7sg
had never been trained on any of these nodes.

| node | positives | AP | chance (prevalence) | normalized lift | AP − chance, 95% CI |
|---|---|---|---|---|---|
| strings.bowed.violin | 394 | **0.903** | 0.624 | 0.74 | [+0.219, +0.336] |
| percussion.mallet | 211 | **0.794** | 0.441 | 0.63 | [+0.300, +0.408] |
| strings.bowed.cello | 226 | **0.723** | 0.466 | 0.48 | [+0.199, +0.312] |
| woodwinds.flute | 175 | **0.614** | 0.311 | 0.44 | [+0.193, +0.410] |
| brass.trombone | 228 | **0.552** | 0.317 | 0.34 | [+0.165, +0.310] |
| woodwinds.clarinet | 137 | **0.437** | 0.214 | 0.28 | [+0.162, +0.301] |

All six clear the M5 routing bar (lift ≥ 0.1) on real data.

**MoisesDB pooled (B2, ≥ 3 positive artists).** Detection is real but weak. AP is above
chance, but absolute AP is low:

| node | positives | AP | chance | AP − chance, CI |
|---|---|---|---|---|
| percussion | 104 | 0.203 | 0.164 | [+0.006, +0.149] |
| percussion.mallet | 9 | 0.054 | 0.014 | [+0.001, +0.093] |
| strings / strings.bowed | 23 | 0.101 | 0.032 | [+0.010, +0.346] |
| strings.bowed.section | 17 | 0.088 | 0.024 | [+0.004, +0.493] |

**No real test set scores these nodes**, so there is no real-music evidence for them:
- viola, French horn, tuba, oboe, bassoon, harp, harmonica;
- the 3 new sax subtypes;
- the 4 mallet subtypes;
- tambourine, shaker, claps, hand drums.

They exist only in the 25-song golden set. The plan keeps that set report-only, and this
round didn't score it.

**Synthetic test** (seen sources; report only, not generalization): mAP 0.803. New-leaf AP:
- above 0.9: section, harmonica, tambourine, shaker, claps, hand drums, glockenspiel, flute;
- 0.6-0.9: violin, viola, cello, trombone, tuba, alto sax, oboe, vibraphone;
- below 0.6: harp 0.37, horn 0.43, soprano 0.53, baritone 0.30, clarinet 0.23, bassoon 0.41,
  marimba 0.36, xylophone 0.57.

**Not run: LSO** (`SKIP_LSO=1`). There are no unseen-instrument numbers, so it is unknown
whether the thin leaves (viola/cello with 2 instruments) learned the instrument or the
sampler.

### What this means

- **v8 is a real advance on the new families**: six OpenMIC families, from nothing to strong.
  It is also better on distorted/clean guitar and bass (IDMT).
- **But it forgets** one artist's woodwinds and synth lead on MoisesDB beyond the pre-set
  tolerance. By the rule, v7sg stays.
- **Any fix is a new pre-registered round**, diagnosed on synthetic val/LSO only. Candidates:
  - keep a coarse `woodwinds` output trained on all reeds;
  - balance synth-lead data against the new sustained families;
  - run the LSO model when disk allows.
- **Artifacts** (all research-only, outside the repo): `preds/v8serve/scratch_cnn_synthetic_v8sg.pt`;
  eval runs `eval_runs/20261017T0339*`; adoption JSONs `preds/v8adopt/`; render `v8serve/`
  (about 10.5 GB) plus mels `emb/mels/synthetic_v8s_*`.
