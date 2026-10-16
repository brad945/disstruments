# Proposed Amendment A3: Instrumake (DRAFT, not approved)

**Status:** draft for Bradley, written 2026-10-16. Nothing here is built until Bradley
approves it. Once approved, it moves into `PRD.md` as Amendment A3 (scope law, §13).

## The idea in one line

Disstruments takes a song apart and names what's in it. **Instrumake** turns that
breakdown into a starter kit you open in your DAW (FL Studio, MuseScore), so you can
rebuild the song, remix it, or improvise over it.

## Why (Bradley's words, paraphrased)

A musician hears a song and wants "90%, ideally 100%" of what makes it up: the specific
instruments and sounds, the notes, key, BPM and time signature. Then they want to click the
notes into a piano roll and play along or rework it (e.g. a remix or parody).

## What already exists elsewhere (so we don't pretend it's new)

Stems: Demucs, AudioShake, Moises. Notes: Fadr, Suno Studio, Basic Pitch. Note-level
editing: RipX. Sound to synth patch: Synplant 2. The gap Instrumake fills: **which exact
instrument and sound each part is, with honest confidence**, attached to the stems and
notes, in one export. See ML_ENGINEERING §10.

## Proposed features

- **F35 (M): Rebuild kit export.** One download per song:
  - stems (only for the uploader, NG7 unchanged);
  - one MIDI track per pitched part (bass, lead, keys, vocal melody), each named after its
    identified instrument with a matching General MIDI sound pre-set, so it plays
    immediately in FL Studio or any DAW;
  - tempo, key and time signature written into the MIDI file;
  - an instrument sheet: each part, what we think it is, the confidence, and "?" where unsure
    (F15 rules apply).
  - Depends on F11 (MIDI transcription), pulled forward from draft 3, and F8 time signature.
- **F36 (S): MuseScore export.** The same parts as MusicXML (sheet music) for MuseScore.
- **F37 (S): Sound suggestions.** For each identified instrument, a short list of where to
  get that sound: the General MIDI program, plus free instruments or presets that match
  (e.g. Rhodes: GM 5, plus named free Rhodes plugins). We only link or recommend; we never
  ship third-party presets.
- **F38 (S): When unsure, offer both.** If the model can't decide (alto vs tenor sax,
  Rhodes vs Wurlitzer), the kit lists both candidates with a short sample of each, instead
  of guessing. This is Bradley's "just pick one or let me choose" idea.
- **F39 (C, research track): Sound matching.** Given a separated part, find the closest
  sound in a free or owned sound library (or synth settings). It narrows NG1 (preset
  inference) to a measured research track: no shipping without an eval.
- **F40 (C, research track): Generate a fitting sound.** When no library sound matches,
  generate one (timbre transfer, e.g. DDSP-style). Research only, after F39.

## What stays out

- **NG2 (YouTube/Spotify links)** stays out. It's a separate legal decision; upload only.
- **NG4 (no DAW, no editing)** stays. Instrumake *exports to* your DAW; it doesn't become one.
- **NG7 (no redistribution)** stays: stems and kits go only to the uploader.

## How we'd know it works (eval, before shipping)

- **MIDI notes:** note accuracy measured on real multitracks with known notes, using
  licence-checked data, with the harness and bootstrap CIs as today.
- **Instrument sheet:** the same golden-set scoring we already have.
- **The real test:** Bradley rebuilds 3 golden songs in FL Studio from the kit and
  scores, per part, "usable as is / needed fixing / useless". Logged like the golden set.

## Licences to check before building

- Basic Pitch (Spotify) for MIDI transcription: believed Apache-2.0; verify and record in
  SYSTEMS_DESIGN §4.
- Any sound library we recommend: link only, and no redistribution.

## Order

1. **I1:** kit export with stems, MIDI per part, tempo/key/time signature, and the
   instrument sheet (F35).
2. **I2:** MuseScore export, sound suggestions, "offer both" (F36–F38).
3. **I3:** research track: sound matching, then generation (F39–F40).

## Questions for Bradley

1. Which DAW first: FL Studio (MIDI export, recommended, works everywhere) or MuseScore
   (sheet music)?
2. "Vocal type": what do you want shown? E.g. male/female/range, singing vs rap vs spoken,
   falsetto. The male/female split was removed in taxonomy v2 pending this decision.
3. Approve I1 now, or wait until the accuracy fixes and golden-2 are done?
