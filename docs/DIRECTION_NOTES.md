# Direction Notes: "Remake your favorite song" (not yet a PRD amendment)

*Bradley, 2026-10-06. This informs design choices. `PRD.md` stays the scope authority until this is formally amended.*

## The idea
Instrumake/Disstruments eventually becomes a **music-education product**: learn production by remaking songs you love.

**The loop:**
1. **Analyze** a song.
2. **Show a recipe:**
   - leaf-level instruments ("Rhodes", "palm-muted Rickenbacker bass"), with when each plays;
   - sound character;
   - key, BPM and chords;
   - MIDI per part.
3. **The user remakes it** in their own DAW.
4. **They upload the remake** and get **scored per part** on how close it is to the original (instrument choice, timbre, notes, timing, mix balance).

## Where the moat is
- **Commodity (others do it already):** stem separation, key/BPM and chords. Moises and similar apps already ship these. We use them; we don't compete on them.
- **Differentiator #1, leaf-level instrument ID.** This is the current ML track (PRD Amendment A1, `ML_ENGINEERING.md`).
- **Differentiator #2, later: "how was this sound made."** Synth family, preset or parameters, and the effects chain (PRD NG1 moonshot). It needs training data where the recipe behind each sound is known.

## What this changes now (no new features)
- **Keep full provenance on every synthetic clip.** Store the taxonomy leaf, source library or synth, preset name, synth parameters, effects chain, MIDI source, render seed and loudness, even while M2 trains only on instrument labels. Then "sound match" and "how was it made" models can train later on data we already have.
- **Prefer sound sources that expose parameters.** Open-source synths rendered programmatically (Surge XT, Dexed via pedalboard/DawDreamer) give exact parameter ground truth; one-shot sample libraries don't.
- **Design the eval harness for reuse.** Per-part similarity scoring ("your remake vs the original") is the same machinery as model evaluation: per-stem, per-leaf, calibrated. Keep the harness general enough to score a remake as well as a model.
- **MIDI per part (F11) moves up in value.** It's a recipe ingredient and the "notes/timing" half of remake scoring.

## Licence constraint (Bradley, 2026-10-07)
The end state is a commercial product or a free public one; either way it must be
**licence-clean**. That means:
- **Permissive backbones only** (CLAP/BEATs/PaSST, not MERT).
- **Training data** limited to CC0 / CC-BY / MIT sources and procedural or licensed MIDI.
- **NC datasets** (MedleyDB, jRhodes3d) are for evaluation and research only.

Every synthetic clip carries licence ids, so the clean subset can be filtered out at any
time.

## Open questions (decide before any amendment)
- How is remake similarity scored? Embedding distance (e.g. MERT) per stem, symbolic note F1 from MIDI, or both, and how is that shown to a learner?
- Who is the user: hobbyist producers, music-school students, or both? This decides pricing, curriculum and genre coverage (rock/indie/pop first).
- Legal: the recipe and scoring use copyrighted songs. The current posture (stems only ever go back to the uploader, PRD §9) must hold for remakes too.
