# Synthetic dataset v2: deleted 2026-10-09; how to regenerate it exactly

**What it was.** 100,000 rendered 10 s clips (82,127 train / 9,210 val / 8,663 test,
artist-disjoint), taxonomy 3.0.0, 24 kHz mono FLAC; stem audio kept for 20% of clips.
48 GB on disk. It was used for the M3 data-scaling curve (`docs/debriefs/m3.md`) and then
deleted to free disk. Its result (more of the same data doesn't help real music) doesn't
need the audio. Models trained on it are kept in
`~/datasets/disstruments-synth/preds/m3scale/`.

**Exact inputs (the renderer is deterministic given these):**
- **Code:** git commit `2eb1524` (`backend/disstruments/ml/synth/`; nothing under it changed
  before the deletion)
- **Sources:** `backend/disstruments/ml/synth/sources.yaml` at that commit, sha256
  `eb3096167e440229e48c3f37cd09113ddd255e36dae5d04c65fd148108ffccf8`. Sample files are
  re-fetchable from each entry's `download` URL + sha256.
- **SFZ key-range cache** used for octave fitting: `docs/data/synth_v2_key_ranges.json`.
  Copy it to `~/datasets/disstruments-synth/sources/.key_ranges.json` before rebuilding
  so octave shifts match.
- **MIDI:** Lakh MIDI `clean_midi.tar.gz`, sha256
  `de1bb64cbc0cf35545a05b5c3e786aa6890cfa144edffc4b827ff41bf8c33dc5`
  (http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz).
- **Engines:** sfizz_render built from sfizz `f5c6e29`; FluidSynth 2.6.1 (Homebrew);
  Surge XT 1.3.4 and Dexed 1.0.1 VST3 (Dexed unused after the audit); pedalboard 0.9.25.
- **Manifest:** `docs/data/synth_v2_manifest.json` (per-leaf/per-source counts, timing).

**Command:**
```
git checkout 2eb1524 -- backend/disstruments/ml/synth   # or check out the commit
cp docs/data/synth_v2_key_ranges.json ~/datasets/disstruments-synth/sources/.key_ranges.json  # expand ~
python -m disstruments.ml.synth build --midi-root ~/datasets/disstruments-synth/midi/clean_midi \
  --out ~/datasets/disstruments-synth/v2 --n-clips 100000 --workers 7 --seed 1 --sr 24000 \
  --windows-per-file 8 --stem-fraction 0.2
```
About 2.5 h on the M5 (0.089 s/clip with 7 workers) plus ~11 min of planning.
**Caveat:** plugin and library versions (Surge, sfizz, pedalboard, numpy/scipy) can change
DSP output by tiny amounts. Labels and recipes reproduce exactly; audio is
near-identical, not guaranteed bit-identical across tool versions.
