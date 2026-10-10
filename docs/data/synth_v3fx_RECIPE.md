# Synthetic dataset v3fx: deleted 2026-10-09; regeneration recipe

**What it was.** 9,999 clips at 24 kHz, no stem audio, **fx profile v2** (M4 lever L2:
convolution rooms, amp/cab EQ, saturation, MP3 round trips, noise), same seed/windows as
v1. Result: no effect on real OpenMIC; also the leave-production-out test set
(`docs/debriefs/m4.md`). The trained model is kept: `preds/m4/scratch_cnn_synthetic_v3fx.pt`.

**Inputs:**
- **Code:** commit `34beb04`. `ml/synth/` is identical to `4015524` except `sources.yaml`,
  which wasn't used.
- **Sources:** `docs/data/synth_v3fx_sources_pinned.yaml` (the 29-source registry at
  `4015524`, pinned so concurrent source additions couldn't leak in).
- **MIDI:** Lakh `clean_midi.tar.gz`, sha256 `de1bb64c…3dc5` (see `synth_v2_RECIPE.md`).

**Command:**
```
python -m disstruments.ml.synth build --midi-root ~/datasets/disstruments-synth/midi/clean_midi \
  --out ~/datasets/disstruments-synth/v3fx --n-clips 10000 --workers 6 --seed 0 --sr 24000 \
  --stem-fraction 0.0 --fx-profile v2 --registry docs/data/synth_v3fx_sources_pinned.yaml
```
About 20 min render + 4 min planning on the M5.
