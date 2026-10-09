# M4 Plan: Close the Synthetic→Real Gap with Diversity (started 2026-10-09)

M3's scaling curve showed that real-music performance is capped by **distribution, not
volume** (8k → 82k clips: real OpenMIC flat at ~0.785). M4 attacks the distribution.
Each lever is an ablation scored against the M3 from-scratch baseline (v1, 8k clips) on
OpenMIC test with artist-bootstrap CIs. When MedleyDB lands, everything is re-scored on
real *fine-grained* labels.

| # | Lever | What | Cost |
|---|---|---|---|
| L1 | **Real co-training** | Add OpenMIC *train* (14,915 real clips, coarse labels, masked) to the synthetic training set. The hierarchy lets coarse real labels regularize the fine heads (ML_ENGINEERING §5 lever 2). | $0, ~2 h |
| L2 | **Production randomization v2** | Re-render 10k clips with stronger, licence-clean randomization: synthetic room IRs (convolution, random RT60/early reflections), amp-style drive + speaker-cab EQ on guitars/bass, EQ tilt, saturation, **lossy-codec simulation (MP3 at random bitrates, since OpenMIC/FMA clips are MP3s)**, bandwidth limits, noise floor. | $0, ~1 h render |
| L3 | **Source diversity** | More instruments for the thin, weak leaves: basses (Karoryfer Growlybass/Swagbass/Black&Blue), guitars (Black&Green, Shinyguitar, FSBS dist1/direct), drums (Big Rusty), piano (Salamander), plus more varied Surge synth recipes (wavetable/FM/supersaw). All CC0/CC-BY. | $0; ~4 GB of downloads ≈ 4 h at current bandwidth |
| L4 | Separation-artifact augmentation | Demucs round-trip on a fraction of training mixes, for stem-level inputs | Deferred until MedleyDB (stem eval) is available |

**Reporting rules:**
- Always show the trained-9 / untrained-11 split.
- **Watch for "overfitting to OpenMIC":** L1 trains on OpenMIC train, so OpenMIC test gains
  partly reflect in-domain learning. The licence-safe, fine-grained check is MedleyDB.
- Licences: OpenMIC clips carry per-clip FMA licences (some NC/ND). Models trained with
  L1 are research-only unless filtered to CC0/CC-BY clips (`openmic-2018-metadata.csv`
  `license_title`).
