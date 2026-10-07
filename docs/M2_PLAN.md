# M2 Plan: Synthetic Data Pipeline v1 + First Baseline (PLAN ONLY, awaiting approval)

*Drafted 2026-10-06. No M2 code until Bradley approves. Budget: ≤ $20 compute (approved).
Licences verified against the source pages on 2026-10-06; URLs are in §6.*

## 1. Goal and deliverables
- **Render pipeline v1:** MIDI is rendered through instrument sources, with randomized
  effects and mastering, into labelled 10 s clips. Target ~10k clips across ~20 leaves
  (ML_ENGINEERING §8).
- **Full provenance per clip** (DIRECTION_NOTES), even though M2 trains only on instrument
  labels.
- **First baseline:** a frozen pretrained audio backbone, cached embeddings, and a linear
  classifier per taxonomy node.
- **Eval report** through the M1 harness:
  - per-leaf AP and F1;
  - the primary metric with an artist-level bootstrap CI;
  - ECE (raw; temperature scaling is M3);
  - results on a synthetic held-out set, OpenMIC, and the MedleyDB **split v2** test set.

## 2. Stages

| # | Stage | What gets built | Cost | Time |
|---|---|---|---|---|
| S0 | Licence + source registry | `ml/synth/sources.yaml`: every source with leaf, engine, patch/sample, licence, URL, and the commercial-OK flag. Licences go to SYSTEMS_DESIGN §4. | $0 | 0.5 day |
| S1 | Downloads + engines | Fetch the chosen libraries (~5–12 GB, outside the repo, sha256-pinned like `fetch_weights.sh`). Install `pedalboard` (VST3 host + effects), `sfizz` (SFZ), `fluidsynth` (SF2), and Surge XT / Dexed / Helm VST3s. | $0 | 0.5–1 day |
| S2 | MIDI selection | Pick Lakh MIDI files (CC BY 4.0 compilation). Assign *our* leaves to tracks by role (bass line → a bass leaf, etc.), cut 10 s windows by note density, and split **by MIDI file** so the same composition never lands in train and test. | $0 | 1 day |
| S3 | Renderer | Render each stem with the chosen source and randomized velocity, articulation and patch params. Per stem: EQ, compression, reverb IR, and amp/distortion for guitars. Mix 1–5 stems with random levels and panning, then a mastering chain (bus compressor, limiter), then loudness normalization. Emit mix + stems + `labels.json`. Deterministic per seed; parallel across CPU cores. | $0 (local) | 3–4 days |
| S4 | Dataset loader | `ml/datasets/synthetic.py` follows the existing `Record` contract. Labels are **perfect and fully observed**: present instruments are positive and every other node is a true negative. Builds a per-clip metadata index. | $0 | 1 day |
| S5 | Embedding cache | Run the frozen backbone on mixes and stems at 24 kHz. Cache layer-wise pooled embeddings to disk (float16, one file per clip). Same for OpenMIC and MedleyDB clips. | $0 on the M5 (MPS); ≤ $5 if the cloud fallback is needed | 1 day |
| S6 | Linear probe | Logistic regression per taxonomy node on (learned or fixed) layer-weighted embeddings, numpy/torch. Max-propagation at inference (§4.3). Class-balanced loss. Train on synthetic only (v1). | $0 | 1–2 days |
| S7 | Evaluation + write-up | Harness runs on: synthetic test (MIDI-disjoint), OpenMIC (20-class projection, available now), MedleyDB split v2 test (**needs audio access**), the prior baseline, and the PANNs incumbent where comparable. Debrief `docs/debriefs/m2.md`. | $0 | 1–2 days |

**Total: about 2–3 weeks of sessions, ~$0–5 of compute.** Everything runs on the M5; I'll
check with you before any cloud spend, and no stage gets near $20.

**Disk:** sources ~5–12 GB, plus renders of 10k clips × (mix + ≤5 stems) as 24 kHz mono FLAC
(~10–15 GB), plus the embedding cache (~2–6 GB depending on backbone). All of it lives
outside the repo under `~/datasets/disstruments-synth/`.

### Per-clip metadata (`labels.json`), stored now, used later
- clip id, render seed, renderer git SHA, taxonomy version
- MIDI source file (Lakh md5), window start/end, tempo, key (if known)
- for each stem:
  - taxonomy leaf and source id (library/synth + version)
  - preset or patch name, and **full synth parameter dict** (VST3 parameter values from
    pedalboard, when the source is a synth)
  - articulation and velocity jitter
  - effects chain: ordered list of (plugin, params) plus the reverb IR id
  - integrated loudness (LUFS) and gain in the mix
- mix: pan/levels, mastering chain + params, final LUFS / true peak
- licence ids of every source used (so any clip can be filtered to "commercial-clean")

## 3. Decisions I need from you

> **2026-10-07: Bradley approved D2–D8 as recommended. D1: run MERT-95M and CLAP side by
> side.** Goal stated: eventually a commercial product, or a free public product. So the
> **shippable path must be licence-clean**:
> - CLAP (or another permissive backbone) is the production candidate; MERT is the research
>   comparison only.
> - Anything trained on NC data (MedleyDB audio, jRhodes3d) or on Lakh-MIDI renders is
>   research-only. A shippable model retrains on `commercial_clean` clips rendered from
>   procedural or licensed MIDI.
> - "Free" doesn't automatically mean non-commercial under CC-BY-NC (ads or a later paid
>   tier would break it), so treat any public release as commercial.

| # | Fork | Options | My recommendation |
|---|---|---|---|
| **D1** | **Backbone (licence!)** | (a) MERT-95M (CC-BY-**NC**-4.0), (b) MERT-330M (same NC licence, ~3× slower), (c) a commercially clean model: LAION-CLAP (CC0 / Apache-2.0), BEATs (MIT), PaSST (Apache-2.0) | **Run MERT-95M *and* CLAP in the same table.** MERT is the plan and fine for a portfolio, but its NC licence means it can never ship in a paid "remake" product. Having a clean backbone in the ablation from day one keeps that path open and makes a strong portfolio point. MERT-330M waits for M3. |
| **D2** | MIDI source | (a) Lakh MIDI (real song structure; CC BY compilation, but the compositions are real songs, so not cleared for a commercial product), (b) procedurally generated MIDI (clean, less musical), (c) both | **(a) for M2.** Each clip records its MIDI source, so a later commercial dataset can filter or swap it. Add (b) in M4 as a domain-randomization lever. |
| **D3** | The ~20 leaves for v1 | See §4 | Approve or edit the list in §4. |
| **D4** | Sources | Pick from the shortlist in §5 | Tier 1 in §5 (all CC0/MIT/CC-BY; commercial-OK). |
| **D5** | Rhodes / Wurlitzer | No commercially clean free *sample* source exists. (a) model with FM/synth (Dexed with our own patches, Surge XT), (b) jRhodes3d samples (CC-BY-**NC**), (c) skip in v1 | **(a), and add (b) tagged NC-only.** It's a clean ablation: does a synth-modelled Rhodes generalize to real Rhodes? |
| **D6** | Training unit | (a) isolated stems only, (b) mixes only, (c) both | **(c).** The app tags Demucs stems *and* the full mix. Separation-artifact augmentation stays in M4. |
| **D7** | Real eval set while MedleyDB is pending | (a) wait for MedleyDB, (b) report OpenMIC (available now; 2.6 GB, CC BY 4.0) and add MedleyDB v2 when access lands | **(b).** S0–S6 don't need MedleyDB, so its access is only on the critical path for the final S7 numbers. |
| **D8** | Synth presets | (a) factory presets (several unverified licences), (b) our own / procedurally randomized patches, plus CC0 Surge templates and CC-BY Helm presets | **(b).** Clean licence, *and* exact parameter ground truth for the future "how was it made" work. |

## 4. Proposed v1 leaf set (20, rock/indie/pop first)
- **guitar:** `acoustic.nylon`, `acoustic.steel`, `electric.clean`, `electric.distorted`
- **bass:** `electric.picked`, `electric.fingered`, `upright`, `synth.analog`, `synth.sub_808`
- **keys:** `piano`, `electric_piano.rhodes`, `electric_piano.fm`, `organ.drawbar`, `synth.pad`, `synth.lead`, `synth.pluck`
- **drums:** `acoustic_kit`, `electronic.tr808`
- **woodwinds:** `saxophone.tenor`
- **brass:** `trumpet`

Deferred to M3, along with the other 45 leaves:
- `bass.electric.slap`: no verified permissive slap source; the Rickenbacker slap set is CC-BY-NC-SA.
- Wurlitzer: no clean source.
- `tr909`

Vocals aren't rendered (no MIDI → voice source).

## 5. Source shortlist (pick one per row, or approve Tier 1)

**Tier 1: all commercial-OK** (CC0 / MIT / CC-BY / GPL-engine-output)

| Leaf(s) | Source | Licence | Size |
|---|---|---|---|
| electric guitar clean/distorted | FreePats FSBS electric guitars (clean ×3, distorted ×2) + Karoryfer Black And Green / Shinyguitar / Emilyguitar; amp/distortion via pedalboard | CC0 | 0.1–2 GB |
| acoustic nylon / steel | FreePats nylon (CC0), FreePats steel (GPL-3+ with free-use exception), U. Iowa MIS guitar | CC0 / GPL+exc / "any use" | < 1 GB |
| bass picked / fingered / upright | FreePats Yamaha RBX (picked); Karoryfer Growlybass, Swagbass, Fashionbass, Black&Blue (fingered); Karoryfer Meatbass/Sneakybass (upright) | CC0 | 1–4 GB |
| synth bass / pad / lead / pluck / 808 | Surge XT (GPL engine, CC0 templates, **our own patches**), Helm presets (CC BY 4.0) | output is ours (Surge FAQ / GPL FAQ) | small |
| piano | Salamander Grand (CC BY 3.0), VCSL pianos (CC0) | CC BY / CC0 | 0.4–1.2 GB |
| FM e-piano / Rhodes (modelled) | Dexed with our own patches, VCSL TX81Z, FreePats FM piano | GPL output / CC0 | small |
| drawbar organ | FreePats setBfree renders (CC0) or setBfree directly | CC0 / GPL output | small |
| acoustic kit | Karoryfer Big Rusty / Unruly / Virtuosity (CC0), SM Drums (public domain), Muldjord / DRS / Naked (CC BY 4.0) | CC0 / PD / CC BY | 0.4–2.2 GB each |
| 808 kit | Surge XT synthesized; optionally CC0 Freesound files | output ours / CC0 | small |
| tenor sax, trumpet | Karoryfer Weresax, VCSL, FreePats (sax); VSCO-2 CE, U. Iowa MIS (trumpet) | CC0 / "any use" | 1–3 GB |
| GM fallback | FluidR3_GM, MuseScore_General (both MIT) | MIT | 40–200 MB |

**Tier 2: non-commercial only.** Allowed for research and tagged NC in the metadata:
- jRhodes3d Rhodes (CC-BY-NC);
- NSynth (CC BY 4.0, but isolated notes; useful for pretraining, not mixes).

**Excluded:**
- **Spitfire LABS / Splice:** their EULA *explicitly bans AI training*.
- **Pianobook:** personal-use only.
- **DiViNe Wurlitzer:** EULA bans redistribution and "sample banks".
- **Free 808/909 packs:** no published terms.
- **Factory presets with unverified licences:** Surge non-template patches, Dexed cartridges, Zyn banks, Vital presets. The synths themselves are fine with our own patches.
- **Philharmonia and Sonatina:** usable inside mixes, but **never share isolated-note renders** from them ("as is" / whole-work clauses).

## 6. Risks

| Risk | Why it matters | Mitigation in M2 |
|---|---|---|
| **Synthetic → real domain gap** | Synthetic clips are cleaner and more uniform than real records, so they can score well on synthetic and poorly on real. | Randomized effects and mastering chains from day one. Always report synthetic *and* real (OpenMIC/MedleyDB) numbers side by side: the gap *is* a headline metric (§5 of ML_ENGINEERING). |
| **Label noise from source → leaf mapping** | E.g. a GM "Electric Piano 1" patch that is really FM, not Rhodes; a "clean" guitar sample with drive. | `sources.yaml` maps each patch to a leaf **explicitly**, plus a listening audit of ~5 renders per source before full render (you, ~20 min). GM fallback is only used for non-target accompaniment. |
| **Stem bleed at inference** | The app tags Demucs stems, which contain bleed and artifacts that clean training stems don't. | Train on mixes and stems (D6). Measure stem-level eval on real MedleyDB stems. Demucs round-trip augmentation is M4. |
| **Few real eval examples** | MedleyDB v2 test has 31 leaves with ≥1 positive but only 17 with ≥3, and the v1 leaf set overlaps only partly. | Primary metric on leaves with ≥3 positives (`map_leaf_min3`). Artist-level bootstrap CI. Per-leaf counts printed beside every number. |
| **MedleyDB access delay** | Blocks the real-data half of S7. | D7: OpenMIC now, MedleyDB when it lands; nothing else waits on it. |
| **Licence debt** | Lakh compositions and MERT weights are NC for a product; some GM banks admit unknown sample origins. | Every clip carries licence ids; a `commercial_clean` filter exists from day one; CLAP baseline alongside MERT (D1). |
| **Disk / time** | ~20–30 GB total; rendering speed of VST3 synths via pedalboard is unmeasured. | Measure render speed on 100 clips first (S3 gate); downsize to 5k clips if needed. |

## 7. Licence facts (verified 2026-10-06)
- **MERT-v1-95M / 330M:** cc-by-nc-4.0 (huggingface.co/m-a-p/MERT-v1-330M, /MERT-v1-95M).
- **LAION-CLAP:** CC0 (github.com/LAION-AI/CLAP); `laion/larger_clap_music` is Apache-2.0.
- **BEATs:** MIT. **PaSST:** Apache-2.0. **PANNs:** MIT code, CC BY 4.0 weights.
- **GPL synths/hosts:** the output isn't covered by the code's copyright (GNU GPL FAQ; Surge FAQ: "The output of Surge XT is yours").
- **Spitfire EULA §10, Splice Terms §3.1.1.3 / §5.3.3:** explicit AI-training bans.
- **Freesound:** per-file CC; uploader "No generative AI" preferences exist since July 2026 (they target generative models).
- **Lakh MIDI:** CC BY 4.0 compilation; per-file attribution "not feasible" per its page.
- Full table with URLs: research notes from 2026-10-06, to be copied into SYSTEMS_DESIGN §4 at S0.
