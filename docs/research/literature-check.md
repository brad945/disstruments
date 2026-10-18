# Disstruments literature check (2026-10-18)

Status: COMPLETE (~45 web searches/fetches + read-only cross-check of repo debriefs m3/m5/v8/golden).
Citations marked (U) = not opened / venue unverified; confirm before citing.

## Verdict table

| # | Our claim (as actually measured in debriefs) | Verdict | Closest prior work |
|---|---|---|---|
| 1 | All backbones trained on SYNTHETIC only; 1.22M scratch CNN ties frozen MERT-95M, MERT-MLP and LoRA-MERT (and beats CLAP) on real OpenMIC trained-9 (0.785 vs 0.774, CI [-0.009,+0.027]) | NEW in this precise form (foundation features don't close the sim2real gap). Synthetic-only -> real instrument ID per se is PARTLY KNOWN | Hung et al. ICASSP 2019 (MuseScore synth, mostly qualitative real eval); Ostermann et al. AAM 2023 (case study, real-eval unverified); Tanaka et al. ISMIR 2023 LBD (Slakh pretrain + real FT); Persian MERT-on-synthetic-mixes arXiv 2511.05717; Ding & Lerch ISMIR 2023; MARBLE |
| 2a | 10x same-source synthetic data: real flat (0.784->0.785) while synthetic val rises (0.948->0.979); source diversity (NSynth/Iowa) ~2x unseen nylon/steel AP | PARTLY KNOWN (well established in drum transcription / separation; not shown for instrument ID) | Zehren et al. arXiv 2407.19823 (ADT); arXiv 2601.09520 (ADT, SoundFont diversity); Jeon et al. MERL 2402.18407 / TR2026-012 (random-mix plateau) |
| 2b | Per-CLIP +-12 dB gain aug fixes a learned level prior (loudness-normalised renders; AP slope ~0.04/-3 dB on MoisesDB); leaf mAP 0.41->0.49 | Technique KNOWN (standard aug; PCEN as invariant front end); the DIAGNOSIS (level shortcut induced by loudness-normalised synthetic renders, measured as AP-vs-dB on real multitracks) appears NEW | Jeon & Lee ISMIR 2022 (loud commercial masters break separation); McFee et al. MUDA ISMIR 2015; Lostanlen et al. SPL 2019 (PCEN); Geirhos et al. NMI 2020 (shortcut) (U); Sturm 2014 (horse) |
| 3 | Licence-clean model (procedural MIDI, commercial-only samples) costs 0.490->0.471 MoisesDB leaf mAP | NEW as a measured ablation for an MIR classifier. Procedural/algorithmic-composition data itself is KNOWN (AAM) | Stable Audio Open (CC-only, cost unquantified); AAM (algorithmic composition + real samples); Morreale et al. ISMIR 2023 (U authors) data-collection critique; Data Provenance Initiative; procedural guitar arXiv 2508.07987 |
| 4 | Calibration transfer | PARTLY KNOWN (general ML: calibration degrades under shift). Music-specific measurements are new datapoints but modest | Ovadia et al. NeurIPS 2019; Lukashevich et al. Audio Mostly 2023; Yu et al. NeurIPS 2022 multi-domain TS; Alexandari ICML 2020 / LaSCal NeurIPS 2024 (label shift); bird calibration arXiv 2511.08261; TCP_alpha arXiv 2608.20326 |
| 5 | Pre-registered adoption gates, test-reuse ledger, artist-paired bootstrap, musician golden set with negatives + "type not sure" + lenient confusables | Elements KNOWN; the package (esp. pre-registration with a gate that actually failed) is NEW to MIR | Flexer ISMIR 2007 / Pampalk 2005 (artist filter); Sturm TMM 2014; Gouyon/Sturm et al. 2014; Sturm & Flexer arXiv 2301.01578; Urbano et al. JIIS 2013; OpenMIC (already has explicit negatives); pre-registration for predictive modelling arXiv 2311.18807; NeurIPS pre-reg workshops |
| 6 | Ties Essentia MTG-Jamendo instrument model (0.573 vs 0.566, 14 shared nodes) | Not a contribution; external sanity check | Bogdanov et al. MTG-Jamendo ML4MD 2019 (U); Essentia models CC BY-NC-SA |
| 7 | Adding strings/brass/woodwind/mallets + 9-way split caused forgetting (synth lead AP 0.325->0.165) | PARTLY KNOWN / confounded; contrasts with hierarchy-helps-coarse papers. Not true CIL | Sechet et al. DAFx 2024 (hierarchy -> more reliable coarse); Zhong et al. ICASSP 2023 (joint fine+coarse helps); Flores Garcia ISMIR 2021; audio CIL arXiv 2609.11447 |
| - | Fine-grained subtype ID in FULL MIXES (Rhodes/Wurli/organ, synth lead/pad/bass, 808 vs kit, slap/upright/picked, distorted/clean) + MoisesDB stem-derived leaf tagging protocol | NEW task/benchmark framing (no prior found) | Guitar FX: Comunita et al. JAES 2021, Hinrichs DAFx 2020 (isolated guitar); Chen/Su/Yang ISMIR 2015; Wu & Lerch ISMIR 2016; LALM hallucination: arXiv 2604.19300, 2609.20195 |

## Claim 4 — must be restated (from repo debriefs)
- (a) Temperature fit on synthetic val (T~1) is useless on real music (MERT ~3x overconfident; real T=2.94). Clean NEW-ish datapoint.
- (b) Per-class Platt for PANNs fit on OpenMIC train DID partially transfer to MoisesDB full songs: ECE voice 0.56->0.11, drums 0.74->0.26, guitar 0.71->0.26; precision >=0.90; residual under-confidence; piano weak.
- (c) Organ (~0.03 scores on organ songs) is a RANKING failure (PANNs organ AP at chance on golden set), which no monotone calibration can fix. Don't call it calibration failure.
- (d) Procedural (licence-clean) model is more overconfident on real music (T 3.75 vs 2.03) — a nice link between claims 3 and 4.

## Claim 3 caveats
- No pooled paired CI for 0.490 vs 0.471 in m5.md (only per-class CIs). Compute one.
- Confounded: composition change (procedural vs Lakh) + dropped jRhodes3d (CC BY-NC) source. Synthetic tests overstate cost (0.14 on Lakh-composed test vs 0.02 real) — itself a nice finding.
- AAM (Ostermann 2023) already offers algorithmically composed, freely licensed multitracks; cite as the procedural precedent.

## AAM check (claim 1 novelty threat)
- Crossref abstract: "3000 artificial music tracks ... real instrument samples ... algorithmic composition ... three case studies ... instrument recognition". Abstract does NOT state real-music evaluation; full text inaccessible (Springer auth wall). Follow-up: Fricke, Vatolkin, Ostermann, "Application of Neural Architecture Search to Instrument Recognition in Polyphonic Audio", EvoMUSART 2023, LNCS 13988 pp.117-131 (abstract unavailable). Must read both before submission.

## Coordinator correction (mid-task)
Gain augmentation is +-12 dB per training CLIP (whole input), not per stem. Motivation: loudness-normalised renders + non-level-invariant log1p(100*mel) features => model learned a level prior (MoisesDB AP fell ~0.04 per -3 dB). Need searches on gain/level augmentation / level invariance in instrument recognition / tagging.

## Raw findings log

### Synthetic-to-real (claims 1, 2)
- Tanaka, Luo, Cheuk, Yoshii, Morishima, Dixon. "On the Use of Synthesized Datasets and Transformer Adaptors for Musical Instrument Recognition." ISMIR 2023 LBD (LP-10). https://ismir2023program.ismir.net/lbd_324.html — pretrain transformer on Slakh2100, FINE-TUNE on real target datasets; weight init beats scratch and adaptors. Difference from us: they fine-tune on real; we use zero real training data (synthetic-only), and compare to foundation models.
- Zhong, Cooper, Yamagishi, Minematsu. "Exploring Isolated Musical Notes as Pre-training Data for Predominant Instrument Recognition in Polyphonic Music." arXiv 2306.08850 (APSIPA 2023). https://arxiv.org/abs/2306.08850 — pretrain on isolated notes (NSynth-like), fine-tune on IRMAS. Again fine-tunes on real.
- Hung, Chen, Yang (?), "Multitask learning for frame-level instrument recognition" ICASSP 2019, arXiv 1811.01143 https://arxiv.org/abs/1811.01143 — MuseScore synthetic dataset (~344k pieces), synthetic-only training, reports good qualitative results on real songs (limited quantitative real eval).
- Zehren, Alunno, Bientinesi. "Analyzing and reducing the synthetic-to-real transfer gap in MIR: the task of automatic drum transcription." arXiv 2407.19823 (2024). https://arxiv.org/abs/2407.19823 — ADT; finds that more synthetic data volume is not the main lever; diversity (even musically incoherent) and realistic performance MIDI reduce gap. VERY close analogue to our claim 2 but different task (ADT not instrument ID). (Already cited as "Zehren 2024".)
- "Towards Realistic Synthetic Data for Automatic Drum Transcription" arXiv 2601.09520 (2026) — argues SoundFont libraries lack acoustic diversity; use high-quality one-shots. Analogue for our NSynth/Iowa diversity finding (different task).
- STAR Drums (TISMIR) https://transactions.ismir.net/articles/10.5334/tismir.244 — synthetic drums mixed with real melodic stems beats pure MIDI-rendered.
- Annotation-free AMT with scalable synthetic data + adversarial domain confusion, arXiv 2312.10402 — AMT, synthetic-only + DA.
- "Transfer of knowledge among instruments in AMT" arXiv 2305.00426 — synthetic zero-shot works but can't fully match real diverse recordings.

### Hierarchical / fine-grained (claims 7)
- Sechet, Bugiotti, Kowalski, d'Herouville, Langiewicz. "A Hierarchical Deep Learning Approach for Minority Instrument Detection." DAFx 2024. https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_60.pdf — Hornbostel-Sachs hierarchy on MedleyDB; claims hierarchy gives MORE reliable coarse-level detection. Opposes/contrasts our finding that splitting into 9 subtypes hurt coarse ranking (though ours confounded with adding new families + forgetting).
- Flores Garcia, Aguilar, Manilow, Pardo. "Leveraging Hierarchical Structures for Few-Shot Musical Instrument Recognition." ISMIR 2021. arXiv 2107.07029 https://arxiv.org/abs/2107.07029 — hierarchical prototypical nets on MedleyDB; lower mistake severity. (Already cited.)

### Augmentation (claim 2, gain aug)
- Kratimenos, Avramidis, Garoufis, Zlatintsi, Maragos. "Augmentation Methods on Monophonic Audio for Instrument Classification in Polyphonic Music." EUSIPCO 2020. arXiv 1911.12505 https://arxiv.org/abs/1911.12505 — synthesize polyphonic training by mixing monophonic segments (genre-matched, pitch/tempo-synced); eval on real IRMAS test (LRAP ~0.80). Closest "artificial mixes -> real mixes" instrument work. Note: they do use real (IRMAS train) single-predominant audio, not MIDI renders.
- Ramires & Serra. "Data Augmentation for Instrument Classification Robust to Audio Effects." arXiv 1907.08520 (DAFx 2019). https://arxiv.org/abs/1907.08520 — effects robustness for one-shots.
- McFee, Humphrey, Bello. "A Software Framework for Musical Data Augmentation" (MUDA). ISMIR 2015. https://archives.ismir.net/ismir2015/paper/000228.pdf — instrument recognition in mixes w/ pitch shift, time stretch, background noise, DRC; preliminary.
- Schlüter & Grill, "Exploring data augmentation for improved singing voice detection with neural networks" ISMIR 2015 — includes random loudness / filter aug (known; need to verify specific gain finding).
- No paper found that isolates level/gain augmentation as a fix for a learned loudness prior induced by loudness-normalised synthetic renders + non-level-invariant log features. Gain aug itself is commonplace (audiomentations Gain; standard in SED/DCASE pipelines), so claim must be framed as a diagnostic finding (level-prior measured as AP slope per dB), not novelty of the technique.

### Foundation models vs small (claim 1)
- Yuan et al. "MARBLE: Music Audio Representation Benchmark for Universal Evaluation." NeurIPS 2023 D&B. arXiv 2306.10548 https://arxiv.org/abs/2306.10548 — NSynth instrument & MTG-Jamendo instrument probing; MERT-95M ~70.7 NSynth acc.
- Li et al. "MERT" ICLR 2024 arXiv 2306.00107.
- MERT weights CC-BY-NC.

### Calibration (claim 4)
- Lukashevich, Grollmisch, Abeßer, Stober. "How reliable are posterior class probabilities in automatic music classification?" Audio Mostly 2023, ACM, doi:10.1145/3616195.3616228 — genre+instrument classification overconfidence; deep ensembles + temperature scaling; in-domain. Closest MIR calibration work; (as far as snippets show) not cross-dataset transfer of calibrators.
- Singh, Singh, Kumar, Arora. "TCP_alpha: Margin-Controlled Confidence estimation for reliable MIR." arXiv 2608.20326 (Aug 2026) — confidence head; raga ID; robustness under domain shift; 5% target labels restore.
- "Uncertainty Calibration for Deep Audio Classifiers" arXiv 2206.13071 (Interspeech 2022?) — ESC-50/GTZAN, ESC->GTZAN OOD; ECE.
- "Uncertainty Calibration of Multi-Label Bird Sound Classifiers" arXiv 2511.08261 (2025/2026) — BirdSet, multi-label ECE, notes focal->soundscape shift degrades calibration.
- General ML: Ovadia et al. NeurIPS 2019 "Can you trust your model's uncertainty?" — calibration degrades under dataset shift (well known). Yu et al. NeurIPS 2022 multi-domain temperature scaling. => claim 4 is a known general phenomenon; music-specific per-class Platt transfer failure for PANNs on multitracks is a modest new empirical datapoint.

### Licence-clean (claim 3)
- Evans et al. "Stable Audio Open." arXiv 2407.14358 (ICASSP 2025) — CC-only training (Freesound+FMA); acknowledges limited capability, does not quantify cost vs non-CC.
- Barnett? / Morreale et al. "Data Collection in Music Generation Training Sets: A Critical Analysis." ISMIR 2023 https://archives.ismir.net/ismir2023/paper/000003.pdf — provenance audit.
- Longpre et al. "Data Provenance Initiative" arXiv 2310.16787 — licence audit (text).
- No MIR paper found that quantifies accuracy cost of licence-clean training data for a classifier. Claim 3 looks NEW (as a measured ablation), though small effect and single model.
- Procedural data: "Exploring Procedural Data Generation for Automatic Acoustic Guitar Fingerpicking Transcription" arXiv 2508.07987 (2025) — procedural tabs + Karplus-Strong, synthetic-only reasonable, + small real FT beats real-only. Closest procedural-MIDI precedent (transcription, not instrument ID, not licence framing).

### Evaluation rigour (claim 5)
- Sturm. "A Simple Method to Determine if a Music Information Retrieval System is a 'Horse'." IEEE TMM 2014. https://www.researchgate.net/publication/265645782
- Sturm, Flexer et al. "Validity in Music Information Research Experiments." arXiv 2301.01578 https://arxiv.org/abs/2301.01578
- Gouyon, Sturm, Oliveira, Hespanhol, Langlois. "On Evaluation Validity in Music Autotagging." arXiv 1410.0001 https://arxiv.org/abs/1410.0001
- Flexer. "A Closer Look on Artist Filters for Musical Genre Classification." ISMIR 2007. https://archives.ismir.net/ismir2007/paper/000341.pdf ; Pampalk, Flexer, Widmer ISMIR 2005 (artist filter, 71%->27%).
- Urbano, Schedl, Serra. "Evaluation in Music Information Retrieval." JIIS 41(3), 2013 (statistical testing, MIREX critique). https://www.researchgate.net/publication/257580715
- Green et al. "A Critical Survey of Research in Music Genre Recognition" ISMIR 2024 (MPG pure link) https://pure.mpg.de/pubman/item/item_3621900_1/component/file_3621901/mus-24-gre-01-critical.pdf
- Pre-registration: "Pre-registration for Predictive Modeling" arXiv 2311.18807; NeurIPS 2020/2021 Pre-registration workshops (Bertinetto, Henriques, Albanie, Paganini, Varol). No MIR pre-registration paper found.
- Won, Ferraro, Bogdanov, Serra "Evaluation of CNN-based automatic music tagging models" SMC 2020 (known; standardised tagging eval).
- => claim 5: individual elements are known (artist filtering, validity critiques, bootstrap CIs); combination (pre-registered adoption rules + test-reuse ledger + negatives-labelled golden set + lenient confusable scoring) appears unreported in MIR. Good as a methods section / secondary contribution, not a headline.

### Forgetting / hierarchy (claim 7)
- No instrument-recognition-specific class-incremental paper found. Audio analogues: "Investigating catastrophic forgetting in sound event classification" arXiv 2609.11447 (2026; forgetting concentrated in head); "Class-Incremental Learning for SELD" arXiv 2411.12830.
- Our claim 7 is not true CIL (joint retraining with expanded label set + data mix shift); it's "taxonomy expansion / data-mixture interference." Closer framing: negative transfer when adding classes; hierarchical granularity trade-off (Sechet 2024 claims hierarchy helps coarse; Flores Garcia 2021).

### Fine-grained subtypes
- Guitar effects: Comunità, Stowell, Reiss JAES 2021 (arXiv 2012.03216); Hinrichs/Gerkens/Ostermann DAFx 2020 "Recognizing guitar effects and their parameter settings"; Stein et al. 2010 (IDMT-SMT-Audio-Effects). All monophonic/isolated guitar, not full mixes.
- No full-mix Rhodes vs Wurlitzer, synth-type, 808 vs acoustic drum recognition paper found.

### LALMs / products with confidence
- Ghosh et al. "Music Flamingo" ICLR 2026, arXiv 2511.10289 — NSynth 80.76, Medley-Solos 90.86 (isolated sounds, not mixes).
- Audio Flamingo Next arXiv 2604.10905.
- HalluAudio arXiv 2604.19300 — instrument ID under ambiguity near-random for Qwen2-Audio.
- "Music Hallucination in Audio-Language Models: A Hierarchical Formulation and Empirical Study" arXiv 2609.20195 (2026) — SALMONN instrument probing 95.3% acc with 98.5% yes-bias; supports need for explicit negatives (our golden set). Cite for motivation.

### More (round 2)
- Ostermann, Vatolkin, Ebeling. "AAM: a dataset of Artificial Audio Multitracks for diverse MIR tasks." EURASIP JASMP 2023. https://doi.org/10.1186/s13636-023-00278-7 — algorithmic composition + real instrument samples; 9 instrument groups, frame-level instrument recognition experiments. Could not access full text to verify whether trained-on-AAM models were tested on real music (VERIFY before claiming novelty of "synthetic-only -> real" for instrument recognition). Closest dataset-level precedent to our procedural pipeline.
- Majchrzak & Mańdziuk. "Training chord recognition models on artificially generated audio." arXiv 2508.05878 (2025) — AAM-only can serve when nothing else available.
- MuScriptor arXiv 2607.08168 (2026) — synthetic pretraining helps most in low-real-data regimes; synthetic alone insufficient for transcription.
- Jeon, Wichern, Germain, Le Roux (MERL). "Why does music source separation benefit from cacophony?" arXiv 2402.18407 (ICASSPW 2024) — random mixing; TR2026-012 analyses why gains plateau despite limitless combinations => analogue to our "10x data no gain".
- Jeon & Lee. "Towards robust music source separation on loud commercial music." ISMIR 2022, arXiv 2208.14355 — level/limiting mismatch between training and commercial masters degrades separation; musdb-L/XL. Closest analogue for our level-prior/gain finding (different task).
- Uhlich et al. ICASSP 2017 — random-gain/channel-swap/random-mix stem augmentation for separation (standard).
- YourMT3+ arXiv 2407.04822 — cross-dataset stem augmentation for transcription.
- OpenMIC SOTA (supervised, real train): PaSST 0.843, EAsT-KD 0.852, DyMN-L 0.855 mAP (Schmid, Koutini, Widmer 2023/2024). Our numbers (trained-9 subset AP) are NOT comparable; reviewers will ask for a real-trained upper bound on same subset.
- Foundation model instrument probes: MuQ arXiv 2501.01108 (NSynth: MuQ 79.2, MusicFM 76.2, MERT 72.6); MTG-Jamendo instrument ROC/AP: MERT-L 75.5/18.8, MuQ 74.8/19.1, MusicFM 74.6/18.5. Castellon, Donahue, Liang "Codified audio LM (Jukebox) probing" ISMIR 2021. Ding & Lerch "Audio Embeddings as Teachers for Music Classification" ISMIR 2023 (small models distilled from embeddings; instrument on OpenMIC) https://archives.ismir.net/ismir2023/paper/000068.pdf — RELEVANT to claim 1 (small model vs embeddings).
- Essentia models: CC BY-NC-SA (commercial license available).
- Synthetic-pattern pretraining for audio: arXiv 2410.00511; procedural audio pretraining arXiv 2606.14791.
- Audio-sheet retrieval (Carvalho, Widmer) arXiv 2309.12158: synthetic MSMD-only generalises poorly.

### OPEN QUESTION for Bradley
- Claim 1 wording: were the MERT/CLAP probes trained on the same synthetic data (=> "foundation features don't close the sim2real gap") or on real OpenMIC train (=> much stronger "synthetic-only small CNN = real-trained foundation probe")? Must be stated precisely.

### Round 3
- Zhong, Hirano, Shimada, Tateishi, Takahashi, Mitsufuji (Sony). "An Attention-based Approach to Hierarchical Multi-label Music Instrument Classification." ICASSP 2023. arXiv 2302.08136 https://arxiv.org/abs/2302.08136 — fine+coarse joint training on real multi-instrument recordings; joint training beats flat. (NB: Bradley's "Zhong et al. 2023" may refer to this or to the isolated-notes paper 2306.08850 — check which is cited.)
- Ding & Lerch. "Audio Embeddings as Teachers for Music Classification." ISMIR 2023 — small CP-ResNet/MobileNet students regularised by OpenL3/PaSST/etc embeddings on OpenMIC; teacher knowledge helps small models. Contrast: in our sim2real setting, foundation features give no advantage.
- "Persian Musical Instruments Classification Using Polyphonic Data Augmentation." arXiv 2511.05717 (2025) — synthetic polyphonic mixes from isolated recordings; MERT fine-tuned on 50k synthetic mixes; real test 0.841 acc. Closest "MERT trained on synthetic mixes, tested real".
- Chen, Su, Yang. "Electric guitar playing technique detection in real-world recordings." ISMIR 2015 https://archives.ismir.net/ismir2015/paper/000119.pdf ; Wu & Lerch "On drum playing technique detection in polyphonic mixtures" ISMIR 2016; Abeßer bass plucking styles (isolated notes). Fine-grained subtype ID IN FULL MIXES (Rhodes/Wurli, synth lead/pad/bass, 808 vs acoustic kit, slap/upright/picked bass, distorted/clean) — no prior work found.
- No MoisesDB instrument-tagging benchmark found (MoisesDB used for separation, stem retrieval: arXiv 2411.19806, 2609.35672, 2609.26956). => our MoisesDB stem-derived leaf-level tagging protocol is a candidate contribution.
- MuScriptor 2607.08168: randomly chosen soundfont from >250 + detuning; no ablation isolating soundfont count.
- "Transfer of knowledge among instruments in AMT" arXiv 2305.00426.
- Alexandari et al. ICML 2020 (bias-corrected calibration + label shift); LaSCal NeurIPS 2024 (label-shift calibration without target labels). Relevant mechanism for claim 4: OpenMIC clip priors != full-song priors.

## CROSS-CHECK AGAINST REPO DEBRIEFS (read-only)
- Claim 1 precise: ALL models (scratch CNN, MERT probe/MLP/LoRA, CLAP probe/LoRA) trained on SYNTHETIC only; real data used only for eval (+ calibration). m3.md: scratch 0.785 vs MERT probe 0.774 (Δ +0.011, CI [-0.009,+0.027]); MERT LoRA 0.776; CLAP probe 0.656. So claim = "foundation features (frozen or LoRA-adapted on synthetic) do not close the sim2real gap better than a 1.2M scratch CNN". That is a sharper and more novel claim than "small model ties foundation model".
- Claim 2 data scaling: 8k->82k clips: real trained-9 0.784 -> 0.785 (Δ +0.001 CI [-0.008,+0.011]) while synthetic val 0.948 -> 0.979. Nice clean "in-domain keeps rising, real flat" curve.
- Gain aug (coordinator correction): per-CLIP ±12 dB. m5.md: on MoisesDB AP rises monotonically with input gain (e.g. guitar.electric.distorted 0.635@-6dB, 0.756 as-is, 0.789@+3dB); after aug mean AP drop 0->-6 dB fell 0.047 -> 0.021; leaf mAP 0.41 -> 0.49. Cause: loudness-normalised renders make "louder => present" a learnable shortcut. Discovered via app transcode (ffmpeg mono->stereo -3 dB pan law).
- Claim 4 NUANCE (important): m5.md shows per-class Platt fit on OpenMIC train DID partly transfer to MoisesDB for voice/drums/guitar (ECE 0.56->0.11, 0.74->0.26, 0.71->0.26; precision >=0.90) but remains under-confident; piano poor (recall@0.3 0.31). Organ "~0.001-0.04" on golden songs is a RANKING failure (PANNs organ AP at chance 0.22 vs random 0.22) that monotone calibration cannot fix. Also m3.md: temperature fit on synthetic (T~1) is useless on real (MERT ~3x overconfident; real T=2.94). So the honest claim is: (a) synthetic-fit calibration does not transfer to real; (b) real clip-level (OpenMIC) Platt partially transfers to full-song multitracks, with residual under-confidence and class-dependent failure; (c) some "calibration failures" are actually ranking failures. Do NOT write "calibration transfer fails" flatly.
- Claim 7: v8 adoption FAILED: MoisesDB subset ΔmAP -0.0227 (CI [-0.041,+0.009]) > allowed -0.0119; keys.synth.lead AP 0.325 -> 0.165 (CI entirely < -0.01). Pre-registered gate worked as designed — nice evidence for claim 5.
- Golden set: 25 songs (MTG-Jamendo), single musician labeller (Bradley), e.g. organ in only 3/25 songs => very wide CIs on rare leaves.
- NOTE: OpenMIC already has explicit crowd-sourced NEGATIVE labels — "explicit negatives" alone isn't novel; novelty is a musician-labelled full-song set with negatives at fine-grained leaf level + "type not sure" parent labels + lenient confusable scoring.

## RECOMMENDATIONS

### (a) Strongest novel contributions
1. Claim 1 (sharpened) + 2a: "Under synthetic-only training, a 1.2M scratch CNN matches frozen and LoRA-adapted MERT on real music; more synthetic data saturates real performance while in-domain keeps rising; source diversity, not volume, moves it." First controlled sim2real study for instrument ID with foundation-model baselines.
2. Claim 2b: a measured level shortcut induced by loudness-normalised renders (AP-vs-dB slope on real multitracks), fixed by per-clip gain aug; the single largest real-music gain (0.41->0.49).
3. Claim 3: first measured licence cost for an MIR classifier (~0.02 mAP on real; synthetic tests overstate it 7x), with a licence-clean model that ties a CC-BY-NC foundation model.
Secondary: fine-grained full-mix leaf taxonomy + MoisesDB stem-derived leaf protocol + golden set; pre-registered gates (incl. a failed one) as methods.

### (b) Candidate titles
- "Synthetic-Only Training for Fine-Grained Instrument Recognition in Real Mixes: Diversity, Level Shortcuts, and the Limits of Foundation Features"
- "More Renders Don't Help: Source Diversity and Gain Invariance Close the Synthetic-to-Real Gap in Instrument Recognition"
- "What Does Licence-Clean Training Cost? A Measured Ablation for Fine-Grained Instrument Recognition"
- "Loudness-Normalised Renders Teach a Level Shortcut: Diagnosing Synthetic-to-Real Failure in Instrument Tagging"
- "Does MERT Close the Sim-to-Real Gap? Small CNNs vs Music Foundation Models Trained on Rendered MIDI"

### (c) Must-cite (closest)
Manilow+ Slakh WASPAA 2019; Hung+ ICASSP 2019 (1811.01143); Ostermann+ AAM JASMP 2023; Tanaka+ ISMIR 2023 LBD; Zhong+ 2306.08850 and/or Zhong+ ICASSP 2023 (2302.08136); Kratimenos+ EUSIPCO 2020 (1911.12505); Zehren+ 2407.19823; Jeon & Lee ISMIR 2022 (2208.14355); Li+ MERT ICLR 2024 / Yuan+ MARBLE 2023; Ding & Lerch ISMIR 2023; Humphrey+ OpenMIC ISMIR 2018; Pereira+ MoisesDB ISMIR 2023; Lukashevich+ Audio Mostly 2023; Ovadia+ NeurIPS 2019; Flexer ISMIR 2007; Sturm TMM 2014; Evans+ Stable Audio Open 2407.14358; Sechet+ DAFx 2024; Flores Garcia+ ISMIR 2021; Schmid+ EfficientAT/DyMN (OpenMIC SOTA ~0.855) (U).

### (d) Reviewer hits
- No real-trained upper bound on the same trained-9/leaf subsets -> can't size the sim2real gap or tell "tie" from "shared ceiling". OpenMIC SOTA 0.855 (all 20, real-trained) will be quoted against you.
- Golden set: 25 songs, single labeller (author), no inter-annotator agreement; rare leaves with ~3 positive songs -> huge CIs; MTG-Jamendo overlap with Essentia training data must be checked if Essentia is scored on it.
- MoisesDB leaf labels derived from stem names (label noise, 240 tracks/45 artists; woodwinds 14 positives).
- Licence ablation confounded (composition + dropped Rhodes source), no pooled CI, single seed.
- Claim 7 confounded (new families + label split + data-mix shift together) — not CIL; don't overclaim.
- Calibration: PANNs Platt partially transfers; organ is ranking, not calibration. Only temperature/Platt tried; no label-shift-aware method.
- Gain aug: "why not PCEN / per-clip normalisation?" needs an ablation.
- Single architecture/seed for most ablations; MERT-95M only (not 330M, MuQ, MusicFM).
- LBD-scale novelty vs full paper: full paper needs real-trained reference + seeds.

### (e) Most valuable extra experiment
Train the same 1.22M CNN and the MERT probe on REAL data (OpenMIC train for trained-9; MoisesDB artist-disjoint folds for leaves) and report the synthetic-only vs real-trained vs synthetic+real(fine-tune) triangle, 3 seeds, paired artist bootstrap. This sizes the gap, tests whether MERT's advantage appears only with real labels, and turns claim 1 into a clean result. Cheap add-ons: PCEN/per-clip-normalisation vs gain aug; pooled CI + 3 seeds for licence ablation; a second labeller on the golden set (kappa).
