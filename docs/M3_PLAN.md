# M3 Plan: Ablation Ladder + Calibration (started 2026-10-08)

ML_ENGINEERING §4.2 (ladder, with Amendment A2's from-scratch rung) and §4.4 / F30
(calibration). Goal: measure what each step up the ladder buys, for the licence-clean
backbone (CLAP) and the research reference (MERT-95M), on synthetic test **and** real
OpenMIC (trained-9 classes), and make the confidences mean something.

## Runs autonomously on the M5 ($0)

| # | Rung / item | Notes |
|---|---|---|
| R1 | Frozen backbone + linear probe | Done in M2 |
| R2 | Frozen backbone + 2-layer MLP head | Same embedding caches; minutes |
| R3 | **LoRA** on the top transformer blocks (CLAP audio encoder, MERT-95M) | `peft`; trains on MPS; ~1–2 h per model; batch sized for 16 GB |
| R5 | **From scratch** (A2): small CNN on log-mel, random init | Licence-clean by construction; log-mels cached once |
| C1 | **Temperature scaling per taxonomy level**, fit on a held-out **real** split (OpenMIC train, never trained on) vs on synthetic val | Reports ECE before/after on OpenMIC test; "which data you calibrate on" is itself the experiment |

Every rung is scored by the M1 harness with an artist-level paired bootstrap against R1.

## Needs Bradley's approval (not started)
- **R4 full fine-tune** on a rented GPU (~$20–100 per run). Only worth it if R3 shows
  headroom.
- **Scaling to 100k clips / ~60 leaves.** About 160 GB of renders at the current format,
  but only ~124 GB of disk is free. That means either compressing (lower sample rate /
  mono-only stems) or an external disk, plus more sample-library downloads at ~1 GB/hour.
- **MedleyDB evaluation** (access pending) for stem-level results and e-piano.

## Guardrails
- One GPU job at a time (16 GB RAM; two concurrent jobs were killed in M2).
- MERT-derived models stay research-only (CC-BY-NC weights); CLAP and from-scratch
  models are the licence-clean candidates. Synthetic data rendered from Lakh MIDI is
  research-only regardless (DIRECTION_NOTES).
