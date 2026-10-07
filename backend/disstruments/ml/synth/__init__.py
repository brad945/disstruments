"""Synthetic training-data pipeline (PRD F28, ML_ENGINEERING §3.2, docs/M2_PLAN.md).

MIDI windows -> per-stem rendering through registered sources (sfizz / FluidSynth / VST3
synths) -> randomized per-stem effects -> mix -> mastering -> labelled clip with full
provenance (`labels.json`). Labels are perfect by construction: the renderer decides which
taxonomy leaf each stem is, and nothing unlabelled is ever audible.
"""
