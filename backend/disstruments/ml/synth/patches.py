"""Our own synth patches (M2 D8): per-leaf parameter recipes with seeded randomization.

Factory presets (several with unverified licences) are never loaded. Every patch starts
from the plugin's init state (`plugin.reset()` + `_init_*`) and sets a randomized recipe,
so the returned recipe *and* the full parameter dict recorded by the engine are exact
ground truth for "how was this sound made" (docs/DIRECTION_NOTES.md).

Surge XT exposes many parameters only as discrete display strings ("300.6 ms"), so values
are snapped to the nearest valid value (`set_nearest`).
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

_NUM = re.compile(r"^\s*(-?inf|-?[\d.]+)\s*([a-zA-Z%]*)")
_SCALE = {"s": 1000.0, "ms": 1.0, "khz": 1000.0, "hz": 1.0}


def _parse(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    m = _NUM.match(str(v))
    if not m:
        return None
    x = float(m.group(1))
    return x * _SCALE.get(m.group(2).lower(), 1.0)


_VALID: dict[tuple[str, str], list[tuple[float, Any]]] = {}


def _valid_numeric(plugin, name: str) -> list[tuple[float, Any]]:
    key = (type(plugin).__name__ + str(getattr(plugin, "name", "")), name)
    if key not in _VALID:
        p = getattr(plugin, "_parameter_cache", None) or plugin.parameters
        valid = list(getattr(p[name], "valid_values", []) or [])
        nums = [(_parse(v), v) for v in valid]
        _VALID[key] = [(x, v) for x, v in nums if x is not None and np.isfinite(x)]
    return _VALID[key]


def set_nearest(plugin, name: str, target: float) -> Any:
    """Set a numeric parameter to the valid value nearest `target` (in the parameter's
    base unit: ms for times, Hz, dB, %). Returns the value actually set. Valid-value lists
    are parsed once per (plugin, parameter) and cached."""
    nums = _valid_numeric(plugin, name)
    if not nums:
        setattr(plugin, name, target)
        return target
    x, v = min(nums, key=lambda t: abs(t[0] - target))
    setattr(plugin, name, v)
    return v


def _set(plugin, rec: dict, name: str, value: Any) -> None:
    """Choice/string params are set directly; numeric ones via nearest valid value."""
    if isinstance(value, (str, bool)):
        setattr(plugin, name, value)
        rec[name] = value
    else:
        rec[name] = set_nearest(plugin, name, float(value))


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _lu(rng, lo, hi):   # log-uniform (times, frequencies)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# ---------------------------------------------------------------------- Surge XT
def _surge_base(plugin, rec, rng):
    _set(plugin, rec, "a_osc_1_type", "Classic")
    _set(plugin, rec, "a_osc_2_mute", bool(rng.random() < 0.5))
    _set(plugin, rec, "a_osc_drift", _u(rng, 0, 0.3))


def _surge_bass_analog(plugin, rng):
    rec: dict = {}
    _surge_base(plugin, rec, rng)
    _set(plugin, rec, "a_osc_1_shape", _u(rng, -100, 100))          # saw <-> square
    _set(plugin, rec, "a_osc_1_sub_mix", _u(rng, 0, 60))
    _set(plugin, rec, "a_octave", -1.0)
    _set(plugin, rec, "a_filter_1_type", str(rng.choice(["LP 24 dB", "LP Vintage Ladder", "LP 12 dB"])))
    _set(plugin, rec, "a_filter_1_cutoff", _lu(rng, 180, 1500))
    _set(plugin, rec, "a_filter_1_resonance", _u(rng, 0, 45))
    _set(plugin, rec, "a_filter_1_feg_mod_amount", _u(rng, 6, 40))
    _set(plugin, rec, "a_filter_eg_decay", _lu(rng, 60, 500))
    _set(plugin, rec, "a_filter_eg_sustain", _u(rng, 0, 40))
    _set(plugin, rec, "a_amp_eg_attack", _lu(rng, 1, 10))
    _set(plugin, rec, "a_amp_eg_sustain", _u(rng, 60, 100))
    _set(plugin, rec, "a_amp_eg_release", _lu(rng, 30, 200))
    return rec


def _surge_bass_808(plugin, rng):
    rec: dict = {}
    _set(plugin, rec, "a_osc_1_type", "Sine")
    _set(plugin, rec, "a_octave", float(rng.choice([-2.0, -1.0])))
    _set(plugin, rec, "a_waveshaper_type", str(rng.choice(["Off", "Soft", "Soft Harmonic 2"])))
    _set(plugin, rec, "a_waveshaper_drive", _u(rng, 0, 12))
    _set(plugin, rec, "a_amp_eg_attack", _lu(rng, 1, 5))
    _set(plugin, rec, "a_amp_eg_decay", _lu(rng, 300, 1800))
    _set(plugin, rec, "a_amp_eg_sustain", _u(rng, 0, 35))
    _set(plugin, rec, "a_amp_eg_release", _lu(rng, 80, 500))
    _set(plugin, rec, "a_portamento", _u(rng, 0, 60) if rng.random() < 0.4 else 0.0)
    return rec


def _surge_pad(plugin, rng):
    rec: dict = {}
    _surge_base(plugin, rec, rng)
    _set(plugin, rec, "a_osc_1_type", str(rng.choice(["Classic", "Wavetable", "Modern"])))
    _set(plugin, rec, "a_osc_1_unison_voices", f"{int(rng.integers(3, 8))} voices")
    _set(plugin, rec, "a_osc_1_unison_detune", _u(rng, 8, 35))
    _set(plugin, rec, "a_osc_2_octave", float(rng.choice([-1.0, 0.0, 1.0])))
    _set(plugin, rec, "a_filter_1_type", str(rng.choice(["LP 12 dB", "LP 24 dB"])))
    _set(plugin, rec, "a_filter_1_cutoff", _lu(rng, 700, 6000))
    _set(plugin, rec, "a_filter_1_resonance", _u(rng, 0, 30))
    _set(plugin, rec, "a_amp_eg_attack", _lu(rng, 150, 1500))
    _set(plugin, rec, "a_amp_eg_sustain", _u(rng, 70, 100))
    _set(plugin, rec, "a_amp_eg_release", _lu(rng, 400, 2500))
    return rec


def _surge_lead(plugin, rng):
    rec: dict = {}
    _surge_base(plugin, rec, rng)
    _set(plugin, rec, "a_osc_1_shape", _u(rng, -100, 100))
    nv = int(rng.integers(1, 4))
    _set(plugin, rec, "a_osc_1_unison_voices", f"{nv} voice" + ("s" if nv > 1 else ""))
    _set(plugin, rec, "a_filter_1_type", str(rng.choice(["LP 24 dB", "LP 12 dB", "LP Vintage Ladder"])))
    _set(plugin, rec, "a_filter_1_cutoff", _lu(rng, 1500, 9000))
    _set(plugin, rec, "a_filter_1_resonance", _u(rng, 0, 50))
    _set(plugin, rec, "a_amp_eg_attack", _lu(rng, 1, 40))
    _set(plugin, rec, "a_amp_eg_sustain", _u(rng, 70, 100))
    _set(plugin, rec, "a_amp_eg_release", _lu(rng, 50, 400))
    _set(plugin, rec, "a_portamento", _u(rng, 0, 80) if rng.random() < 0.3 else 0.0)
    drive = rng.random() < 0.3                                    # always set both keys
    _set(plugin, rec, "a_waveshaper_type", "Soft" if drive else "Off")
    _set(plugin, rec, "a_waveshaper_drive", _u(rng, 2, 10) if drive else 0.0)
    return rec


def _surge_pluck(plugin, rng):
    rec: dict = {}
    _surge_base(plugin, rec, rng)
    _set(plugin, rec, "a_osc_1_shape", _u(rng, -100, 100))
    _set(plugin, rec, "a_filter_1_type", str(rng.choice(["LP 24 dB", "LP 12 dB"])))
    _set(plugin, rec, "a_filter_1_cutoff", _lu(rng, 250, 1500))
    _set(plugin, rec, "a_filter_1_resonance", _u(rng, 5, 45))
    _set(plugin, rec, "a_filter_1_feg_mod_amount", _u(rng, 24, 60))
    _set(plugin, rec, "a_filter_eg_decay", _lu(rng, 60, 350))
    _set(plugin, rec, "a_filter_eg_sustain", 0.0)
    _set(plugin, rec, "a_amp_eg_attack", _lu(rng, 1, 5))
    _set(plugin, rec, "a_amp_eg_decay", _lu(rng, 120, 600))
    _set(plugin, rec, "a_amp_eg_sustain", _u(rng, 0, 15))
    _set(plugin, rec, "a_amp_eg_release", _lu(rng, 80, 400))
    return rec


# ---------------------------------------------------------------------- Dexed (DX7 FM)
def _dx_op(plugin, rec, k, *, coarse, fine=0, level, rates, levels, vel=3, detune=7):
    for name, v in (("f_coarse", coarse), ("f_fine", fine), ("output_level", level),
                    ("key_velocity", vel), ("osc_detune", detune)):
        _set(plugin, rec, f"op{k}_{name}", v)
    _set(plugin, rec, f"op{k}_mode", "RATIO")
    for i, (r, l) in enumerate(zip(rates, levels), start=1):
        _set(plugin, rec, f"op{k}_eg_rate_{i}", r)
        _set(plugin, rec, f"op{k}_eg_level_{i}", l)


def _dexed_ep(plugin, rng, *, tine: bool):
    """Two parallel carrier/modulator stacks (DX7 algorithm 5 family). `tine=True` is the
    bright FM e-piano (ratio-14 tine, louder bell); `tine=False` models a Rhodes-like
    mellow bark (ratio-1 modulators, lower index, strong velocity sensitivity)."""
    rec: dict = {}
    _set(plugin, rec, "algorithm", 5.0)
    _set(plugin, rec, "feedback", float(rng.integers(0, 7)))
    decay = lambda: [99, int(rng.integers(20, 45)), int(rng.integers(15, 35)), int(rng.integers(30, 60))]  # noqa: E731
    car_lv = [99, int(rng.integers(75, 92)), 0, 0]
    for k in (1, 3):                                       # carriers
        _dx_op(plugin, rec, k, coarse=1, fine=int(rng.integers(0, 4)), level=int(rng.integers(90, 99)),
               rates=decay(), levels=car_lv, vel=int(rng.integers(2, 5)), detune=int(rng.integers(5, 10)))
    if tine:
        _dx_op(plugin, rec, 2, coarse=1, level=int(rng.integers(55, 78)), rates=decay(), levels=car_lv,
               vel=int(rng.integers(3, 7)))
        _dx_op(plugin, rec, 4, coarse=14, level=int(rng.integers(55, 82)),
               rates=[99, int(rng.integers(55, 80)), 30, 60], levels=[99, 0, 0, 0], vel=int(rng.integers(3, 7)))
    else:
        _dx_op(plugin, rec, 2, coarse=1, level=int(rng.integers(45, 68)), rates=decay(), levels=car_lv,
               vel=int(rng.integers(5, 8)))
        _dx_op(plugin, rec, 4, coarse=1, level=int(rng.integers(40, 62)),
               rates=[99, int(rng.integers(40, 65)), 30, 60], levels=[99, 40, 0, 0], vel=int(rng.integers(5, 8)))
    for k in (5, 6):                                       # silence the rest
        _dx_op(plugin, rec, k, coarse=1, level=0, rates=[99, 99, 99, 99], levels=[0, 0, 0, 0])
    return rec


RECIPES: dict[str, dict[str, Callable]] = {
    "surge_xt": {"bass.synth.analog": _surge_bass_analog, "bass.synth.sub_808": _surge_bass_808,
                 "keys.synth.pad": _surge_pad, "keys.synth.lead": _surge_lead,
                 "keys.synth.pluck": _surge_pluck},
    "dexed": {"keys.electric_piano.fm": lambda p, r: _dexed_ep(p, r, tine=True),
              "keys.electric_piano.rhodes": lambda p, r: _dexed_ep(p, r, tine=False)},
}


def describe_parameters(plugin) -> dict[str, Any]:
    """Name -> current value / display string / first valid values (for writing recipes)."""
    return {n: {"value": getattr(p, "raw_value", None), "string": getattr(p, "string_value", None),
                "valid": list(getattr(p, "valid_values", []) or [])[:12]}
            for n, p in plugin.parameters.items()}


def make_patch(engine: str, leaf: str, rng: np.random.Generator, plugin) -> dict[str, Any]:
    """Apply a randomized recipe for `leaf` to `plugin`; return the recipe actually used."""
    recipes = RECIPES.get(engine) or {}
    if leaf not in recipes:
        raise NotImplementedError(f"no {engine} patch recipe for {leaf}")
    return recipes[leaf](plugin, rng)
