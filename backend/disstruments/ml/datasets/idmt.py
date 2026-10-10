"""Fraunhofer IDMT-SMT datasets — TEST-ONLY (never trained on; Bradley's approval).

Licence: Zenodo records say CC BY-NC-ND 4.0; the bundled description PDFs say CC BY-SA 4.0
"for evaluation purpose". The stricter NC-ND governs: research/evaluation only, no
redistribution of derivatives. Every record is split="test".

Three loaders (all solo recordings of ONE instrument, so every node outside the recorded
instrument's subtree — cymbals included — is a known negative; masks are only ever
*inside* the instrument's subtree, where the source cannot decide the leaf):

idmt_bass  (IDMT-SMT-Bass, zenodo 7188892; `<root>/PS/<STYLE>/BS_k_EQ_m_<STYLE>_NO_s_f.wav`,
           optional `<root>/ES/{SLU,SLD}/BS_4_EQ_m_PS_FS_ES_SL?_s_f.wav`)
    Single notes. Plucking style from the folder, cross-checked against the filename token.
      FS -> bass.electric.fingered     PK -> bass.electric.picked
      SP, ST -> bass.electric.slap     MU -> bass.electric, fingered/picked/slap UNKNOWN
    Basses 1-3 are fretted (named in the PDF), so bass.electric.fretless is a known
    negative for them. Bass 4 appears only in ES/SLU+SLD (slides; the Zenodo text says
    they were "recorded with fretless bass guitar"): bass.electric.fretless positive,
    fingered/picked/slap UNKNOWN (technique and fretless are sibling leaves; a fingered
    fretless note is not decidable against the fretted-technique leaves). The rest of ES/
    is NOT loaded: all of it is FS with an expression technique, and ES/NO is byte-identical
    to PS/FS (would double-count). Group (`artist`) = physical bass, so the bootstrap
    resamples instruments (3 fretted + 1 fretless).

idmt_bass_tracks  (IDMT-SMT-Bass-Single-Track, zenodo 7544099; `<root>/annotation/NNN.xml`
           + `<root>/audio/NNN.wav`)
    17 bass lines with per-note `excitationStyle` (same codes as above). Track positive =
    union over its notes; if any note is MU, the technique leaves not already positive are
    UNKNOWN (a muted passage might be any of them). Group = instrument model. All 17
    tracks use the same bass (Fame Baphomet 4 NTB = idmt_bass bass 1, same group key):
    ONE bootstrap group, so instrument-resampled CIs collapse to zero width. Pass
    `group_by="track"` for a track-level bootstrap (states a weaker claim).

idmt_fx  (IDMT-SMT-Audio-Effects, zenodo 7544032; extracted as
           `<root>/<Gitarre monophon|Gitarre polyphon|Bass monophon>/Samples/<Effect>/<fileID>.wav`,
           optional `<root>/<family>/Lists/<Effect>/*.xml`)
    Single notes / chords, DI-recorded, effect applied in a DAW. fileID =
    `<I><setting><playstyle>-<midi:2><3 digits>-<fxgroup><fxtype:2><fxsetting>-<filenr>`
    (verified against the Lists XML `audiofile` fields when the XML is present).
      guitar (G mono, P poly):  NoFX -> guitar.electric.clean
                                Distortion, Overdrive -> guitar.electric.distorted
      bass (B):  NoFX/Distortion/Overdrive -> bass.electric, ALL its children UNKNOWN
                 (the numeric `playstyle` code has no published code table; the PDF prose
                 "finger-hard / finger-soft / picked" is not a mapping, and the basses are
                 unnamed so fretless is not decidable either).
    The folder must agree with the filename's fxtype (IDs read from the Lists XML).
    EQ (an undocumented 12th folder, fxgroup "No FX", amp-style EQ such as "Fender Twin")
    is skipped by default; `eq_as_clean=True` maps guitar EQ to guitar.electric.clean.
    Every other effect (chorus, flanger, phaser, tremolo, vibrato, delays, reverb) is
    SKIPPED (counted in stats): those are clean-ish tones with modulation/time effects;
    calling them "clean" would be a judgement call the dataset does not make.
    Caveat: NoFX is an unamplified DI signal — a domain gap vs amp'd clean guitar in mixes.
    Group = `instrumentsetting` digit (guitar settings shared by G and P files). A setting
    is an instrument x pick-up combination (2 guitars, 2 basses per the record), so groups
    are finer than physical instruments: CIs are optimistic.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..taxonomy import Taxonomy
from .base import DatasetIndex, Record, check_unique_ids, observed_exhaustive

LICENCE = ("CC BY-NC-ND 4.0 per Zenodo (bundled PDFs say CC BY-SA 4.0, evaluation purpose); "
           "test-only")

FINGERED, PICKED, SLAP, FRETLESS = ("bass.electric.fingered", "bass.electric.picked",
                                    "bass.electric.slap", "bass.electric.fretless")
TECHNIQUES = (FINGERED, PICKED, SLAP)
# plucking-style code -> (positive node, unknown subtree roots)
PLUCK = {"FS": (FINGERED, ()), "PK": (PICKED, ()), "SP": (SLAP, ()), "ST": (SLAP, ()),
         "MU": ("bass.electric", TECHNIQUES)}

# IDMT-SMT-Bass `BS_k`: physical bass (PDF "Filenames" table; 4 = fretless, ES/SL* only).
# Keys shared with idmt_bass_tracks (same Fame Baphomet bass, per the single-track README).
BASS_IDS = {"1": "fame_baphomet_4_ntb", "2": "warwick_4_string", "3": "yamaha_bb604_jazz_bass",
            "4": "idmt_fretless_bass"}
FRETLESS_BASSES = {"4"}

_PS_RE = re.compile(r"^BS_(\d)_EQ_(\d)_(FS|MU|PK|SP|ST)_NO_(\d)_(\d+)$")
_ES_SL_RE = re.compile(r"^BS_(\d)_EQ_(\d)_PS_FS_ES_(SLU|SLD)_(\d)_(\d+)$")
_OPEN_STRING_MIDI = {"1": 28, "2": 33, "3": 38, "4": 43}       # E1 A1 D2 G2

FX_GUITAR = {"NoFX": "guitar.electric.clean", "Distortion": "guitar.electric.distorted",
             "Overdrive": "guitar.electric.distorted"}
FX_BASS = {"NoFX", "Distortion", "Overdrive"}
FX_OTHER = {"Chorus", "Flanger", "Phaser", "Tremolo", "Vibrato", "FeedbackDelay",
            "SlapbackDelay", "Reverb"}
# Effect folder -> fxtype ID, read from the Lists XML (<fxtype><ID>/<name>) of the release.
# EQ (12) sits in fxgroup 1 "No FX" (amp-style EQ, e.g. setting "Fender Twin") but is not
# one of the 11 published effects; skipped unless `eq_as_clean=True` (guitar only).
FX_TYPE = {"NoFX": "11", "EQ": "12", "FeedbackDelay": "21", "SlapbackDelay": "22", "Reverb": "23",
           "Chorus": "31", "Flanger": "32", "Phaser": "33", "Tremolo": "34", "Vibrato": "35",
           "Distortion": "41", "Overdrive": "42"}
_FX_RE = re.compile(r"^([BGP])(\d)(\d)-(\d{2})(\d{3})-(\d)(\d{2})(\d)-(\d+)$")


def _instrument_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.casefold()).strip("_")


def _labels(tax: Taxonomy, positive_nodes: Iterable[str],
            unknown_roots: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """Solo recording: everything is observed except the given unknown subtrees."""
    pos = tax.close_upward(positive_nodes)
    unk = [u for u in unknown_roots if u not in pos]
    return pos, observed_exhaustive(tax, pos, unk)


def _per_leaf(records: list[Record], tax: Taxonomy) -> dict[str, int]:
    c = Counter(n for r in records for n in r.positive if tax.is_leaf(n) or
                not (set(tax.children(n)) & r.positive))
    return dict(sorted(c.items()))


# ---------------------------------------------------------------------- IDMT-SMT-Bass
def load_bass(root: Path | str, *, include_fretless: bool = True, strict: bool = True,
              taxonomy: Taxonomy | None = None) -> DatasetIndex:
    tax = taxonomy or Taxonomy.load()
    root = Path(root)
    files = sorted((root / "PS").glob("*/*.wav"))
    if include_fretless:
        files += sorted((root / "ES").glob("SL[UD]/*.wav"))
    if not files:
        raise FileNotFoundError(f"idmt_bass: no PS/<style>/*.wav under {root}")
    records, problems, styles = [], [], Counter()
    for f in files:
        folder, stem = f.parent.name, f.stem
        m = _PS_RE.match(stem)
        if m and f.parent.parent.name == "PS":
            k, pickup, style, string, fret = m.groups()
            if style != folder:
                problems.append(f"{folder}/{f.name}: folder style != filename style")
                continue
            node, unknown = PLUCK[style]
            pos_nodes = [node]
            if k in FRETLESS_BASSES:            # never happens in the release; stay honest
                pos_nodes.append(FRETLESS)
        else:
            m = _ES_SL_RE.match(stem)
            if not (m and m.group(3) == folder and m.group(1) in FRETLESS_BASSES):
                problems.append(f"{folder}/{f.name}: unrecognised name")
                continue
            k, pickup, style, string, fret = m.groups()
            pos_nodes, unknown = [FRETLESS], TECHNIQUES
        if k not in BASS_IDS:
            problems.append(f"{folder}/{f.name}: unknown bass id {k}")
            continue
        pos, obs = _labels(tax, pos_nodes, unknown)
        styles[style] += 1
        records.append(Record(
            dataset="idmt_bass", item_id=f"{f.parent.parent.name}/{folder}/{stem}", split="test",
            artist=BASS_IDS[k], positive=pos, observed=obs, source_labels=(style,),
            audio={"mix": f},
            extra={"bass": k, "pickup": int(pickup), "string": int(string), "fret": int(fret),
                   "midi_pitch": _OPEN_STRING_MIDI.get(string, 0) + int(fret)}))
    if problems and strict:
        raise ValueError(f"idmt_bass: {len(problems)} file(s) not parseable: {problems[:5]}")
    check_unique_ids("idmt_bass", [r.item_id for r in records])
    return DatasetIndex("idmt_bass", records,
                        {"dataset": "idmt_bass", "include_fretless": include_fretless,
                         "taxonomy_version": tax.version, "licence": LICENCE},
                        {"n_items": len(records), "styles": dict(sorted(styles.items())),
                         "n_groups": len({r.artist for r in records}),
                         "per_leaf": _per_leaf(records, tax), "skipped": problems})


# ---------------------------------------------------------------------- Single-Track
def _xml_text(node: ET.Element, tag: str) -> str:
    el = node.find(tag)
    return (el.text or "").strip() if el is not None else ""


def load_bass_tracks(root: Path | str, *, group_by: str = "instrument", strict: bool = True,
                     taxonomy: Taxonomy | None = None) -> DatasetIndex:
    if group_by not in ("instrument", "track"):
        raise ValueError("group_by must be 'instrument' or 'track'")
    tax = taxonomy or Taxonomy.load()
    root = Path(root)
    xmls = sorted((root / "annotation").glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"idmt_bass_tracks: no annotation/*.xml under {root}")
    records, missing_audio, unknown_codes = [], [], Counter()
    for x in xmls:
        doc = ET.parse(x).getroot()
        glob = doc.find("globalParameter")
        model = _xml_text(glob, "instrumentModel") if glob is not None else ""
        wav_name = (_xml_text(glob, "audioFileName") if glob is not None else "") or f"{x.stem}.wav"
        notes = []
        for ev in doc.iter("event"):
            notes.append({"onset": float(_xml_text(ev, "onsetSec") or 0),
                          "offset": float(_xml_text(ev, "offsetSec") or 0),
                          "pitch": int(float(_xml_text(ev, "pitch") or 0)),
                          "pluck": _xml_text(ev, "excitationStyle"),
                          "expression": _xml_text(ev, "expressionStyle")})
        plucks = Counter(n["pluck"] for n in notes)
        bad = {p: c for p, c in plucks.items() if p not in PLUCK}
        if bad:
            unknown_codes.update(bad)
            if strict:
                raise ValueError(f"idmt_bass_tracks {x.name}: unknown excitationStyle {bad}")
        pos_nodes = [PLUCK[p][0] for p in plucks if p in PLUCK]
        unknown = set()
        for p in plucks:
            unknown |= set(PLUCK[p][1]) if p in PLUCK else set(tax.children("bass.electric"))
        if not pos_nodes:
            pos_nodes = ["bass.electric"]
        pos, obs = _labels(tax, pos_nodes, unknown)
        audio = root / "audio" / wav_name
        if not audio.exists():
            missing_audio.append(wav_name)
        inst = _instrument_key(model) if model else None
        records.append(Record(
            dataset="idmt_bass_tracks", item_id=x.stem, split="test",
            artist=(inst if group_by == "instrument" else f"track:{x.stem}"),
            positive=pos, observed=obs, source_labels=tuple(sorted(plucks)),
            audio={"mix": audio} if audio.exists() else {},
            extra={"instrument": inst, "pluck_counts": dict(plucks),
                   "expression_counts": dict(Counter(n["expression"] for n in notes)),
                   "notes": notes}))
    check_unique_ids("idmt_bass_tracks", [r.item_id for r in records])
    n_groups = len({r.artist for r in records})
    stats = {"n_items": len(records), "n_groups": n_groups, "missing_audio": missing_audio,
             "per_leaf": _per_leaf(records, tax), "unknown_codes": dict(unknown_codes)}
    if n_groups < 2:
        stats["bootstrap_warning"] = (f"{n_groups} bootstrap group: group-resampled CIs have zero "
                                      "width; this set cannot measure cross-instrument variance")
    return DatasetIndex("idmt_bass_tracks", records,
                        {"dataset": "idmt_bass_tracks", "group_by": group_by,
                         "taxonomy_version": tax.version, "licence": LICENCE}, stats)


# ---------------------------------------------------------------------- Audio-Effects
def _fx_lists(root: Path) -> dict[str, dict[str, str]]:
    """fileID -> XML `audiofile` fields, from any `Lists/**/*.xml` present."""
    out: dict[str, dict[str, str]] = {}
    for x in root.glob("*/Lists/*/*.xml"):
        try:
            doc = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        for af in doc.iter("audiofile"):
            fields = {c.tag: (c.text or "").strip() for c in af}
            if fields.get("fileID"):
                out[fields["fileID"]] = fields
    return out


def load_fx(root: Path | str, *, include_poly: bool = True, eq_as_clean: bool = False,
            strict: bool = True,
            taxonomy: Taxonomy | None = None) -> DatasetIndex:
    tax = taxonomy or Taxonomy.load()
    root = Path(root)
    files = sorted(root.glob("*/Samples/*/*.wav"))
    if not files:
        raise FileNotFoundError(f"idmt_fx: no <family>/Samples/<Effect>/*.wav under {root}")
    lists = _fx_lists(root)
    records, problems = [], []
    skipped, per_effect, xml_checked = Counter(), Counter(), 0
    for f in files:
        effect, stem = f.parent.name, f.stem
        m = _FX_RE.match(stem)
        if not m:
            problems.append(f"{effect}/{f.name}: unrecognised fileID")
            continue
        inst, setting, play, midi, _rest, fxgroup, fxtype, fxsetting, filenr = m.groups()
        if effect in FX_TYPE and FX_TYPE[effect] != fxtype:
            problems.append(f"{effect}/{f.name}: fxtype {fxtype} != folder ({FX_TYPE[effect]})")
            continue
        meta = lists.get(stem)
        if meta is not None:                    # cross-check the filename decoding
            xml_checked += 1
            want = {"instrument": inst, "instrumentsetting": setting, "playstyle": play,
                    "midinr": str(int(midi)), "fxgroup": fxgroup, "fxtype": fxtype,
                    "fxsetting": fxsetting}
            diff = {k: (v, meta.get(k)) for k, v in want.items()
                    if meta.get(k) is not None and meta.get(k).lstrip("0") != v.lstrip("0")}
            if diff:
                problems.append(f"{effect}/{f.name}: filename vs Lists XML mismatch {diff}")
                continue
        if inst == "P" and not include_poly:
            skipped["poly_excluded"] += 1
            continue
        guitar_map = dict(FX_GUITAR, **({"EQ": "guitar.electric.clean"} if eq_as_clean else {}))
        if effect in FX_OTHER or (effect == "EQ" and not (eq_as_clean and inst in "GP")):
            skipped[effect] += 1
            continue
        if inst in "GP" and effect in guitar_map:
            pos, obs = _labels(tax, [guitar_map[effect]], ())
            group = f"guitar_setting:{setting}"
        elif inst == "B" and effect in FX_BASS:
            pos, obs = _labels(tax, ["bass.electric"], tax.children("bass.electric"))
            group = f"bass_setting:{setting}"
        else:
            problems.append(f"{effect}/{f.name}: unknown effect folder")
            continue
        per_effect[f"{inst}:{effect}"] += 1
        records.append(Record(
            dataset="idmt_fx", item_id=f"{f.parent.parent.parent.name}/{effect}/{stem}", split="test",
            artist=group, positive=pos, observed=obs, source_labels=(effect,),
            audio={"mix": f},
            extra={"effect": effect, "instrument": inst, "polyphonic": inst == "P",
                   "setting": int(setting), "playstyle": int(play), "midi": int(midi),
                   "fxtype": int(fxtype), "fxsetting": int(fxsetting)}))
    if problems and strict:
        raise ValueError(f"idmt_fx: {len(problems)} problem file(s): {problems[:5]}")
    check_unique_ids("idmt_fx", [r.item_id for r in records])
    return DatasetIndex("idmt_fx", records,
                        {"dataset": "idmt_fx", "include_poly": include_poly, "eq_as_clean": eq_as_clean,
                         "taxonomy_version": tax.version, "licence": LICENCE},
                        {"n_items": len(records), "per_effect": dict(sorted(per_effect.items())),
                         "skipped": dict(sorted(skipped.items())), "problems": problems,
                         "xml_checked": xml_checked,
                         "n_groups": len({r.artist for r in records}),
                         "playstyle_codes": dict(Counter(f"{r.extra['instrument']}:{r.extra['playstyle']}"
                                                         for r in records)),
                         "per_leaf": _per_leaf(records, tax)})
