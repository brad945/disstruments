"""Listening audit (M2 S3 gate): a few *dry* examples per source -> one local HTML page.

The point: catch source -> leaf mislabels (a "clean" guitar with drive, an "FM" piano
that is really a sampled Rhodes) by ear before they become label noise. Each source gets
`n` short excerpts rendered from real MIDI parts of its role, without effects, joined
with gaps into one MP3. The page has yes / no / unsure buttons per source and a button
that copies the verdicts as text to paste back.
"""
from __future__ import annotations

import html
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .engines import default_engines
from .midi import ROLE_LEAVES, load_windows
from .sources import load_registry


def _role_for(leaf: str) -> str | None:
    return next((r for r, ls in ROLE_LEAVES.items() if leaf in ls), None)


def build_audition(midi_root: Path, out: Path, *, n: int = 3, seconds: float = 6.0,
                   sr: int = 44100, seed: int = 0, max_files: int = 400) -> Path:
    import soundfile as sf

    reg = load_registry()
    eng = default_engines()
    rng = np.random.default_rng(seed)
    files = sorted(midi_root.rglob("*.mid"))
    rng.shuffle(files)
    pool: dict[str, list] = {}
    for f in files[:max_files]:                      # parts by role, for realistic phrases
        for w in load_windows(f, midi_root, max_windows=1):
            for p in w.parts:
                pool.setdefault(p.role, []).append(p)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in reg.sources:
        for leaf in (src.programs or {leaf: () for leaf in src.leaves}):
            if src.kind == "sf2" and leaf not in src.programs:
                continue
            role = _role_for(leaf)
            parts = pool.get(role or "", [])
            if not parts or src.engine not in eng:
                continue
            chunks = []
            for k in range(n):
                p = parts[int(rng.integers(len(parts)))]
                p = replace(p, leaf=leaf, notes=[x for x in p.notes if x.start < seconds])
                if not p.notes:
                    continue
                try:
                    a, _ = eng[src.engine].render(p, src, reg.root, sr, seconds, rng)
                except Exception as e:  # noqa: BLE001
                    rows.append((src, leaf, None, f"render failed: {e}"))
                    break
                peak = float(np.max(np.abs(a))) or 1.0
                chunks += [a / peak * 0.8, np.zeros(int(sr * 0.6), np.float32)]
            if not chunks:
                continue
            name = f"{src.id}__{leaf}.mp3".replace("/", "_")
            sf.write(out / name, np.concatenate(chunks), sr, format="MP3")
            rows.append((src, leaf, name, ""))
    page = out / "index.html"
    page.write_text(_html(rows))
    (out / "manifest.json").write_text(json.dumps(
        [{"source": s.id, "leaf": l, "file": f, "note": e} for s, l, f, e in rows], indent=1))
    return page


def _html(rows) -> str:
    items = []
    for i, (src, leaf, f, err) in enumerate(rows):
        audio = f'<audio controls preload="none" src="{html.escape(f)}"></audio>' if f else f"<em>{html.escape(err)}</em>"
        btns = "".join(f'<label><input type="radio" name="r{i}" value="{v}"> {v}</label> '
                       for v in ("yes", "no", "unsure"))
        items.append(f'<tr data-k="{html.escape(src.id)}|{html.escape(leaf)}"><td><b>{html.escape(leaf)}</b>'
                     f'<br><small>{html.escape(src.name)} · {html.escape(src.license)}</small></td>'
                     f'<td>{audio}</td><td>{btns}<br><input placeholder="note (optional)" size="28"></td></tr>')
    return f"""<!doctype html><meta charset="utf-8"><title>Disstruments source audition</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}td{{padding:8px;border-bottom:1px solid #ddd;vertical-align:top}}
button{{font-size:15px;padding:8px 14px}}</style>
<h2>Does each clip sound like the label?</h2>
<p>Each row is one sound source, labelled with the instrument type it will train. Listen for
anything that is clearly <b>not</b> that instrument (e.g. a "clean" guitar that's distorted, an
"upright bass" that sounds like a synth). Dry renders, no effects. "unsure" is fine.</p>
<table>{''.join(items)}</table>
<p><button onclick="copyOut()">Copy my answers</button> then paste them back to Claude.</p>
<script>
function copyOut(){{const lines=[];document.querySelectorAll('tr[data-k]').forEach(tr=>{{
 const r=tr.querySelector('input[type=radio]:checked');const n=tr.querySelector('input[placeholder]').value;
 lines.push(tr.dataset.k+' = '+(r?r.value:'skipped')+(n?' ('+n+')':''));}});
 navigator.clipboard.writeText(lines.join('\\n')).then(()=>alert('Copied '+lines.length+' answers'));}}
</script>"""
