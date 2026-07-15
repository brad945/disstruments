"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";

type Activation = [number, number];
type InstrumentTag = { label: string; confidence: number; stem: string; activations: Activation[] };
type Report = {
  song: { id: number; title: string; artist: string; duration_s: number };
  global: any;
  instruments: InstrumentTag[];
  instruments_status: string;
  stems: { name: string }[];
  stems_status: string;
  structure: { status: string };
  provenance: { stages: { stage: string; model: string; status: string }[] };
};

const LANE_COLORS: Record<string, string> = {
  vocals: "var(--lane-vocals)", voice: "var(--lane-vocals)",
  drums: "var(--lane-drums)", cymbals: "var(--lane-drums)",
  bass: "var(--lane-bass)",
};
const laneColor = (t: InstrumentTag) =>
  LANE_COLORS[t.label] ?? LANE_COLORS[t.stem] ?? "var(--lane-other)";

export default function SongPage() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<Report | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    fetch(`/api/v1/songs/${id}/analysis`).then(async (r) => {
      if (r.ok) setReport(await r.json());
      else setNotFound(true);
    });
  }, [id]);

  if (notFound) return <div className="container"><div className="panel">No analysis yet — the job may still be running. <a href="/">back</a></div></div>;
  if (!report) return <div className="container"><div className="panel">loading…</div></div>;

  const g = report.global || {};
  const dur = report.song.duration_s || 1;

  return (
    <div className="container">
      <div className="panel">
        <h1 style={{ margin: "0 0 4px" }}>{report.song.title}</h1>
        <div style={{ color: "var(--muted)", marginBottom: 14 }}>{report.song.artist || "unknown artist"}</div>
        <div className="cards">
          <Card k="key" v={g.key?.value ?? "—"} c={conf(g.key?.confidence)} />
          <Card k="bpm" v={g.bpm?.value ?? "—"} c={conf(g.bpm?.confidence)} />
          <Card k="loudness" v={g.lufs != null ? `${g.lufs} LUFS` : "—"} />
          <Card k="time sig" v={g.time_signature ?? "—"} c="assumed" />
          <Card k="length" v={fmtDur(dur)} />
        </div>
      </div>

      <Waveform songId={report.song.id} />

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Instruments</h2>
        {report.instruments_status !== "ok" && (
          <div className="badge bad" style={{ marginBottom: 8 }}>tagging stage failed — partial report</div>
        )}
        {report.instruments.length === 0 && <div className="footnote">none detected above threshold</div>}
        {report.instruments.map((t) => (
          <div key={`${t.stem}-${t.label}`} className={`lane ${t.confidence < 0.6 ? "uncertain" : ""}`}>
            <div className="label">
              <span>{t.label}{t.confidence < 0.6 ? " ?" : ""}</span>
              <span className="conf mono">{t.confidence.toFixed(2)}</span>
            </div>
            <div className="track">
              {t.activations.map(([a, b], i) => (
                <div key={i} className="block"
                  style={{ left: `${(a / dur) * 100}%`, width: `${Math.max(0.5, ((b - a) / dur) * 100)}%`, background: laneColor(t) }} />
              ))}
            </div>
          </div>
        ))}
        <div className="footnote">
          “?” = confidence 0.30–0.60, treat as uncertain. Detections below 0.30 are hidden. Lane position shows when the instrument is audible.
        </div>
      </div>

      <StemMixer songId={report.song.id} stems={report.stems} status={report.stems_status} />

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Structure & chords</h2>
        <div className="footnote">coming in draft 2 (structure segmentation F9, chords F10)</div>
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Provenance</h2>
        {report.provenance?.stages?.map((s, i) => (
          <div className="progress-stage" key={i}>
            <span className={`dot ${s.status === "ok" ? "ok" : "failed"}`} />
            <span className="mono">{s.stage}</span>
            <span style={{ color: "var(--muted)" }}>{s.model} · {s.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Card({ k, v, c }: { k: string; v: any; c?: string }) {
  return <div className="card"><div className="k">{k}</div><div className="v mono">{v}</div>{c && <div className="c">{c}</div>}</div>;
}
const conf = (c?: number) => (c != null ? `conf ${c.toFixed(2)}` : undefined);
const fmtDur = (s: number) => `${Math.floor(s / 60)}:${Math.round(s % 60).toString().padStart(2, "0")}`;

function Waveform({ songId }: { songId: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`/api/v1/songs/${songId}/audio`);
        if (!r.ok) return;
        const buf = await r.arrayBuffer();
        const ctx = new AudioContext();
        const audio = await ctx.decodeAudioData(buf);
        if (cancelled) return;
        const canvas = canvasRef.current!;
        const w = (canvas.width = canvas.offsetWidth * 2);
        const h = (canvas.height = 180);
        const g = canvas.getContext("2d")!;
        const data = audio.getChannelData(0);
        const step = Math.floor(data.length / w);
        g.fillStyle = "#1a1e25"; g.fillRect(0, 0, w, h);
        g.strokeStyle = "#57e389"; g.globalAlpha = 0.8; g.beginPath();
        for (let x = 0; x < w; x++) {
          let min = 1, max = -1;
          for (let i = 0; i < step; i++) {
            const v = data[x * step + i] ?? 0;
            if (v < min) min = v; if (v > max) max = v;
          }
          g.moveTo(x, ((1 + min) / 2) * h);
          g.lineTo(x, ((1 + max) / 2) * h);
        }
        g.stroke();
        ctx.close();
      } catch { /* waveform is decorative; never block the report */ }
    })();
    return () => { cancelled = true; };
  }, [songId]);
  return <div className="panel"><canvas ref={canvasRef} className="waveform" /></div>;
}

function StemMixer({ songId, stems, status }: { songId: number; stems: { name: string }[]; status: string }) {
  const refs = useRef<Record<string, HTMLAudioElement | null>>({});
  const [muted, setMuted] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  const effectiveMuted = useMemo(() => {
    const out: Record<string, boolean> = {};
    for (const s of stems) out[s.name] = solo ? s.name !== solo : !!muted[s.name];
    return out;
  }, [stems, muted, solo]);

  useEffect(() => {
    for (const s of stems) {
      const el = refs.current[s.name];
      if (el) el.muted = effectiveMuted[s.name];
    }
  }, [effectiveMuted, stems]);

  const playAll = () => {
    const els = stems.map((s) => refs.current[s.name]).filter(Boolean) as HTMLAudioElement[];
    if (playing) { els.forEach((e) => e.pause()); setPlaying(false); return; }
    const t = Math.max(...els.map((e) => e.currentTime));
    els.forEach((e) => { e.currentTime = t; e.play(); });
    setPlaying(true);
  };

  if (status !== "ok") {
    return <div className="panel"><h2 style={{ marginTop: 0 }}>Stems</h2>
      <div className="badge bad">separation stage failed</div></div>;
  }
  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Stems</h2>
        <button className="btn" onClick={playAll}>{playing ? "⏸ pause all" : "▶ play all (synced)"}</button>
      </div>
      {stems.map((s) => (
        <div className="stemrow" key={s.name}>
          <span className="name">{s.name}</span>
          <button className={`mutebtn ${effectiveMuted[s.name] && !solo ? "active" : ""}`}
            onClick={() => setMuted((m) => ({ ...m, [s.name]: !m[s.name] }))}>M</button>
          <button className={`mutebtn solobtn ${solo === s.name ? "active" : ""}`}
            onClick={() => setSolo(solo === s.name ? null : s.name)}>S</button>
          <audio ref={(el) => { refs.current[s.name] = el; }} controls preload="none"
            src={`/api/v1/songs/${songId}/stems/${s.name}`} />
          <a className="btn ghost" href={`/api/v1/songs/${songId}/stems/${s.name}`} download={`${s.name}.wav`}>↓</a>
        </div>
      ))}
      <div className="footnote">M = mute, S = solo. Stems are only ever served to you (the uploader).</div>
    </div>
  );
}
