"use client";
import { useCallback, useEffect, useRef, useState } from "react";

type SongRow = {
  id: number; title: string; artist: string; duration_s: number;
  genres: string[]; job_id: number | null; job_status: string | null;
};
type StageEvent = { name: string; status: string };

export default function Home() {
  const [songs, setSongs] = useState<SongRow[]>([]);
  const [uploading, setUploading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [genres, setGenres] = useState("");
  const [stages, setStages] = useState<StageEvent[]>([]);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const esRef = useRef<EventSource | null>(null);

  const refresh = useCallback(async () => {
    const r = await fetch("/api/v1/songs");
    if (r.ok) setSongs(await r.json());
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const watchJob = useCallback((jobId: number) => {
    esRef.current?.close();
    setStages([]); setJobStatus("queued");
    const es = new EventSource(`/api/v1/jobs/${jobId}/events`);
    esRef.current = es;
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "stage") {
        setStages((prev) => {
          const next = prev.filter((s) => s.name !== ev.stage);
          return [...next, { name: ev.stage, status: ev.status }];
        });
      } else if (ev.type === "job") {
        setJobStatus(ev.status);
        if (["done", "failed", "partial"].includes(ev.status)) { es.close(); refresh(); }
      }
    };
    es.onerror = () => { es.close(); refresh(); };
  }, [refresh]);

  const upload = useCallback(async (file: File) => {
    setError(null); setUploading(true);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("genres", genres);
    try {
      const r = await fetch("/api/v1/songs", { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) { setError(body.detail || body.code || "upload failed"); return; }
      await refresh();
      if (body.job_id && !body.deduplicated) watchJob(body.job_id);
      if (body.deduplicated) setError("already analyzed — reused existing result");
    } finally { setUploading(false); }
  }, [genres, refresh, watchJob]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) upload(f);
  };

  return (
    <div className="container">
      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Analyze a song</h2>
        <div style={{ marginBottom: 10 }}>
          <input type="text" placeholder="genre tags (comma-separated, optional) — e.g. indie rock, pop"
            value={genres} onChange={(e) => setGenres(e.target.value)} />
        </div>
        <div className={`dropzone ${drag ? "drag" : ""}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}>
          {uploading ? "uploading…" : "drop an audio file here (mp3 / wav / flac / m4a / ogg) or click to choose"}
          <input ref={fileInput} type="file" accept="audio/*" hidden
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </div>
        {error && <div className="footnote" style={{ color: "var(--warn)" }}>{error}</div>}
        {jobStatus && (
          <div style={{ marginTop: 14 }}>
            <div className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
              job: {jobStatus}
            </div>
            {stages.map((s) => (
              <div className="progress-stage" key={s.name}>
                <span className={`dot ${s.status === "running" ? "running" : s.status === "ok" ? "ok" : s.status === "failed" ? "failed" : ""}`} />
                <span className="mono">{s.name}</span>
                <span style={{ color: "var(--muted)" }}>{s.status}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Library</h2>
        {songs.length === 0 ? (
          <div className="footnote">nothing analyzed yet</div>
        ) : (
          <table className="songs">
            <thead><tr><th>title</th><th>artist</th><th>length</th><th>genres</th><th>status</th></tr></thead>
            <tbody>
              {songs.map((s) => (
                <tr key={s.id}>
                  <td><a href={`/songs/${s.id}`}>{s.title}</a></td>
                  <td>{s.artist || "—"}</td>
                  <td className="mono">{fmtDur(s.duration_s)}</td>
                  <td>{s.genres.map((g) => <span key={g} className="badge" style={{ marginRight: 4 }}>{g}</span>)}</td>
                  <td><span className={`badge ${s.job_status === "done" ? "ok" : s.job_status === "partial" ? "warn" : s.job_status === "failed" ? "bad" : ""}`}>{s.job_status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function fmtDur(s: number) {
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
