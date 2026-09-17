"use client";

import { useState } from "react";
import JobPanel from "../../components/JobPanel";
import { errMsg, postForm } from "../../lib/api";

export default function ToolsPage() {
  const [audio, setAudio] = useState<File | null>(null);
  const [method, setMethod] = useState("rmvpe");
  const [analyzeJob, setAnalyzeJob] = useState<string | null>(null);
  const [f0Job, setF0Job] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function needAudio(): boolean {
    if (!audio) {
      setError("Upload an audio file first.");
      return false;
    }
    return true;
  }

  async function analyze() {
    setError("");
    if (!needAudio() || !audio) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("audio", audio);
      const { jobId: id } = await postForm<{ jobId: string }>("/api/extra/analyze", fd);
      setAnalyzeJob(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function extractF0() {
    setError("");
    if (!needAudio() || !audio) return;
    try {
      const fd = new FormData();
      fd.append("audio", audio);
      fd.append("method", method);
      const { jobId: id } = await postForm<{ jobId: string }>("/api/extra/f0", fd);
      setF0Job(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  return (
    <div>
      <div className="mb-4">
        <h2 className="title" style={{ margin: 0 }}>
          Tools
        </h2>
        <p className="muted" style={{ margin: "4px 0 0" }}>
          Understand your audio and your pitch curves.
        </p>
      </div>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <div className="card">
        <label>Audio file (shared by both tools)</label>
        <input
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.mp4,.aac,.aiff,.webm"
          onChange={(e) => setAudio(e.target.files?.[0] || null)}
        />
      </div>

      <div className="card">
        <h2>Audio Analyzer</h2>
        <div className="row">
          <button type="button" className="cta" onClick={analyze} disabled={busy}>
            {busy ? "Analyzing…" : "Analyze Audio"}
          </button>
          <span className="muted">Sample rate, duration, channels + spectrogram plot.</span>
        </div>
      </div>
      <JobPanel jobId={analyzeJob} />

      <div className="card">
        <h2>F0 Curve Extractor</h2>
        <div className="row">
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            style={{ maxWidth: 200 }}
          >
            {["crepe", "fcpe", "rmvpe"].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button type="button" className="ghost" onClick={extractF0}>
            Extract F0 Curve
          </button>
        </div>
      </div>
      <JobPanel jobId={f0Job} />
    </div>
  );
}
