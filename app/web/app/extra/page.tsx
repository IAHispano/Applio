"use client";

import { useState } from "react";
import JobPanel from "../../components/JobPanel";
import { errMsg, postForm, submitJob } from "../../lib/api";

export default function ExtraPage() {
  const [audio, setAudio] = useState<File | null>(null);
  const [method, setMethod] = useState("rmvpe");
  const [pth, setPth] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [infoJob, setInfoJob] = useState<string | null>(null);
  const [f0Job, setF0Job] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function analyze() {
    setError("");
    if (!audio) {
      setError("Upload an audio file first.");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("audio", audio);
      const { jobId: id } = await postForm<{ jobId: string }>("/api/extra/analyze", fd);
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function modelInfo() {
    setError("");
    try {
      const { jobId: id } = await submitJob("/api/extra/model-info", { pthPath: pth });
      setInfoJob(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function f0() {
    setError("");
    if (!audio) {
      setError("Upload an audio file first.");
      return;
    }
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
      <h2 className="title mb-4">Extra</h2>
      <div className="mb-4">
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
        <label>Audio file (shared by analyzer + F0)</label>
        <input
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.mp4,.aac,.aiff,.webm"
          onChange={(e) => setAudio(e.target.files?.[0] || null)}
        />
      </div>

      <div className="card">
        <h2>Audio Analyzer</h2>
        <button type="button" onClick={analyze} disabled={busy}>
          {busy ? "Analyzing…" : "Analyze Audio"}
        </button>
      </div>
      <JobPanel jobId={jobId} />

      <div className="card">
        <h2>F0 Curve Extractor</h2>
        <div className="row">
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {["crepe", "fcpe", "rmvpe"].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button type="button" className="ghost" onClick={f0}>
            Extract F0 Curve
          </button>
        </div>
      </div>
      <JobPanel jobId={f0Job} />

      <div className="card">
        <h2>Model Information</h2>
        <div className="row">
          <input
            type="text"
            value={pth}
            onChange={(e) => setPth(e.target.value)}
            placeholder="logs/my-model/model.pth"
            style={{ flex: 1 }}
          />
          <button type="button" className="ghost" onClick={modelInfo}>
            Inspect
          </button>
        </div>
      </div>
      <JobPanel jobId={infoJob} compact />
    </div>
  );
}
