"use client";

import { useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import { errMsg, fetchModels, postForm } from "../../lib/api";

export default function VoiceBlenderPage() {
  const [models, setModels] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [p1, setP1] = useState("");
  const [p2, setP2] = useState("");
  const [f1, setF1] = useState<File | null>(null);
  const [f2, setF2] = useState<File | null>(null);
  const [ratio, setRatio] = useState(0.5);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        if (m.models[0]) setP1(m.models[0]);
        if (m.models[1]) setP2(m.models[1]);
      })
      .catch(() => {});
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!name || (!p1 && !f1) || (!p2 && !f2)) {
      setError("Name + two models are required (path or upload).");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("modelName", name);
      fd.append("pthPath1", p1);
      fd.append("pthPath2", p2);
      if (f1) fd.append("pth_file_1", f1);
      if (f2) fd.append("pth_file_2", f2);
      fd.append("ratio", String(ratio));
      const { jobId: id } = await postForm<{ jobId: string }>("/api/voice-blender", fd);
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || "Submit failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Voice Blender"
        description="Fuse and interpolate two trained voice models into a unique hybrid checkpoint."
      />
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <form onSubmit={onSubmit}>
        <div className="card">
          <div className="grid2">
            <div>
              <label>New model name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="my-fusion"
              />
            </div>
            <div>
              <label>Blend ratio: {ratio} (0 = model 1, 1 = model 2)</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={ratio}
                onChange={(e) => setRatio(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Model 1 path</label>
              <input type="text" list="vmodels" value={p1} onChange={(e) => setP1(e.target.value)} />
              <input type="file" accept=".pth,.onnx" onChange={(e) => setF1(e.target.files?.[0] || null)} />
            </div>
            <div>
              <label>Model 2 path</label>
              <input type="text" list="vmodels" value={p2} onChange={(e) => setP2(e.target.value)} />
              <input type="file" accept=".pth,.onnx" onChange={(e) => setF2(e.target.files?.[0] || null)} />
            </div>
          </div>
          <datalist id="vmodels">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="submit" className="cta" disabled={busy}>
              {busy ? "Blending…" : "Fuse Models"}
            </button>
          </div>
        </div>
      </form>
      <JobPanel jobId={jobId} />
    </div>
  );
}
