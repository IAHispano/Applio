"use client";

import { useEffect, useState } from "react";
import JobPanel from "../JobPanel";
import { apiGet, apiSend, errMsg, postForm } from "../api";

export default function DownloadPanel() {
  const [link, setLink] = useState("");
  const [linkJob, setLinkJob] = useState<string | null>(null);
  const [dropFile, setDropFile] = useState<File | null>(null);
  const [dropMsg, setDropMsg] = useState("");
  const [pretrained, setPretrained] = useState<Array<{ name: string; sampleRates: string[] }>>([]);
  const [model, setModel] = useState("Titan");
  const [sr, setSr] = useState("40k");
  const [custom, setCustom] = useState(false);
  const [urlG, setUrlG] = useState("");
  const [urlD, setUrlD] = useState("");
  const [preJob, setPreJob] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<{ models: Array<{ name: string; sampleRates: string[] }> }>("/api/download/pretraineds")
      .then((p) => {
        setPretrained(p.models);
        if (p.models[0]) {
          setModel(p.models[0].name);
          if (p.models[0].sampleRates[0]) setSr(p.models[0].sampleRates[0]);
        }
      })
      .catch((e) => setError(errMsg(e)));
  }, []);

  async function downloadLink() {
    setError("");
    try {
      const { jobId } = await apiSend<{ jobId: string }>("/api/download", "POST", { modelLink: link });
      setLinkJob(jobId);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function drop() {
    setDropMsg("");
    if (!dropFile) return;
    const fd = new FormData();
    fd.append("file", dropFile);
    try {
      const r = await postForm<{ file: string; modelDir: string }>("/api/download/drop", fd);
      setDropMsg(`Saved ${r.file} → ${r.modelDir} ✓`);
    } catch (e) {
      setDropMsg(errMsg(e));
    }
  }

  async function downloadPretrained() {
    setError("");
    try {
      const body = custom ? { urlG, urlD } : { model, sampleRate: sr };
      const { jobId } = await apiSend<{ jobId: string }>("/api/download/pretraineds", "POST", body);
      setPreJob(jobId);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  return (
    <div>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <div className="card">
        <h2>From link</h2>
        <div className="row">
          <input
            type="text"
            value={link}
            onChange={(e) => setLink(e.target.value)}
            placeholder="Model link (Drive, HuggingFace, direct zip)…"
            style={{ flex: 1 }}
          />
          <button type="button" className="cta" onClick={downloadLink}>
            Download
          </button>
        </div>
      </div>
      <JobPanel jobId={linkJob} compact />

      <div className="card">
        <h2>Drop files</h2>
        <div className="row">
          <input
            type="file"
            accept=".pth,.index,.onnx"
            onChange={(e) => setDropFile(e.target.files?.[0] || null)}
          />
          <button type="button" className="ghost" onClick={drop}>
            Save File
          </button>
          <span className="muted">{dropMsg}</span>
        </div>
      </div>

      <div className="card">
        <h2>Pretrained models</h2>
        {!custom ? (
          <div className="grid2">
            <div>
              <label>Pretrained</label>
              <select
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  const m = pretrained.find((p) => p.name === e.target.value);
                  if (m?.sampleRates[0]) setSr(m.sampleRates[0]);
                }}
              >
                {pretrained.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>Sampling Rate</label>
              <select value={sr} onChange={(e) => setSr(e.target.value)}>
                {(pretrained.find((p) => p.name === model)?.sampleRates || [sr]).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <div className="grid2">
            <div>
              <label>Pretrained G URL</label>
              <input type="text" value={urlG} onChange={(e) => setUrlG(e.target.value)} />
            </div>
            <div>
              <label>Pretrained D URL</label>
              <input type="text" value={urlD} onChange={(e) => setUrlD(e.target.value)} />
            </div>
          </div>
        )}
        <label>
          <input type="checkbox" checked={custom} onChange={(e) => setCustom(e.target.checked)} /> Custom
          Pretrained
        </label>
        <div className="row" style={{ marginTop: 8 }}>
          <button type="button" className="ghost" onClick={downloadPretrained}>
            Download
          </button>
        </div>
      </div>
      <JobPanel jobId={preJob} />
    </div>
  );
}
