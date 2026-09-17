"use client";

import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchModels, submitJob } from "../lib/api";
import JobPanel from "./JobPanel";

const F0 = ["crepe", "crepe-tiny", "rmvpe", "fcpe"];
const FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"];

export default function BatchForm() {
  const [models, setModels] = useState<string[]>([]);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [inputFolder, setInputFolder] = useState("assets/audios");
  const [outputFolder, setOutputFolder] = useState("assets/audios/batch_output");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        if (m.models[0]) setPthPath(m.models[0]);
      })
      .catch(() => {});
    apiGet<{ audios: string[] }>("/api/models").catch(() => null);
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!pthPath) {
      setError("Select a voice model.");
      return;
    }
    setBusy(true);
    try {
      const { jobId: id } = await submitJob("/api/inference/batch", {
        pthPath,
        indexPath,
        inputFolder,
        outputFolder,
        pitch,
        indexRate,
        volumeEnvelope: 1,
        protect: 0.33,
        f0Method,
        exportFormat,
        embedderModel: "contentvec",
        sid: 0,
      });
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || "Submit failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <div className="card">
        <h2>Batch Conversion</h2>
        <p className="muted">
          Converts every supported audio file in the input folder (server-side paths) →{" "}
        </p>
        <div className="grid2">
          <div>
            <label>Input Folder (server path)</label>
            <input type="text" value={inputFolder} onChange={(e) => setInputFolder(e.target.value)} />
          </div>
          <div>
            <label>Output Folder (server path)</label>
            <input type="text" value={outputFolder} onChange={(e) => setOutputFolder(e.target.value)} />
          </div>
          <div>
            <label>Voice Model</label>
            <input type="text" list="bmodels" value={pthPath} onChange={(e) => setPthPath(e.target.value)} />
            <datalist id="bmodels">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label>Index File (optional)</label>
            <input type="text" value={indexPath} onChange={(e) => setIndexPath(e.target.value)} />
          </div>
          <div>
            <label>Pitch: {pitch}</label>
            <input
              type="range"
              min={-24}
              max={24}
              step={1}
              value={pitch}
              onChange={(e) => setPitch(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Search Feature Ratio: {indexRate}</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={indexRate}
              onChange={(e) => setIndexRate(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Pitch extraction</label>
            <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {F0.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Export Format</label>
            <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
              {FORMATS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button type="submit" className="cta" disabled={busy}>
            {busy ? "Submitting…" : "Convert Folder"}
          </button>
        </div>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      </div>
      <JobPanel jobId={jobId} />
    </form>
  );
}
