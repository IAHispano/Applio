"use client";

import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchModels, postForm } from "../api";
import JobPanel from "../JobPanel";

interface Voice {
  shortName: string;
  friendlyName: string;
  gender: string;
  locale: string;
}

export default function TtsForm() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [filter, setFilter] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [rate, setRate] = useState(0);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [terms, setTerms] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    apiGet<{ voices: Voice[] }>("/api/tts/voices")
      .then((v) => {
        setVoices(v.voices);
        if (v.voices[0]) setVoice(v.voices[0].shortName);
      })
      .catch((e) => setError(errMsg(e)));
    fetchModels()
      .then((m) => {
        setModels(m.models);
        if (m.models[0]) setPthPath(m.models[0]);
      })
      .catch(() => {});
  }, []);

  const shown = voices.filter(
    (v) =>
      !filter ||
      v.shortName.toLowerCase().includes(filter.toLowerCase()) ||
      v.friendlyName.toLowerCase().includes(filter.toLowerCase()),
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!terms) {
      setError("You must agree to the Terms of Use to proceed.");
      return;
    }
    if (!text && !file) {
      setError("Enter text or upload a .txt file.");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("ttsText", text);
      if (file) fd.append("txt_file", file);
      fd.append("ttsVoice", voice);
      fd.append("ttsRate", String(rate));
      fd.append("pthPath", pthPath);
      fd.append("indexPath", indexPath);
      fd.append("pitch", String(pitch));
      fd.append("indexRate", String(indexRate));
      fd.append("f0Method", f0Method);
      const { jobId: id } = await postForm<{ jobId: string }>("/api/tts", fd);
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || "Submit failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <form onSubmit={onSubmit}>
        <div className="card">
          <h2>Text</h2>
          <label>Text to synthesize</label>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Hello, this is Applio."
          />
          <label>…or upload .txt (UTF-8)</label>
          <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <div className="grid2">
            <div>
              <label>Voice filter</label>
              <input
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="e.g. en-US, Aria, Guy…"
              />
            </div>
            <div>
              <label>
                Voice ({shown.length} of {voices.length})
              </label>
              <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                {shown.slice(0, 400).map((v) => (
                  <option key={v.shortName} value={v.shortName}>
                    {v.friendlyName} ({v.gender})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>Speaking rate: {rate} (−100…100)</label>
              <input
                type="range"
                min={-100}
                max={100}
                step={1}
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
              />
            </div>
          </div>
        </div>
        <div className="card">
          <h2>Voice Model</h2>
          <div className="grid2">
            <div>
              <label>Voice Model</label>
              <input
                type="text"
                list="tmodels"
                value={pthPath}
                onChange={(e) => setPthPath(e.target.value)}
              />
              <datalist id="tmodels">
                {models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </div>
            <div>
              <label>Index (optional)</label>
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
                {["crepe", "crepe-tiny", "rmvpe", "fcpe"].map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <label className="terms">
            <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
            <span>I agree to the terms of use.</span>
          </label>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="submit" className="cta" disabled={busy}>
              {busy ? "Submitting…" : "Synthesize + Convert"}
            </button>
          </div>
        </div>
      </form>
      <JobPanel jobId={jobId} />
    </div>
  );
}
