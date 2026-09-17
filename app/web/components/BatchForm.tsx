"use client";

import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchModels, submitJob } from "../lib/api";
import { useI18n } from "../lib/i18n";
import { useSpeakers } from "../lib/useSpeakers";
import JobPanel from "./JobPanel";

const F0 = ["crepe", "crepe-tiny", "rmvpe", "fcpe"];
const FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"];

export default function BatchForm() {
  const [models, setModels] = useState<string[]>([]);
  const { t } = useI18n();
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [inputFolder, setInputFolder] = useState("assets/audios");
  const [outputFolder, setOutputFolder] = useState("assets/audios/batch_output");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [sid, setSid] = useState(0);
  const [terms, setTerms] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const speakers = useSpeakers(pthPath);

  useEffect(() => {
    if (!speakers.includes(sid)) setSid(0);
  }, [speakers, sid]);

  // PresetsPanel can also target the batch form (Gradio had preset settings per tab).
  useEffect(() => {
    const onApply = (e: Event) => {
      const v = (e as CustomEvent).detail as {
        pitch: number;
        index_rate: number;
        rms_mix_rate: number;
        protect: number;
      };
      if (typeof v.pitch === "number") setPitch(Math.max(-24, Math.min(24, v.pitch)));
      if (typeof v.index_rate === "number") setIndexRate(v.index_rate);
      if (typeof v.rms_mix_rate === "number") setVolumeEnvelope(v.rms_mix_rate);
      if (typeof v.protect === "number") setProtect(v.protect);
    };
    window.addEventListener("applio:apply-preset-batch", onApply);
    return () => window.removeEventListener("applio:apply-preset-batch", onApply);
  }, []);

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
    if (!terms) {
      setError(t("You must agree to the Terms of Use to proceed."));
      return;
    }
    if (!pthPath) {
      setError(t("Select a voice model."));
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
        volumeEnvelope,
        protect,
        f0Method,
        exportFormat,
        embedderModel,
        splitAudio,
        f0Autotune,
        cleanAudio,
        sid,
      });
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || t("Submit failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <div className="card">
        <h2>{t("Batch Conversion")}</h2>
        <p className="muted">
          {t("Converts every supported audio file in the input folder (server-side paths) →")}{" "}
        </p>
        <div className="grid2">
          <div>
            <label>{t("Input Folder (server path)")}</label>
            <input type="text" value={inputFolder} onChange={(e) => setInputFolder(e.target.value)} />
          </div>
          <div>
            <label>{t("Output Folder (server path)")}</label>
            <input type="text" value={outputFolder} onChange={(e) => setOutputFolder(e.target.value)} />
          </div>
          <div>
            <label>{t("Voice Model")}</label>
            <input type="text" list="bmodels" value={pthPath} onChange={(e) => setPthPath(e.target.value)} />
            <datalist id="bmodels">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label>{t("Index File (optional)")}</label>
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
            <label>{t("Pitch extraction")}</label>
            <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {F0.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Volume Envelope: {volumeEnvelope} (default 1)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volumeEnvelope}
              onChange={(e) => setVolumeEnvelope(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Protect Voiceless Consonants: {protect} (default 0.5)</label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={protect}
              onChange={(e) => setProtect(Number(e.target.value))}
            />
          </div>
          <div>
            <label>{t("Embedder Model")}</label>
            <select value={embedderModel} onChange={(e) => setEmbedderModel(e.target.value)}>
              {[
                "contentvec",
                "spin",
                "spin-v2",
                "chinese-hubert-base",
                "japanese-hubert-base",
                "korean-hubert-base",
                "custom",
              ].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>{t("Speaker ID")}</label>
            <select value={sid} onChange={(e) => setSid(Number(e.target.value))}>
              {speakers.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>{t("Export Format")}</label>
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
          <label className="terms">
            <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
            <span>{t("I agree to the terms of use")}</span>
          </label>
          <button type="submit" className="cta" disabled={busy}>
            {busy ? t("Submitting…") : t("Convert Folder")}
          </button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <label>
            <input type="checkbox" checked={splitAudio} onChange={(e) => setSplitAudio(e.target.checked)} />{" "}
            {t("Split Audio")}
          </label>
          <label>
            <input type="checkbox" checked={f0Autotune} onChange={(e) => setF0Autotune(e.target.checked)} />{" "}
            {t("Autotune")}
          </label>
          <label>
            <input type="checkbox" checked={cleanAudio} onChange={(e) => setCleanAudio(e.target.checked)} />{" "}
            {t("Clean Audio")}
          </label>
        </div>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      </div>
      <JobPanel jobId={jobId} />
    </form>
  );
}
