"use client";

import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchModels, submitJob } from "../lib/api";
import { useI18n } from "../lib/i18n";
import { useSpeakers } from "../lib/useSpeakers";
import JobPanel from "./JobPanel";
import ModelDropdown from "./ui/ModelDropdown";
import SliderField from "./ui/SliderField";

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
            <label htmlFor="batch-input-folder">{t("Input Folder (server path)")}</label>
            <input
              id="batch-input-folder"
              type="text"
              value={inputFolder}
              onChange={(e) => setInputFolder(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="batch-output-folder">{t("Output Folder (server path)")}</label>
            <input
              id="batch-output-folder"
              type="text"
              value={outputFolder}
              onChange={(e) => setOutputFolder(e.target.value)}
            />
          </div>
          <div>
            <label>{t("Voice Model")}</label>
            <ModelDropdown
              models={models}
              selectedModel={pthPath}
              onSelect={setPthPath}
              onUnload={() => setPthPath("")}
            />
          </div>
          <div>
            <label htmlFor="batch-index-path">{t("Index File (optional)")}</label>
            <input
              id="batch-index-path"
              type="text"
              value={indexPath}
              onChange={(e) => setIndexPath(e.target.value)}
            />
          </div>
          <div>
            <SliderField
              id="batch-pitch"
              label={t("Pitch")}
              value={pitch}
              min={-24}
              max={24}
              step={1}
              unit="st"
              onChange={setPitch}
            />
          </div>
          <div>
            <SliderField
              id="batch-index-rate"
              label={t("Search Feature Ratio")}
              value={indexRate}
              min={0}
              max={1}
              step={0.05}
              onChange={setIndexRate}
            />
          </div>
          <div>
            <label htmlFor="batch-f0-method">{t("Pitch extraction")}</label>
            <select id="batch-f0-method" value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {F0.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <SliderField
              id="batch-volume-envelope"
              label={t("Volume Envelope")}
              value={volumeEnvelope}
              min={0}
              max={1}
              step={0.05}
              onChange={setVolumeEnvelope}
            />
          </div>
          <div>
            <SliderField
              id="batch-protect"
              label={t("Protect Voiceless Consonants")}
              value={protect}
              min={0}
              max={0.5}
              step={0.01}
              onChange={setProtect}
            />
          </div>
          <div>
            <label htmlFor="batch-embedder-model">{t("Embedder Model")}</label>
            <select
              id="batch-embedder-model"
              value={embedderModel}
              onChange={(e) => setEmbedderModel(e.target.value)}
            >
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
            <label htmlFor="batch-speaker-id">{t("Speaker ID")}</label>
            <select id="batch-speaker-id" value={sid} onChange={(e) => setSid(Number(e.target.value))}>
              {speakers.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="batch-export-format">{t("Export Format")}</label>
            <select
              id="batch-export-format"
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
            >
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
            {busy ? t("Submitting…") : t("Convert Folder")}
          </button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <label htmlFor="batch-split-audio" className="flex items-center gap-2 cursor-pointer">
            <input
              id="batch-split-audio"
              type="checkbox"
              checked={splitAudio}
              onChange={(e) => setSplitAudio(e.target.checked)}
            />{" "}
            {t("Split Audio")}
          </label>
          <label htmlFor="batch-f0-autotune" className="flex items-center gap-2 cursor-pointer">
            <input
              id="batch-f0-autotune"
              type="checkbox"
              checked={f0Autotune}
              onChange={(e) => setF0Autotune(e.target.checked)}
            />{" "}
            {t("Autotune")}
          </label>
          <label htmlFor="batch-clean-audio" className="flex items-center gap-2 cursor-pointer">
            <input
              id="batch-clean-audio"
              type="checkbox"
              checked={cleanAudio}
              onChange={(e) => setCleanAudio(e.target.checked)}
            />{" "}
            {t("Clean Audio")}
          </label>
        </div>
        {error && (
          <div
            role="alert"
            aria-live="assertive"
            className="mt-3 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
          >
            {error}
          </div>
        )}
      </div>
      <JobPanel jobId={jobId} />
    </form>
  );
}
