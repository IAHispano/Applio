"use client";

import {
  ArrowRight,
  Check,
  CheckCircle2,
  Folder,
  FolderArchive,
  Layers,
  Music,
  RotateCcw,
  Sliders,
  StopCircle,
  Wand2,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { apiGet, errMsg, fetchJob, fetchModels, type Job, pollJob, stopJob, submitJob } from "../lib/api";
import { useI18n } from "../lib/i18n";
import { useSpeakers } from "../lib/useSpeakers";
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
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    let stop = () => {};
    fetchJob(jobId)
      .then(({ job: j }) => {
        setJob(j);
        if (j.status !== "done" && j.status !== "error") {
          stop = pollJob(jobId, setJob);
        }
      })
      .catch((e) => setError(errMsg(e)));
    return () => stop();
  }, [jobId]);

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
    <form onSubmit={onSubmit} className="space-y-4">
      {/* 1. Folders & Voice Model Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">
                {t("Batch Source & Voice Model")}
              </h2>
            </div>
            {pthPath && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/10 text-white border border-white/10 font-medium">
                {t("Ready")}
              </span>
            )}
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Converts every supported audio file in the input folder (server-side paths).")}
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="batch-input-folder" className="text-xs font-medium text-neutral-300">
              {t("Input Folder (server path)")}
            </label>
            <input
              id="batch-input-folder"
              type="text"
              value={inputFolder}
              onChange={(e) => setInputFolder(e.target.value)}
              className="w-full mt-1 text-xs"
            />
          </div>
          <div>
            <label htmlFor="batch-output-folder" className="text-xs font-medium text-neutral-300">
              {t("Output Folder (server path)")}
            </label>
            <input
              id="batch-output-folder"
              type="text"
              value={outputFolder}
              onChange={(e) => setOutputFolder(e.target.value)}
              className="w-full mt-1 text-xs"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-neutral-300">{t("Voice Model")}</label>
            <div className="mt-1">
              <ModelDropdown
                models={models}
                selectedModel={pthPath}
                onSelect={setPthPath}
                onUnload={() => setPthPath("")}
              />
            </div>
          </div>
          <div>
            <label htmlFor="batch-index-path" className="text-xs font-medium text-neutral-300">
              {t("Index File (optional)")}
            </label>
            <input
              id="batch-index-path"
              type="text"
              value={indexPath}
              onChange={(e) => setIndexPath(e.target.value)}
              placeholder="logs/model/added.index"
              className="w-full mt-1 text-xs"
            />
          </div>
        </div>
      </div>

      {/* 2. Conversion Parameters Card */}
      <div className="card space-y-5">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">
                {t("Conversion Parameters")}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setPitch(0);
                  setIndexRate(0.75);
                  setVolumeEnvelope(1);
                  setProtect(0.5);
                  setF0Method("rmvpe");
                  setEmbedderModel("contentvec");
                  setExportFormat("WAV");
                  setSplitAudio(false);
                  setF0Autotune(false);
                  setCleanAudio(false);
                }}
                className="text-xs text-neutral-400 hover:text-white flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <RotateCcw size={12} className="text-white" />
                <span>{t("Reset Defaults")}</span>
              </button>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Fine-tune pitch, timbre retrieval, voiceless consonant protection, and synthesis algorithms.")}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <SliderField
            id="batch-pitch"
            label={t("Pitch Shift (Semitones)")}
            value={pitch}
            min={-24}
            max={24}
            step={1}
            unit="st"
            formatValue={(v) => `${v > 0 ? `+${v}` : v} semitones`}
            onChange={setPitch}
          />
          <SliderField
            id="batch-index-rate"
            label={t("Search Feature Ratio (Index Accent)")}
            value={indexRate}
            min={0}
            max={1}
            step={0.05}
            formatValue={(v) => `${v}`}
            onChange={setIndexRate}
          />
          <SliderField
            id="batch-volume-envelope"
            label={t("Volume Envelope (Dynamic Loudness)")}
            value={volumeEnvelope}
            min={0}
            max={1}
            step={0.05}
            formatValue={(v) => `${v}`}
            onChange={setVolumeEnvelope}
          />
          <SliderField
            id="batch-protect"
            label={t("Protect Voiceless Consonants")}
            value={protect}
            min={0}
            max={0.5}
            step={0.01}
            formatValue={(v) => `${v}`}
            onChange={setProtect}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 pt-4 border-t border-white/10">
          <div>
            <label htmlFor="batch-f0-method" className="text-xs font-medium text-neutral-300">
              {t("Pitch Extraction Algorithm")}
            </label>
            <select
              id="batch-f0-method"
              value={f0Method}
              onChange={(e) => setF0Method(e.target.value)}
              className="w-full mt-1 text-xs"
            >
              {F0.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="batch-embedder-model" className="text-xs font-medium text-neutral-300">
              {t("Speech Embedder Model")}
            </label>
            <select
              id="batch-embedder-model"
              value={embedderModel}
              onChange={(e) => setEmbedderModel(e.target.value)}
              className="w-full mt-1 text-xs"
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
          {speakers.length > 1 && (
            <div>
              <label htmlFor="batch-speaker-id" className="text-xs font-medium text-neutral-300">
                {t("Speaker ID")}
              </label>
              <select
                id="batch-speaker-id"
                value={sid}
                onChange={(e) => setSid(Number(e.target.value))}
                className="w-full mt-1 text-xs"
              >
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {t("Speaker")} {s}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div>
            <label htmlFor="batch-export-format" className="text-xs font-medium text-neutral-300">
              {t("Output Audio Format")}
            </label>
            <select
              id="batch-export-format"
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              className="w-full mt-1 text-xs"
            >
              {FORMATS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4 pt-3 border-t border-white/5 flex-wrap">
          <label htmlFor="batch-split-audio" className="flex items-center gap-2 cursor-pointer text-xs text-neutral-300">
            <input
              id="batch-split-audio"
              type="checkbox"
              checked={splitAudio}
              onChange={(e) => setSplitAudio(e.target.checked)}
            />
            <span>{t("Split in Chunks")}</span>
          </label>
          <label htmlFor="batch-f0-autotune" className="flex items-center gap-2 cursor-pointer text-xs text-neutral-300">
            <input
              id="batch-f0-autotune"
              type="checkbox"
              checked={f0Autotune}
              onChange={(e) => setF0Autotune(e.target.checked)}
            />
            <span>{t("Autotune")}</span>
          </label>
          <label htmlFor="batch-clean-audio" className="flex items-center gap-2 cursor-pointer text-xs text-neutral-300">
            <input
              id="batch-clean-audio"
              type="checkbox"
              checked={cleanAudio}
              onChange={(e) => setCleanAudio(e.target.checked)}
            />
            <span>{t("Clean Artifacts")}</span>
          </label>
        </div>
      </div>

      {/* 3. Action Card */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={busy || !pthPath || !inputFolder}
              className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
            >
              <Wand2 size={16} className="shrink-0" />
              <span>{busy ? t("Converting Batch…") : t("Convert Batch")}</span>
            </button>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
          >
            {error}
          </div>
        )}
      </div>

      {/* Running State */}
      {job && (job.status === "running" || job.status === "queued") && (
        <div className="card space-y-3 animate-in fade-in duration-200" role="status">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white">{t("Batch Conversion in Progress…")}</span>
            <button
              type="button"
              className="ghost h-7 px-2.5 text-xs text-red-400 hover:text-red-300 border-red-500/30 rounded-lg flex items-center gap-1"
              onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
            >
              <StopCircle size={13} />
              <span>{t("Cancel")}</span>
            </button>
          </div>
          <div className="loader" role="progressbar" aria-label={t("Batch conversion in progress…")}>
            <div className="loaderBar" />
          </div>
          <div className="flex items-center justify-between text-[11px] text-neutral-400 pt-1">
            <span>{t("Input:")} {inputFolder}</span>
            <span>{t("Output:")} {outputFolder}</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {job && job.status === "error" && (
        <div role="alert" className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs">
          {job.error || t("Batch conversion failed.")}
        </div>
      )}

      {/* Completed State */}
      {job && job.status === "done" && (
        <div className="card space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center gap-3 border-b border-white/10 pb-3">
            <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
              <CheckCircle2 size={18} className="text-white" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white m-0">{t("Batch Conversion Complete")}</h3>
              <p className="text-xs text-neutral-400 m-0 mt-0.5">
                {t("All audio files in the folder have been converted with the selected voice timbre.")}
              </p>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-1">
            <span className="text-xs text-neutral-400 block">{t("Saved Output Directory")}</span>
            <span className="text-xs text-neutral-200 font-medium select-all block break-all">
              {outputFolder}
            </span>
          </div>

          <div className="flex items-center justify-end gap-3 pt-1">
            <Link
              href={`/inference?model=${encodeURIComponent(pthPath)}`}
              className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-1.5"
            >
              <span>{t("Test in Single Inference")}</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      )}
    </form>
  );
}
