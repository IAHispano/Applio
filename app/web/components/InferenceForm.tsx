"use client";

import { Activity, AudioWaveform, ChevronDown, Layers, Music, Sliders, Sparkles, Wand2 } from "lucide-react";
import Link from "next/link";
import type React from "react";
import { useEffect, useMemo, useState } from "react";
import {
  apiGet,
  errMsg,
  fetchJob,
  fetchModels,
  fileBasename,
  type Job,
  outputUrl,
  pollJob,
  stopJob,
  submitInference,
} from "../lib/api";
import { useI18n } from "../lib/i18n";
import { useSpeakers } from "../lib/useSpeakers";
import AudioWavePlayer from "./AudioWavePlayer";
import AudioDropzone from "./ui/AudioDropzone";
import ModelDropdown from "./ui/ModelDropdown";
import SliderField from "./ui/SliderField";

const F0_METHODS = [
  "rmvpe",
  "fcpe",
  "crepe",
  "crepe-tiny",
  "hybrid[crepe+rmvpe]",
  "hybrid[crepe+fcpe]",
  "hybrid[rmvpe+fcpe]",
  "hybrid[crepe+rmvpe+fcpe]",
];

const EMBEDDERS = [
  "contentvec",
  "spin",
  "spin-v2",
  "chinese-hubert-base",
  "japanese-hubert-base",
  "korean-hubert-base",
  "custom",
];

const FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"];

function matchIndex(model: string, indexes: string[]): string {
  if (!model || indexes.length === 0) return "";
  const normModel = model.replace(/\\/g, "/");
  const dir = normModel.includes("/") ? normModel.slice(0, normModel.lastIndexOf("/")) : "";
  const filename = normModel.split("/").pop() ?? "";
  const stem = filename.replace(/\.(pth|onnx)$/i, "").toLowerCase();
  const normIndexes = indexes.map((i) => i.replace(/\\/g, "/"));
  const sameDir = normIndexes.filter((i) => (i.includes("/") ? i.slice(0, i.lastIndexOf("/")) : "") === dir);
  const byStem = (list: string[]) =>
    list.find((i) => (i.split("/").pop() ?? "").toLowerCase().startsWith(stem.slice(0, 8)));
  const matchedNorm =
    byStem(sameDir.length > 0 ? sameDir : normIndexes) || (sameDir.length === 1 ? sameDir[0] : "") || "";
  if (!matchedNorm) return "";
  const matchedIdx = normIndexes.indexOf(matchedNorm);
  return matchedIdx >= 0 ? indexes[matchedIdx] : matchedNorm;
}

interface ModelDetail {
  pthPath: string;
  pthSize: number;
  indexSize: number | null;
  modifiedAt: string;
  folder: string;
}

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function InferenceForm() {
  const { t } = useI18n();

  // Model & Asset states
  const [models, setModels] = useState<string[]>([]);
  const [indexes, setIndexes] = useState<string[]>([]);
  const [sampleAudios, setSampleAudios] = useState<string[]>([]);
  const [loadError, setLoadError] = useState("");

  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [customIndexOpen, setCustomIndexOpen] = useState(false);
  const [sid, setSid] = useState(0);

  // Audio input states
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [inputPath, setInputPath] = useState("");

  // Core conversion settings
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1.0);
  const [protect, setProtect] = useState(0.5);

  // Algorithm & Export
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [embedderModelCustom, setEmbedderModelCustom] = useState("");
  const [exportFormat, setExportFormat] = useState("WAV");

  // Advanced settings
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [f0AutotuneStrength, setF0AutotuneStrength] = useState(1.0);
  const [proposedPitch, setProposedPitch] = useState(false);
  const [proposedPitchThreshold, setProposedPitchThreshold] = useState(155);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.5);

  // Formant Shifting
  const [formantShifting, setFormantShifting] = useState(false);
  const [formantQfrency, setFormantQfrency] = useState(1.0);
  const [formantTimbre, setFormantTimbre] = useState(1.0);

  // FX Rack
  const [postProcess, setPostProcess] = useState(false);
  const [reverb, setReverb] = useState(false);
  const [reverbRoomSize, setReverbRoomSize] = useState(0.5);
  const [reverbWetGain, setReverbWetGain] = useState(0.5);
  const [reverbDryGain, setReverbDryGain] = useState(0.5);

  const [delay, setDelay] = useState(false);
  const [delaySeconds, setDelaySeconds] = useState(0.3);
  const [delayMix, setDelayMix] = useState(0.4);

  const [compressor, setCompressor] = useState(false);
  const [compressorThreshold, setCompressorThreshold] = useState(-12);
  const [compressorRatio, setCompressorRatio] = useState(4);

  const [limiter, setLimiter] = useState(false);
  const [limiterThreshold, setLimiterThreshold] = useState(-2);

  const [chorus, setChorus] = useState(false);
  const [chorusRate, setChorusRate] = useState(1.0);
  const [chorusDepth, setChorusDepth] = useState(0.25);

  const [distortion, setDistortion] = useState(false);
  const [distortionGain, setDistortionGain] = useState(15);

  const [gain, setGain] = useState(false);
  const [gainDb, setGainDb] = useState(0);

  // Job submission & status
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  const speakers = useSpeakers(pthPath);
  const [library, setLibrary] = useState<ModelDetail[]>([]);

  // Cheap filesystem metadata (no Python): fills the model card once a model
  // is picked, where the empty-state hint used to be.
  useEffect(() => {
    apiGet<{ models: ModelDetail[] }>("/api/models/library")
      .then((r) => setLibrary(r.models || []))
      .catch(() => setLibrary([]));
  }, []);

  const selectedMeta = useMemo(
    () => library.find((m) => m.pthPath === pthPath) ?? null,
    [library, pthPath],
  );

  useEffect(() => {
    if (!speakers.includes(sid)) setSid(0);
  }, [speakers, sid]);

  // Load models, indexes and sample audios
  const loadAvailableModels = () => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        setIndexes(m.indexes);
        setSampleAudios(m.audios);
        if (m.models.length > 0 && !pthPath) {
          handleModelSelect(m.models[0], m.indexes);
        }
        setLoadError("");
      })
      .catch((e) => setLoadError(errMsg(e)));
  };

  // biome-ignore lint/correctness/useExhaustiveDependencies: initial model fetch
  useEffect(() => {
    loadAvailableModels();
  }, []);

  // Pre-select model from URL params if available (e.g. from Models library)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const m = params.get("model");
      if (m) {
        setPthPath(m);
        const idx = params.get("index");
        if (idx) setIndexPath(idx);
      }
    }
  }, []);

  // Poll running jobs
  // biome-ignore lint/correctness/useExhaustiveDependencies: poll active job until completion
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const stop = pollJob(job.id, setJob);
    return stop;
  }, [job?.id]);

  const handleModelSelect = (selected: string, idxList = indexes) => {
    setPthPath(selected);
    const matched = matchIndex(selected, idxList);
    setIndexPath(matched);
    setSid(0);
  };

  const handleUnloadModel = () => {
    setPthPath("");
    setIndexPath("");
    setSid(0);
  };

  // Temporary original audio URL for A/B comparison waveplayer
  const originalAudioUrl = useMemo(() => {
    if (audioFile) return URL.createObjectURL(audioFile);
    if (inputPath) return `/${inputPath}`;
    return null;
  }, [audioFile, inputPath]);

  const directAudioUrl = job?.outputFile ? outputUrl(job.outputFile) : null;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError("");
    if (!pthPath) {
      setSubmitError(t("Please select a voice model."));
      return;
    }
    if (!audioFile && !inputPath) {
      setSubmitError(t("Please provide an audio file or sample."));
      return;
    }

    const fd = new FormData();
    if (audioFile) fd.append("audio", audioFile);
    if (inputPath) fd.append("inputPath", inputPath);
    fd.append("pthPath", pthPath);
    fd.append("indexPath", indexPath);
    fd.append("pitch", String(pitch));
    fd.append("indexRate", String(indexRate));
    fd.append("volumeEnvelope", String(volumeEnvelope));
    fd.append("protect", String(protect));
    fd.append("f0Method", f0Method);
    fd.append("embedderModel", embedderModel);
    if (embedderModel === "custom" && embedderModelCustom) {
      fd.append("embedderModelCustom", embedderModelCustom);
    }
    fd.append("exportFormat", exportFormat);
    fd.append("splitAudio", String(splitAudio));
    fd.append("f0Autotune", String(f0Autotune));
    if (f0Autotune) fd.append("f0AutotuneStrength", String(f0AutotuneStrength));
    fd.append("proposedPitch", String(proposedPitch));
    if (proposedPitch) fd.append("proposedPitchThreshold", String(proposedPitchThreshold));
    fd.append("cleanAudio", String(cleanAudio));
    if (cleanAudio) fd.append("cleanStrength", String(cleanStrength));
    fd.append("sid", String(sid));

    if (formantShifting) {
      fd.append("formantShifting", "true");
      fd.append("formantQfrency", String(formantQfrency));
      fd.append("formantTimbre", String(formantTimbre));
    }

    if (postProcess) {
      fd.append("postProcess", "true");
      if (reverb) {
        fd.append("reverb", "true");
        fd.append("reverbRoomSize", String(reverbRoomSize));
        fd.append("reverbWetGain", String(reverbWetGain));
        fd.append("reverbDryGain", String(reverbDryGain));
      }
      if (delay) {
        fd.append("delay", "true");
        fd.append("delaySeconds", String(delaySeconds));
        fd.append("delayMix", String(delayMix));
      }
      if (compressor) {
        fd.append("compressor", "true");
        fd.append("compressorThreshold", String(compressorThreshold));
        fd.append("compressorRatio", String(compressorRatio));
      }
      if (limiter) {
        fd.append("limiter", "true");
        fd.append("limiterThreshold", String(limiterThreshold));
      }
      if (chorus) {
        fd.append("chorus", "true");
        fd.append("chorusRate", String(chorusRate));
        fd.append("chorusDepth", String(chorusDepth));
      }
      if (gain) {
        fd.append("gain", "true");
        fd.append("gainDb", String(gainDb));
      }
      if (distortion) {
        fd.append("distortion", "true");
        fd.append("distortionGain", String(distortionGain));
      }
    }

    setSubmitting(true);
    try {
      const { jobId } = await submitInference(fd);
      const { job: fresh } = await fetchJob(jobId);
      setJob(fresh);
    } catch (err) {
      setSubmitError(errMsg(err) || t("Submit failed"));
    } finally {
      setSubmitting(false);
    }
  }

  const isConverting = Boolean(submitting || (job && job.status !== "done" && job.status !== "error"));

  return (
    <form onSubmit={onSubmit} className="space-y-4 max-w-7xl mx-auto">
      {loadError && (
        <div className="card">
          <strong className="text-white">{t("API offline.")}</strong>{" "}
          <span className="muted">
            {t("Start it with")} <code>npm run dev</code>. {loadError}
          </span>
        </div>
      )}

      {/* Top Grid: Model & Audio Input */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
        {/* Voice Model Selector (5 cols) */}
        <div className="lg:col-span-5 space-y-4 h-full">
          <div className="card space-y-4 h-full flex flex-col">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Music size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Voice Model")}
                  </h2>
                </div>
                {pthPath && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/10 text-white border border-white/10">
                    {t("Ready")}
                  </span>
                )}
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t("Select the target voice checkpoint and paired feature index.")}
              </p>
            </div>

            {/* Custom Model Dropdown */}
            <ModelDropdown
              models={models}
              selectedModel={pthPath}
              indexes={indexes}
              onSelect={handleModelSelect}
              onUnload={handleUnloadModel}
              onRefresh={loadAvailableModels}
            />

            {/* Linked Index Status & Override */}
            {pthPath && (
              <div className="p-3 bg-white/[0.03] border border-white/10 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-400">{t("Index File:")}</span>
                  <button
                    type="button"
                    onClick={() => setCustomIndexOpen(!customIndexOpen)}
                    className="text-[11px] text-neutral-300 hover:text-white underline"
                  >
                    {customIndexOpen ? t("Hide index picker") : t("Change index")}
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-white/70 shrink-0" />
                  <span className="text-xs text-neutral-200 truncate flex-1">
                    {indexPath
                      ? indexPath.split("/").pop()
                      : t("No index paired (using model features only)")}
                  </span>
                </div>

                {customIndexOpen && (
                  <div className="pt-2 border-t border-white/5">
                    <select
                      value={indexPath}
                      onChange={(e) => setIndexPath(e.target.value)}
                      className="w-full text-xs"
                    >
                      <option value="">{t("None (0.0 index rate)")}</option>
                      {indexes.map((idx) => (
                        <option key={idx} value={idx}>
                          {fileBasename(idx)} ({idx})
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}

            {/* With a model picked, the card would otherwise end well above the
                audio card next to it — show what is actually loaded instead. */}
            {pthPath && selectedMeta && (
              <dl className="model-meta">
                <div>
                  <dt>{t("Checkpoint")}</dt>
                  <dd>{humanSize(selectedMeta.pthSize)}</dd>
                </div>
                <div>
                  <dt>{t("Index")}</dt>
                  <dd>{selectedMeta.indexSize ? humanSize(selectedMeta.indexSize) : "—"}</dd>
                </div>
                <div>
                  <dt>{t("Speakers")}</dt>
                  <dd>{speakers.length}</dd>
                </div>
                <div>
                  <dt>{t("Folder")}</dt>
                  <dd className="truncate" title={selectedMeta.folder}>
                    {selectedMeta.folder}
                  </dd>
                </div>
                <div>
                  <dt>{t("Modified")}</dt>
                  <dd>{new Date(selectedMeta.modifiedAt).toLocaleDateString()}</dd>
                </div>
              </dl>
            )}

            {/* Empty state: the card is as tall as the audio one, so use the
                room to say where models come from instead of leaving a void. */}
            {!pthPath && (
              <div className="flex flex-col items-center justify-center text-center gap-3 flex-1 py-6">
                <Music size={26} className="text-neutral-600" />
                <p className="text-xs text-neutral-400 m-0 max-w-[240px] leading-relaxed">
                  {models.length === 0
                    ? t("No models found in logs/. Download one or train your own to get started.")
                    : t("Pick a voice model above — its index file is paired automatically.")}
                </p>
                {models.length === 0 && (
                  <Link href="/download" className="ghost-link">
                    {t("Go to Download")}
                  </Link>
                )}
              </div>
            )}

            {/* Multi-speaker ID selector */}
            {speakers.length > 1 && (
              <div>
                <label htmlFor="speaker-id-select" className="text-xs font-medium text-neutral-300">
                  {t("Speaker ID (Multi-Speaker Model)")}
                </label>
                <select
                  id="speaker-id-select"
                  value={sid}
                  onChange={(e) => setSid(Number(e.target.value))}
                  className="w-full mt-1"
                >
                  {speakers.map((s) => (
                    <option key={s} value={s}>
                      {t("Speaker")} {s}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Audio Input & Drag & Drop Zone (7 cols) */}
        <div className="lg:col-span-7 space-y-4 h-full">
          <div className="card space-y-4 h-full">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AudioWaveform size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Audio Source")}
                  </h2>
                </div>
                {(audioFile || inputPath) && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/10 text-white border border-white/10">
                    {audioFile ? t("Uploaded File") : t("Library Sample")}
                  </span>
                )}
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t("Upload an audio file or select a sample from your library.")}
              </p>
            </div>

            {/* Interactive Drag & Drop + WavePlayer Component */}
            <AudioDropzone
              audioFile={audioFile}
              inputPath={inputPath}
              sampleAudios={sampleAudios}
              onFileSelect={setAudioFile}
              onPathSelect={setInputPath}
              disabled={isConverting}
            />
          </div>
        </div>
      </div>

      {/* Main Conversion Settings Card */}
      <div className="card space-y-5">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">
                {t("Conversion Parameters")}
              </h2>
            </div>
            <button
              type="button"
              onClick={() => {
                setPitch(0);
                setIndexRate(0.75);
                setVolumeEnvelope(1.0);
                setProtect(0.5);
              }}
              className="text-xs text-neutral-400 hover:text-white transition-colors cursor-pointer"
            >
              {t("Reset defaults")}
            </button>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Fine-tune pitch shifting, index feature retrieval, and audio envelope response.")}
          </p>
        </div>

        {/* 4 Core Voice Sliders (2x2 Grid) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Pitch Shift with Quick Octave Buttons */}
          <div className="space-y-2">
            <SliderField
              id="infer-pitch"
              label={t("Pitch Shift (Semitones)")}
              value={pitch}
              min={-24}
              max={24}
              step={1}
              unit="st"
              formatValue={(v) => `${v > 0 ? `+${v}` : v} semitones`}
              onChange={setPitch}
              description={t("-24 to 24 semitones (±12 = 1 full octave)")}
            />
            {/* Octave Quick Buttons */}
            <div className="flex items-center gap-1.5 pt-1">
              <span className="text-[10px] text-neutral-500 mr-1">{t("Quick:")}</span>
              <button
                type="button"
                onClick={() => setPitch(-12)}
                className="px-2 py-0.5 text-[10px] rounded bg-white/5 hover:bg-white/15 text-neutral-300 border border-white/10 transition-colors"
              >
                -12 (Male)
              </button>
              <button
                type="button"
                onClick={() => setPitch(0)}
                className="px-2 py-0.5 text-[10px] rounded bg-white/5 hover:bg-white/15 text-neutral-300 border border-white/10 transition-colors"
              >
                0 (Default)
              </button>
              <button
                type="button"
                onClick={() => setPitch(12)}
                className="px-2 py-0.5 text-[10px] rounded bg-white/5 hover:bg-white/15 text-neutral-300 border border-white/10 transition-colors"
              >
                +12 (Female)
              </button>
            </div>
          </div>

          {/* Search Feature Ratio */}
          <div>
            <SliderField
              id="infer-index-rate"
              label={t("Search Feature Ratio (Index Accent)")}
              value={indexRate}
              min={0}
              max={1}
              step={0.05}
              formatValue={(v) => `${v}`}
              onChange={setIndexRate}
              description={t("Weight of the index feature retrieval (0 = model only, 1 = max accent)")}
            />
          </div>

          {/* Volume Envelope */}
          <div>
            <SliderField
              id="infer-volume-envelope"
              label={t("Volume Envelope (Dynamic Loudness)")}
              value={volumeEnvelope}
              min={0}
              max={1}
              step={0.05}
              formatValue={(v) => `${v}`}
              onChange={setVolumeEnvelope}
              description={t("Match input audio loudness dynamics (1.0 = full dynamic match)")}
            />
          </div>

          {/* Consonant Protection */}
          <div>
            <SliderField
              id="infer-protect"
              label={t("Protect Voiceless Consonants")}
              value={protect}
              min={0}
              max={0.5}
              step={0.01}
              formatValue={(v) => `${v}`}
              onChange={setProtect}
              description={t("Shields voiceless consonants and breath sounds from artifacts (0.5 = neutral)")}
            />
          </div>
        </div>

        {/* Algorithm, Embedder & Export Format Row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-white/10">
          <div>
            <label htmlFor="f0-method-select" className="text-xs font-medium text-neutral-300">
              {t("Pitch Extraction Algorithm")}
            </label>
            <select
              id="f0-method-select"
              value={f0Method}
              onChange={(e) => setF0Method(e.target.value)}
              className="w-full mt-1"
            >
              {F0_METHODS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="embedder-model-select" className="text-xs font-medium text-neutral-300">
              {t("Speech Embedder Model")}
            </label>
            <select
              id="embedder-model-select"
              value={embedderModel}
              onChange={(e) => setEmbedderModel(e.target.value)}
              className="w-full mt-1"
            >
              {EMBEDDERS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="export-format-select" className="text-xs font-medium text-neutral-300">
              {t("Output Audio Format")}
            </label>
            <select
              id="export-format-select"
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              className="w-full mt-1"
            >
              {FORMATS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        {embedderModel === "custom" && (
          <div className="p-3 bg-white/[0.03] border border-white/10 rounded-xl">
            <label htmlFor="embedder-custom-input" className="text-xs font-medium text-neutral-300">
              {t("Custom Embedder Path (rvc/models/embedders/embedders_custom/...)")}
            </label>
            <input
              id="embedder-custom-input"
              type="text"
              value={embedderModelCustom}
              onChange={(e) => setEmbedderModelCustom(e.target.value)}
              placeholder="rvc/models/embedders/embedders_custom/my-embedder"
              className="w-full mt-1"
            />
          </div>
        )}

        {/* Collapsible Accordions: Advanced Tuning, Formant, Audio FX */}
        <div className="space-y-3 pt-2">
          {/* Advanced Pitch & Tuning Accordion */}
          <details className="group border border-white/10 rounded-xl overflow-hidden bg-white/[0.02]">
            <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-neutral-300 hover:text-white flex items-center justify-between select-none">
              <span className="flex items-center gap-2">
                <Activity size={15} />
                <span>{t("Advanced Pitch & Audio Cleanup")}</span>
              </span>
              <ChevronDown
                size={16}
                className="transition-transform duration-200 group-open:rotate-180 text-neutral-400"
              />
            </summary>
            <div className="p-4 border-t border-white/10 space-y-4 bg-black/20">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <label className="flex items-center gap-2 cursor-pointer p-2.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05]">
                  <input
                    type="checkbox"
                    checked={splitAudio}
                    onChange={(e) => setSplitAudio(e.target.checked)}
                  />
                  <span className="text-xs">{t("Split in Chunks")}</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer p-2.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05]">
                  <input
                    type="checkbox"
                    checked={f0Autotune}
                    onChange={(e) => setF0Autotune(e.target.checked)}
                  />
                  <span className="text-xs">{t("Autotune")}</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer p-2.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05]">
                  <input
                    type="checkbox"
                    checked={cleanAudio}
                    onChange={(e) => setCleanAudio(e.target.checked)}
                  />
                  <span className="text-xs">{t("Clean Artifacts")}</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer p-2.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05]">
                  <input
                    type="checkbox"
                    checked={proposedPitch}
                    onChange={(e) => setProposedPitch(e.target.checked)}
                  />
                  <span className="text-xs">{t("Proposed Pitch")}</span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                {f0Autotune && (
                  <SliderField
                    id="infer-autotune-strength"
                    label={t("Autotune Strength")}
                    value={f0AutotuneStrength}
                    min={0.1}
                    max={1}
                    step={0.05}
                    onChange={setF0AutotuneStrength}
                  />
                )}
                {cleanAudio && (
                  <SliderField
                    id="infer-clean-strength"
                    label={t("Clean Strength")}
                    value={cleanStrength}
                    min={0.1}
                    max={1}
                    step={0.05}
                    onChange={setCleanStrength}
                  />
                )}
                {proposedPitch && (
                  <SliderField
                    id="infer-pitch-thresh"
                    label={t("Pitch Threshold")}
                    value={proposedPitchThreshold}
                    min={50}
                    max={1200}
                    step={1}
                    unit="Hz"
                    onChange={setProposedPitchThreshold}
                  />
                )}
              </div>
            </div>
          </details>

          {/* Formant Shifting Accordion */}
          <details className="group border border-white/10 rounded-xl overflow-hidden bg-white/[0.02]">
            <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-neutral-300 hover:text-white flex items-center justify-between select-none">
              <span className="flex items-center gap-2">
                <Sparkles size={15} />
                <span>{t("Formant Shifting (Vocal Tract Modification)")}</span>
              </span>
              <ChevronDown
                size={16}
                className="transition-transform duration-200 group-open:rotate-180 text-neutral-400"
              />
            </summary>
            <div className="p-4 border-t border-white/10 space-y-4 bg-black/20">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formantShifting}
                  onChange={(e) => setFormantShifting(e.target.checked)}
                />
                <span className="text-xs font-medium text-white">{t("Enable Formant Shifting")}</span>
              </label>

              {formantShifting && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                  <SliderField
                    id="infer-formant-qfrency"
                    label={t("Formant Q-Frequency")}
                    value={formantQfrency}
                    min={0.5}
                    max={2.0}
                    step={0.05}
                    onChange={setFormantQfrency}
                    description={t("Modifies formants width and spectral envelope")}
                  />
                  <SliderField
                    id="infer-formant-timbre"
                    label={t("Formant Timbre")}
                    value={formantTimbre}
                    min={0.5}
                    max={2.0}
                    step={0.05}
                    onChange={setFormantTimbre}
                    description={t("Adjusts vocal tract length / timbre brightness")}
                  />
                </div>
              )}
            </div>
          </details>

          {/* Audio FX Rack Accordion */}
          <details className="group border border-white/10 rounded-xl overflow-hidden bg-white/[0.02]">
            <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-neutral-300 hover:text-white flex items-center justify-between select-none">
              <span className="flex items-center gap-2">
                <Layers size={15} />
                <span>{t("Post-Processing Audio FX Chain")}</span>
              </span>
              <ChevronDown
                size={16}
                className="transition-transform duration-200 group-open:rotate-180 text-neutral-400"
              />
            </summary>
            <div className="p-4 border-t border-white/10 space-y-4 bg-black/20">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={postProcess}
                  onChange={(e) => setPostProcess(e.target.checked)}
                />
                <span className="text-xs font-medium text-white">{t("Enable Master FX Rack")}</span>
              </label>

              {postProcess && (
                <div className="space-y-4 pt-2">
                  {/* Reverb */}
                  <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                    <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                      <input type="checkbox" checked={reverb} onChange={(e) => setReverb(e.target.checked)} />
                      <span>{t("Reverb")}</span>
                    </label>
                    {reverb && (
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <SliderField
                          id="fx-reverb-room"
                          label={t("Room Size")}
                          value={reverbRoomSize}
                          min={0.1}
                          max={1}
                          step={0.05}
                          onChange={setReverbRoomSize}
                        />
                        <SliderField
                          id="fx-reverb-wet"
                          label={t("Wet Mix")}
                          value={reverbWetGain}
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={setReverbWetGain}
                        />
                        <SliderField
                          id="fx-reverb-dry"
                          label={t("Dry Mix")}
                          value={reverbDryGain}
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={setReverbDryGain}
                        />
                      </div>
                    )}
                  </div>

                  {/* Delay */}
                  <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                    <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                      <input type="checkbox" checked={delay} onChange={(e) => setDelay(e.target.checked)} />
                      <span>{t("Stereo Delay")}</span>
                    </label>
                    {delay && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <SliderField
                          id="fx-delay-time"
                          label={t("Delay Time")}
                          value={delaySeconds}
                          min={0.05}
                          max={1.0}
                          step={0.05}
                          unit="s"
                          onChange={setDelaySeconds}
                        />
                        <SliderField
                          id="fx-delay-mix"
                          label={t("Delay Mix")}
                          value={delayMix}
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={setDelayMix}
                        />
                      </div>
                    )}
                  </div>

                  {/* Compressor & Limiter */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                        <input
                          type="checkbox"
                          checked={compressor}
                          onChange={(e) => setCompressor(e.target.checked)}
                        />
                        <span>{t("Compressor")}</span>
                      </label>
                      {compressor && (
                        <div className="space-y-2">
                          <SliderField
                            id="fx-comp-thresh"
                            label={t("Threshold")}
                            value={compressorThreshold}
                            min={-40}
                            max={0}
                            step={1}
                            unit="dB"
                            onChange={setCompressorThreshold}
                          />
                          <SliderField
                            id="fx-comp-ratio"
                            label={t("Ratio")}
                            value={compressorRatio}
                            min={1}
                            max={20}
                            step={0.5}
                            formatValue={(v) => `${v}:1`}
                            onChange={setCompressorRatio}
                          />
                        </div>
                      )}
                    </div>

                    <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                        <input
                          type="checkbox"
                          checked={limiter}
                          onChange={(e) => setLimiter(e.target.checked)}
                        />
                        <span>{t("Peak Limiter")}</span>
                      </label>
                      {limiter && (
                        <SliderField
                          id="fx-limiter-ceil"
                          label={t("Ceiling")}
                          value={limiterThreshold}
                          min={-12}
                          max={0}
                          step={0.5}
                          unit="dB"
                          onChange={setLimiterThreshold}
                        />
                      )}
                    </div>
                  </div>

                  {/* Chorus, Distortion & Gain */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                        <input
                          type="checkbox"
                          checked={chorus}
                          onChange={(e) => setChorus(e.target.checked)}
                        />
                        <span>{t("Chorus / Detune")}</span>
                      </label>
                      {chorus && (
                        <div className="space-y-2">
                          <SliderField
                            id="fx-chorus-rate"
                            label={t("Rate")}
                            value={chorusRate}
                            min={0.1}
                            max={5}
                            step={0.1}
                            unit="Hz"
                            onChange={setChorusRate}
                          />
                          <SliderField
                            id="fx-chorus-depth"
                            label={t("Depth")}
                            value={chorusDepth}
                            min={0.05}
                            max={1}
                            step={0.05}
                            onChange={setChorusDepth}
                          />
                        </div>
                      )}
                    </div>

                    <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                        <input
                          type="checkbox"
                          checked={distortion}
                          onChange={(e) => setDistortion(e.target.checked)}
                        />
                        <span>{t("Distortion")}</span>
                      </label>
                      {distortion && (
                        <SliderField
                          id="fx-dist-gain"
                          label={t("Drive Gain")}
                          value={distortionGain}
                          min={0}
                          max={40}
                          step={1}
                          unit="dB"
                          onChange={setDistortionGain}
                        />
                      )}
                    </div>

                    <div className="p-3.5 rounded-xl border border-white/10 bg-white/[0.02] space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white text-xs">
                        <input type="checkbox" checked={gain} onChange={(e) => setGain(e.target.checked)} />
                        <span>{t("Output Gain")}</span>
                      </label>
                      {gain && (
                        <SliderField
                          id="fx-gain-db"
                          label={t("Boost")}
                          value={gainDb}
                          min={-12}
                          max={12}
                          step={0.5}
                          unit="dB"
                          onChange={setGainDb}
                        />
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </details>
        </div>
      </div>

      {/* Convert Action, Live Progress & WavePlayer Result Card */}
      <div className="card space-y-4">
        {/* Action Header: Convert button & Status */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={isConverting || !pthPath || (!audioFile && !inputPath)}
              className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
            >
              <Wand2 size={16} className="shrink-0" />
              <span>{isConverting ? t("Converting Audio…") : t("Convert Audio")}</span>
            </button>

            {job && isConverting && (
              <button
                type="button"
                className="ghost h-10 px-4 flex items-center gap-1.5 text-xs font-medium rounded-xl text-red-400 hover:text-red-300 border-red-500/30"
                onClick={() => stopJob(job.id).catch((e) => setSubmitError(errMsg(e)))}
              >
                {t("Stop Conversion")}
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {job && (
              <span className={`badge ${job.status}`} role="status">
                {job.status === "done" ? t("Completed") : job.status === "running" ? t("In Progress") : job.status}
              </span>
            )}
          </div>
        </div>

        {/* Live Conversion Progress Bar (strictly progress bar, no spinners!) */}
        {isConverting && (
          <div className="p-4 bg-white/[0.03] border border-white/10 rounded-2xl space-y-2">
            <div className="flex items-center justify-between text-xs text-neutral-300">
              <span className="font-medium">
                {submitting
                  ? t("Submitting audio to AI voice model…")
                  : job?.status === "queued"
                    ? t("Queued in processing pipeline…")
                    : t("Processing inference with voice model…")}
              </span>
              <span className="text-neutral-400 capitalize">
                {submitting ? t("Uploading…") : job?.status || t("Working…")}
              </span>
            </div>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-white rounded-full transition-all duration-300 animate-pulse"
                style={{
                  width: submitting ? "30%" : job?.status === "queued" ? "50%" : "85%",
                }}
              />
            </div>
          </div>
        )}

        {/* Submit or Runtime Error */}
        {submitError && (
          <div
            role="alert"
            className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
          >
            {submitError}
          </div>
        )}

        {job && job.status === "error" && (
          <div
            role="alert"
            className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
          >
            {job.error || t("Inference job failed.")}
          </div>
        )}

        {/* Converted Audio WavePlayer Result with A/B Track Switching */}
        {job && directAudioUrl && job.status === "done" && (
          <div className="space-y-2 pt-2">
            <div className="flex items-center justify-between text-xs text-neutral-400 px-1">
              <span className="font-semibold text-white">{t("Conversion Output Waveform")}</span>
              <span>{t("Use A/B toggle to compare with original")}</span>
            </div>

            <AudioWavePlayer
              src={directAudioUrl}
              originalSrc={originalAudioUrl}
              title={`${t("Output:")} ${fileBasename(pthPath).replace(/\.(pth|onnx)$/i, "")}`}
              filename={job.outputFile ? fileBasename(job.outputFile) : undefined}
            />
          </div>
        )}
      </div>
    </form>
  );
}
