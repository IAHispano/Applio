"use client";

import { useEffect, useState } from "react";
import { FileText, Music, Sliders, Wand2, RotateCcw, Volume2 } from "lucide-react";
import {
  apiGet,
  errMsg,
  fetchJob,
  fetchModels,
  fileBasename,
  type Job,
  outputUrl,
  pollJob,
  postForm,
  stopJob,
} from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { useSpeakers } from "../../lib/useSpeakers";
import AudioWavePlayer from "../AudioWavePlayer";
import RadioRow from "../RadioRow";
import ModelDropdown from "../ui/ModelDropdown";
import SliderField from "../ui/SliderField";

interface Voice {
  shortName: string;
  friendlyName: string;
  gender: string;
  locale: string;
}

export default function TtsForm() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const { t } = useI18n();
  const [filter, setFilter] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [rate, setRate] = useState(0);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [protect, setProtect] = useState(0.5);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [embedderModelCustom, setEmbedderModelCustom] = useState("");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [f0AutotuneStrength, setF0AutotuneStrength] = useState(1);
  const [proposedPitch, setProposedPitch] = useState(false);
  const [proposedPitchThreshold, setProposedPitchThreshold] = useState(155);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.5);
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

  const resetDefaults = () => {
    setPitch(0);
    setIndexRate(0.75);
    setVolumeEnvelope(1);
    setProtect(0.5);
    setF0Method("rmvpe");
    setEmbedderModel("contentvec");
    setEmbedderModelCustom("");
    setRate(0);
    setSplitAudio(false);
    setF0Autotune(false);
    setF0AutotuneStrength(1);
    setProposedPitch(false);
    setProposedPitchThreshold(155);
    setCleanAudio(false);
    setCleanStrength(0.5);
  };

  const shown = voices.filter(
    (v) =>
      !filter ||
      v.shortName.toLowerCase().includes(filter.toLowerCase()) ||
      v.friendlyName.toLowerCase().includes(filter.toLowerCase()),
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!text && !file) {
      setError(t("Enter text or upload a .txt file."));
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
      fd.append("f0AutotuneStrength", String(f0AutotuneStrength));
      fd.append("proposedPitch", String(proposedPitch));
      fd.append("proposedPitchThreshold", String(proposedPitchThreshold));
      fd.append("cleanAudio", String(cleanAudio));
      fd.append("cleanStrength", String(cleanStrength));
      fd.append("sid", String(sid));
      const { jobId: id } = await postForm<{ jobId: string }>("/api/tts", fd);
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || t("Submit failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-sm"
        >
          {error}
        </div>
      )}

      <form onSubmit={onSubmit} className="space-y-4">
        {/* Card 1: Speech Synthesis Source */}
        <div className="card space-y-4">
          <div className="border-b border-white/10 pb-3.5 space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText size={18} className="text-white" />
                <h2 className="text-base font-bold text-white m-0">{t("Speech Synthesis Source")}</h2>
              </div>
              <span className="text-xs text-neutral-400">
                {shown.length} {t("voices available")}
              </span>
            </div>
            <p className="text-xs text-neutral-400 m-0 leading-relaxed">
              {t("Enter text or upload a text file to synthesize speech before voice conversion.")}
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <label htmlFor="tts-text-input">{t("Text to Synthesize")}</label>
              <textarea
                id="tts-text-input"
                rows={3}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={t("Hello, this is Applio.")}
                className="w-full resize-y"
              />
            </div>

            <div>
              <label htmlFor="tts-file-input">{t("Or upload a .txt file")}</label>
              <input
                id="tts-file-input"
                type="file"
                accept=".txt"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div>
                <label htmlFor="tts-voice-filter">{t("Voice filter")}</label>
                <input
                  id="tts-voice-filter"
                  type="text"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder={t("e.g. en-US, Aria, Guy…")}
                />
              </div>

              <div>
                <label htmlFor="tts-voice-select">{t("Voice")}</label>
                <select id="tts-voice-select" value={voice} onChange={(e) => setVoice(e.target.value)}>
                  {shown.slice(0, 400).map((v) => (
                    <option key={v.shortName} value={v.shortName}>
                      {v.friendlyName} ({v.gender})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <SliderField
                  id="tts-speaking-rate"
                  label={t("Speaking rate")}
                  value={rate}
                  min={-100}
                  max={100}
                  step={1}
                  unit="%"
                  onChange={setRate}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Card 2: Target Voice Model */}
        <div className="card space-y-4">
          <div className="border-b border-white/10 pb-3.5 space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Music size={18} className="text-white" />
                <h2 className="text-base font-bold text-white m-0">{t("Target Voice Model")}</h2>
              </div>
            </div>
            <p className="text-xs text-neutral-400 m-0 leading-relaxed">
              {t("Select the target voice model and feature index for speech timbre conversion.")}
            </p>
          </div>

          <div className="space-y-3">
            <ModelDropdown
              models={models}
              selectedModel={pthPath}
              onSelect={setPthPath}
              onUnload={() => setPthPath("")}
            />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
              <div>
                <label htmlFor="tts-index-input">{t("Index (optional)")}</label>
                <input
                  id="tts-index-input"
                  type="text"
                  value={indexPath}
                  onChange={(e) => setIndexPath(e.target.value)}
                  placeholder={t("logs/model_name/added_...index")}
                />
              </div>

              <div>
                <label htmlFor="tts-speaker-id">{t("Speaker ID")}</label>
                <select
                  id="tts-speaker-id"
                  value={sid}
                  onChange={(e) => setSid(Number(e.target.value))}
                  disabled={speakers.length <= 1}
                >
                  {speakers.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="tts-export-format">{t("Export Format")}</label>
                <select
                  id="tts-export-format"
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                >
                  {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Card 3: Conversion Parameters */}
        <div className="card space-y-4">
          <div className="border-b border-white/10 pb-3.5 space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders size={18} className="text-white" />
                <h2 className="text-base font-bold text-white m-0">{t("Conversion Parameters")}</h2>
              </div>
              <button
                type="button"
                onClick={resetDefaults}
                className="text-xs text-neutral-400 hover:text-white flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <RotateCcw size={12} className="text-white" />
                <span>{t("Reset Defaults")}</span>
              </button>
            </div>
            <p className="text-xs text-neutral-400 m-0 leading-relaxed">
              {t("Adjust pitch shifting, feature index retrieval, and acoustic post-processing.")}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <SliderField
              id="tts-pitch"
              label={t("Pitch")}
              value={pitch}
              min={-24}
              max={24}
              step={1}
              unit="st"
              onChange={setPitch}
            />
            <SliderField
              id="tts-index-rate"
              label={t("Search Feature Ratio")}
              value={indexRate}
              min={0}
              max={1}
              step={0.05}
              onChange={setIndexRate}
            />
            <SliderField
              id="tts-volume-envelope"
              label={t("Volume Envelope")}
              value={volumeEnvelope}
              min={0}
              max={1}
              step={0.05}
              onChange={setVolumeEnvelope}
            />
            <SliderField
              id="tts-protect"
              label={t("Protect Voiceless Consonants")}
              value={protect}
              min={0}
              max={0.5}
              step={0.01}
              onChange={setProtect}
            />
          </div>

          <div className="space-y-4 pt-2 border-t border-white/5">
            <RadioRow
              label={t("Pitch extraction algorithm")}
              name="f0-convert"
              options={[
                "rmvpe",
                "fcpe",
                "crepe",
                "crepe-tiny",
                "hybrid[crepe+rmvpe]",
                "hybrid[crepe+fcpe]",
                "hybrid[rmvpe+fcpe]",
              ]}
              value={f0Method}
              onChange={setF0Method}
            />

            <RadioRow
              label={t("Embedder Model")}
              name="embedder-convert"
              options={[
                "contentvec",
                "spin",
                "spin-v2",
                "chinese-hubert-base",
                "japanese-hubert-base",
                "korean-hubert-base",
                "custom",
              ]}
              value={embedderModel}
              onChange={setEmbedderModel}
            />

            {embedderModel === "custom" && (
              <div>
                <label htmlFor="tts-custom-embedder">{t("Custom embedder path")}</label>
                <input
                  id="tts-custom-embedder"
                  type="text"
                  value={embedderModelCustom}
                  onChange={(e) => setEmbedderModelCustom(e.target.value)}
                  placeholder="rvc/models/embedders/embedders_custom/my-embedder"
                />
              </div>
            )}
          </div>

          <details className="pt-2 border-t border-white/5">
            <summary className="text-sm font-semibold text-neutral-300 cursor-pointer select-none">
              {t("Advanced Settings")}
            </summary>
            <div className="space-y-4 pt-3">
              <div className="flex flex-wrap gap-4">
                <label htmlFor="tts-split-audio" className="flex items-center gap-2 cursor-pointer text-sm">
                  <input
                    id="tts-split-audio"
                    type="checkbox"
                    checked={splitAudio}
                    onChange={(e) => setSplitAudio(e.target.checked)}
                  />
                  <span>{t("Split Audio")}</span>
                </label>
                <label htmlFor="tts-f0-autotune" className="flex items-center gap-2 cursor-pointer text-sm">
                  <input
                    id="tts-f0-autotune"
                    type="checkbox"
                    checked={f0Autotune}
                    onChange={(e) => setF0Autotune(e.target.checked)}
                  />
                  <span>{t("Autotune")}</span>
                </label>
                <label htmlFor="tts-proposed-pitch" className="flex items-center gap-2 cursor-pointer text-sm">
                  <input
                    id="tts-proposed-pitch"
                    type="checkbox"
                    checked={proposedPitch}
                    onChange={(e) => setProposedPitch(e.target.checked)}
                  />
                  <span>{t("Proposed Pitch")}</span>
                </label>
                <label htmlFor="tts-clean-audio" className="flex items-center gap-2 cursor-pointer text-sm">
                  <input
                    id="tts-clean-audio"
                    type="checkbox"
                    checked={cleanAudio}
                    onChange={(e) => setCleanAudio(e.target.checked)}
                  />
                  <span>{t("Clean Audio")}</span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <SliderField
                  id="tts-autotune-strength"
                  label={t("Autotune Strength")}
                  value={f0AutotuneStrength}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setF0AutotuneStrength}
                />
                <SliderField
                  id="tts-proposed-threshold"
                  label={t("Proposed Pitch Threshold")}
                  value={proposedPitchThreshold}
                  min={50}
                  max={1200}
                  step={1}
                  unit="Hz"
                  onChange={setProposedPitchThreshold}
                />
                <SliderField
                  id="tts-clean-strength"
                  label={t("Clean Strength")}
                  value={cleanStrength}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setCleanStrength}
                />
              </div>
            </div>
          </details>
        </div>

        {/* Card 4: Action Card */}
        <div className="card flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-neutral-400">
            <Volume2 size={16} className="text-white" />
            <span>{t("Synthesize speech text and perform timbre conversion.")}</span>
          </div>
          <button
            type="submit"
            className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
            disabled={busy}
          >
            <Wand2 size={16} className="shrink-0" />
            <span>{busy ? t("Submitting…") : t("Convert Speech")}</span>
          </button>
        </div>
      </form>

      {/* Conversion In Progress */}
      {job && (job.status === "running" || job.status === "queued") && (
        <div className="card space-y-3" role="status">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white">{t("Synthesizing Speech…")}</span>
            <button
              type="button"
              className="ghost h-7 px-2.5 text-xs text-red-400 hover:text-red-300 border-red-500/30 rounded-lg flex items-center gap-1"
              onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
            >
              {t("Cancel")}
            </button>
          </div>
          <div className="loader" role="progressbar" aria-label={t("Synthesizing speech…")}>
            <div className="loaderBar" />
          </div>
        </div>
      )}

      {/* Error state */}
      {job && job.status === "error" && (
        <div
          role="alert"
          className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
        >
          {job.error || t("Speech conversion failed.")}
        </div>
      )}

      {/* Synthesized Output Waveform Player */}
      {job && job.outputFile && job.status === "done" && (
        <div className="card space-y-3 animate-in fade-in duration-200">
          <div className="flex items-center justify-between text-xs text-neutral-400">
            <span className="font-semibold text-white">{t("Synthesized Speech Output")}</span>
            <span className="badge done text-[10px]">{t("Ready")}</span>
          </div>
          <AudioWavePlayer
            src={outputUrl(job.outputFile)}
            title={`${t("TTS Output:")} ${fileBasename(pthPath || "speech").replace(/\.(pth|onnx)$/i, "")}`}
            filename={fileBasename(job.outputFile)}
          />
        </div>
      )}
    </div>
  );
}
