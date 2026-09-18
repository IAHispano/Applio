"use client";

import { Activity, AudioWaveform, Info, LineChart } from "lucide-react";
import { useEffect, useState } from "react";
import AudioPlayer from "../../components/AudioPlayer";
import AnalysisResultCard from "../../components/extra/AnalysisResultCard";
import PageHeader from "../../components/layout/PageHeader";
import ModelInfoCard, { type ModelMetadata } from "../../components/models/ModelInfoCard";
import { apiSend, errMsg, fetchModels, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

export default function ExtraPage() {
  const { t } = useI18n();
  const [audio, setAudio] = useState<File | null>(null);
  const [audios, setAudios] = useState<string[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [method, setMethod] = useState("rmvpe");
  const [pth, setPth] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [f0Job, setF0Job] = useState<string | null>(null);
  const [inspectData, setInspectData] = useState<ModelMetadata | null>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState("");
  const [inspectedPth, setInspectedPth] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setAudios(m.audios);
        setModels(m.models);
        if (m.audios.length > 0) setInputPath(m.audios[0]);
        if (m.models.length > 0) setPth(m.models[0]);
      })
      .catch(() => {});
  }, []);

  function checkAudio(): boolean {
    if (!audio && !inputPath) {
      setError(t("Please select or upload an audio file first."));
      return false;
    }
    return true;
  }

  async function analyze() {
    setError("");
    if (!checkAudio()) return;
    setBusy(true);
    try {
      const fd = new FormData();
      if (audio) fd.append("audio", audio);
      if (inputPath) fd.append("inputPath", inputPath);
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
    setInspectError("");
    if (!pth.trim()) {
      setInspectError(t("Please enter or select a .pth model path."));
      return;
    }
    setInspectLoading(true);
    setInspectData(null);
    setInspectedPth(pth.trim());
    try {
      const res = await apiSend<{ ok: boolean; metadata: ModelMetadata }>("/api/models/inspect", "POST", {
        pthPath: pth.trim(),
      });
      setInspectData(res.metadata);
    } catch (e) {
      setInspectError(errMsg(e));
    } finally {
      setInspectLoading(false);
    }
  }

  async function f0() {
    setError("");
    if (!checkAudio()) return;
    try {
      const fd = new FormData();
      if (audio) fd.append("audio", audio);
      if (inputPath) fd.append("inputPath", inputPath);
      fd.append("method", method);
      const { jobId: id } = await postForm<{ jobId: string }>("/api/extra/f0", fd);
      setF0Job(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  const previewUrl = audio ? URL.createObjectURL(audio) : inputPath ? `/${inputPath}` : null;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Extra Tools")}
        description={t(
          "Inspect acoustic waveforms, plot frequency spectrograms, extract pitch contours, and examine model checkpoints.",
        )}
      />

      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-sm"
        >
          {error}
        </div>
      )}

      {/* Shared Audio Input Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AudioWaveform size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Input Audio Source")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("This audio file will be analyzed by both the Audio Analyzer and the F0 Curve Extractor.")}
          </p>
        </div>

        <div className="grid2">
          <div>
            <label htmlFor="extra-audio-file">{t("Upload local audio file")}</label>
            <input
              id="extra-audio-file"
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.alac,.wma,.aiff,.webm,.ac3"
              onChange={(e) => {
                setAudio(e.target.files?.[0] || null);
                if (e.target.files?.[0]) setInputPath("");
              }}
            />
          </div>
          <div>
            <label htmlFor="extra-audio-path">{t("…or pick from assets/audios")}</label>
            <input
              id="extra-audio-path"
              type="text"
              list="ext-audios"
              value={inputPath}
              onChange={(e) => {
                setInputPath(e.target.value);
                if (e.target.value) setAudio(null);
              }}
              placeholder="assets/audios/input.wav"
            />
            <datalist id="ext-audios">
              {audios.map((a) => (
                <option key={a} value={a} />
              ))}
            </datalist>
          </div>
        </div>

        {/* Audio Preview Player */}
        {previewUrl && (
          <div className="mt-4 pt-3 border-t border-white/10">
            <span className="text-xs text-neutral-400 block mb-1 font-medium">
              {t("Source Audio Preview:")}
            </span>
            <AudioPlayer src={previewUrl} title={audio?.name || inputPath} showAnalyzerLink={false} />
          </div>
        )}
      </div>

      {/* Grid: Analyzer & F0 Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Tool 1: Audio Analyzer */}
        <div className="card space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">{t("Audio Analyzer")}</h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t(
                  "Generates a full 3-panel acoustic plot containing: Spectrogram (frequency vs time), Waveform amplitude envelope, and Spectral Centroid/Bandwidth/Rolloff features.",
                )}
              </p>
            </div>
          </div>

          <div className="pt-3 border-t border-white/5">
            <button
              type="button"
              className="cta w-full h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl"
              onClick={analyze}
              disabled={busy}
            >
              <Activity size={16} className="shrink-0" />
              <span>{busy ? t("Generating Spectrogram…") : t("Generate Spectrogram & Analysis")}</span>
            </button>
          </div>
        </div>

        {/* Tool 2: F0 Curve Extractor */}
        <div className="card space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <LineChart size={18} className="text-white shrink-0" />
                  <h2 className="text-base font-bold text-white m-0">{t("F0 Pitch Curve Extractor")}</h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t(
                  "Extracts frame-by-frame fundamental pitch frequencies (Hz) across time and exports both a high-resolution plot and a CSV data curve.",
                )}
              </p>
            </div>

            <div className="max-w-md">
              <label htmlFor="extra-f0-method">{t("Extraction Method")}</label>
              <select id="extra-f0-method" value={method} onChange={(e) => setMethod(e.target.value)}>
                {["rmvpe", "fcpe", "crepe"].map((m) => (
                  <option key={m} value={m}>
                    {m.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="pt-3 border-t border-white/5">
            <button
              type="button"
              className="cta w-full h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl"
              onClick={f0}
            >
              <LineChart size={16} className="shrink-0" />
              <span>{t("Extract F0 Curve")}</span>
            </button>
          </div>
        </div>
      </div>

      <AnalysisResultCard jobId={jobId} title={t("Acoustic Spectrogram Analysis")} type="analyzer" />
      <AnalysisResultCard jobId={f0Job} title={t("Fundamental Pitch Contour (F0)")} type="f0" />

      {/* Tool 3: Model Checkpoint Inspector */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Info size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Model Checkpoint Inspector")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t(
              "Inspect any .pth file directly to display training epochs, author, vocoder, sampling rate, and hash.",
            )}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 max-w-xl">
          <label htmlFor="extra-model-path" className="sr-only">
            {t("Path to .pth checkpoint")}
          </label>
          <input
            id="extra-model-path"
            type="text"
            list="ext-models"
            value={pth}
            onChange={(e) => setPth(e.target.value)}
            placeholder="logs/my-model/my-model.pth"
            className="flex-1 h-10 px-3 text-sm rounded-xl bg-white/5 border border-white/10"
          />
          <datalist id="ext-models">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <button
            type="button"
            className="cta h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl shrink-0"
            onClick={modelInfo}
          >
            <Info size={16} className="shrink-0" />
            <span>{t("Inspect Checkpoint")}</span>
          </button>
        </div>
      </div>

      <ModelInfoCard
        metadata={inspectData}
        loading={inspectLoading}
        error={inspectError}
        pthPath={inspectedPth}
      />
    </div>
  );
}
