"use client";

import { Activity, FileAudio, Info, LineChart } from "lucide-react";
import { useEffect, useState } from "react";
import AudioPlayer from "../../components/AudioPlayer";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import { errMsg, fetchModels, postForm, submitJob } from "../../lib/api";
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
  const [infoJob, setInfoJob] = useState<string | null>(null);
  const [f0Job, setF0Job] = useState<string | null>(null);
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
    if (!pth.trim()) {
      setError("Please specify a .pth model path.");
      return;
    }
    try {
      const { jobId: id } = await submitJob("/api/extra/model-info", { pthPath: pth });
      setInfoJob(id);
    } catch (e) {
      setError(errMsg(e));
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
    <div className="space-y-4">
      <PageHeader
        title={t("Audio Studio Tools")}
        description={t(
          "Inspect acoustic waveforms, plot frequency spectrograms, extract pitch contours, and examine model checkpoints.",
        )}
      />

      {error && <p style={{ color: "var(--err)" }}>{error}</p>}

      {/* Shared Audio Input Card */}
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <FileAudio size={18} className="text-white" />
          <h2 className="text-base font-bold text-white m-0">{t("Input Audio Source")}</h2>
        </div>
        <p className="text-xs text-neutral-400 m-0 mb-3">
          {t("This audio file will be analyzed by both the Audio Analyzer and the F0 Curve Extractor.")}
        </p>

        <div className="grid2">
          <div>
            <label>{t("Upload local audio file")}</label>
            <input
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.alac,.wma,.aiff,.webm,.ac3"
              onChange={(e) => {
                setAudio(e.target.files?.[0] || null);
                if (e.target.files?.[0]) setInputPath("");
              }}
            />
          </div>
          <div>
            <label>{t("…or pick from assets/audios")}</label>
            <input
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
        <div className="card flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Activity size={18} className="text-emerald-400" />
              <h2 className="text-base font-bold text-white m-0">{t("Audio Analyzer")}</h2>
            </div>
            <p className="text-xs text-neutral-400 m-0 mb-4">
              {t(
                "Generates a full 3-panel acoustic plot containing: Spectrogram (frequency vs time), Waveform amplitude envelope, and Spectral Centroid/Bandwidth/Rolloff features.",
              )}
            </p>
          </div>

          <div className="pt-3 border-t border-white/10">
            <button type="button" className="cta w-full" onClick={analyze} disabled={busy}>
              {busy ? t("Generating Spectrogram…") : t("Generate Spectrogram & Analysis")}
            </button>
          </div>
        </div>

        {/* Tool 2: F0 Curve Extractor */}
        <div className="card flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <LineChart size={18} className="text-amber-400" />
              <h2 className="text-base font-bold text-white m-0">{t("F0 Pitch Curve Extractor")}</h2>
            </div>
            <p className="text-xs text-neutral-400 m-0 mb-3">
              {t(
                "Extracts frame-by-frame fundamental pitch frequencies (Hz) across time and exports both a high-resolution plot and a CSV data curve.",
              )}
            </p>

            <div className="mb-4">
              <label>{t("Extraction Method")}</label>
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {["rmvpe", "fcpe", "crepe"].map((m) => (
                  <option key={m} value={m}>
                    {m.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="pt-3 border-t border-white/10">
            <button type="button" className="cta w-full" onClick={f0}>
              {t("Extract F0 Curve")}
            </button>
          </div>
        </div>
      </div>

      <JobPanel jobId={jobId} />
      <JobPanel jobId={f0Job} />

      {/* Tool 3: Model Checkpoint Inspector */}
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <Info size={18} className="text-blue-400" />
          <h2 className="text-base font-bold text-white m-0">{t("Model Checkpoint Inspector")}</h2>
        </div>
        <p className="text-xs text-neutral-400 m-0 mb-3">
          {t(
            "Inspect any .pth file directly to display training epochs, author, vocoder, sampling rate, and hash.",
          )}
        </p>
        <div className="row">
          <input
            type="text"
            list="ext-models"
            value={pth}
            onChange={(e) => setPth(e.target.value)}
            placeholder="logs/my-model/my-model.pth"
            style={{ flex: 1 }}
          />
          <datalist id="ext-models">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <button type="button" className="cta" onClick={modelInfo}>
            {t("Inspect Checkpoint")}
          </button>
        </div>
      </div>
      <JobPanel jobId={infoJob} />
    </div>
  );
}
