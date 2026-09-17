"use client";

import { useEffect, useState } from "react";
import { errMsg, fetchJob, fetchModels, type Job, pollJob, submitInference } from "../lib/api";

const F0_METHODS = ["crepe", "crepe-tiny", "rmvpe", "fcpe"];
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

export default function InferenceForm() {
  const [models, setModels] = useState<string[]>([]);
  const [indexes, setIndexes] = useState<string[]>([]);
  const [audios, setAudios] = useState<string[]>([]);
  const [loadError, setLoadError] = useState("");

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [inputPath, setInputPath] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [protect, setProtect] = useState(0.33);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [terms, setTerms] = useState(false);

  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  // Preset apply/save bridge (PresetsPanel dispatches / requests these events)
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
    const onRequest = () => {
      window.dispatchEvent(
        new CustomEvent("applio:read-preset", {
          detail: { pitch, index_rate: indexRate, rms_mix_rate: volumeEnvelope, protect },
        }),
      );
    };
    window.addEventListener("applio:apply-preset", onApply);
    window.addEventListener("applio:request-preset", onRequest);
    return () => {
      window.removeEventListener("applio:apply-preset", onApply);
      window.removeEventListener("applio:request-preset", onRequest);
    };
  }, [indexRate, pitch, protect, volumeEnvelope]);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        setIndexes(m.indexes);
        setAudios(m.audios);
        if (m.models.length > 0) {
          setPthPath(m.models[0]);
          const stem = (m.models[0].split("/").pop() ?? "").replace(/\.(pth|onnx)$/, "");
          const match = m.indexes.find((i) => i.toLowerCase().includes(stem.toLowerCase().slice(0, 8)));
          if (match) setIndexPath(match);
        }
        if (m.audios.length > 0) setInputPath(m.audios[0]);
      })
      .catch((e) => setLoadError(errMsg(e)));
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: polling is keyed by job id on purpose
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const stop = pollJob(job.id, setJob);
    return stop;
  }, [job?.id]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError("");
    if (!terms) {
      setSubmitError("You must agree to the Terms of Use to proceed.");
      return;
    }
    if (!pthPath) {
      setSubmitError("Select a voice model (.pth).");
      return;
    }
    if (!audioFile && !inputPath) {
      setSubmitError("Upload an audio file or pick one from assets/audios.");
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
    fd.append("exportFormat", exportFormat);
    fd.append("splitAudio", String(splitAudio));
    fd.append("f0Autotune", String(f0Autotune));
    fd.append("cleanAudio", String(cleanAudio));
    setSubmitting(true);
    try {
      const { jobId } = await submitInference(fd);
      const { job: fresh } = await fetchJob(jobId);
      setJob(fresh);
    } catch (err) {
      setSubmitError(errMsg(err) || "Submit failed");
    } finally {
      setSubmitting(false);
    }
  }

  const directAudioUrl = job?.outputFile ? `/outputs/${job.outputFile.split("/").pop()}` : null;

  return (
    <form onSubmit={onSubmit}>
      {loadError && (
        <div className="card">
          <strong>API offline.</strong>{" "}
          <span className="muted">
            Start it with <code>npm run dev</code>. {loadError}
          </span>
        </div>
      )}

      <div className="flex gap-4 items-start">
        <div className="flex flex-col w-[300px] min-w-[300px]">
          <div className="card">
            <h2>Model Selection</h2>
            <div className="grid2">
              <div>
                <label>Voice Model (.pth)</label>
                <input
                  type="text"
                  list="models"
                  value={pthPath}
                  onChange={(e) => setPthPath(e.target.value)}
                  placeholder="logs/my-model/model.pth"
                />
                <datalist id="models">
                  {models.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>
              <div>
                <label>Index File (.index, optional)</label>
                <input
                  type="text"
                  list="indexes"
                  value={indexPath}
                  onChange={(e) => setIndexPath(e.target.value)}
                  placeholder="logs/my-model/added.index"
                />
                <datalist id="indexes">
                  {indexes.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>
            </div>
            <p className="muted">
              Models are discovered by walking <code>logs/</code>, skipping <code>G_*/D_*</code>. Custom
              absolute paths allowed.
            </p>
          </div>

          <div className="card">
            <h2>Audio Input</h2>
            <label>Upload audio (wav/mp3/flac/ogg/m4a, max 200MB)</label>
            <input
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.aiff,.webm"
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
            />
            <label>…or pick a file already in assets/audios</label>
            <input
              type="text"
              list="audios"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="assets/audios/input.wav"
            />
            <datalist id="audios">
              {audios.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
        </div>
        <div className="flex-1 flex flex-col min-w-0">
          <div className="card">
            <h2>Conversion Settings</h2>
            <div className="grid2">
              <div>
                <label>Pitch: {pitch} (range -24…24, default 0)</label>
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
                <label>Search Feature Ratio: {indexRate} (default 0.75 in UI)</label>
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
                <label>Volume Envelope: {volumeEnvelope}</label>
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
                <label>Protect Voiceless Consonants: {protect}</label>
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
                <label>Pitch extraction algorithm</label>
                <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
                  {F0_METHODS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Embedder Model</label>
                <select value={embedderModel} onChange={(e) => setEmbedderModel(e.target.value)}>
                  {EMBEDDERS.map((m) => (
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
            <details>
              <summary>Advanced (split / autotune / clean)</summary>
              <div className="row">
                <label>
                  <input
                    type="checkbox"
                    checked={splitAudio}
                    onChange={(e) => setSplitAudio(e.target.checked)}
                  />{" "}
                  Split Audio
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={f0Autotune}
                    onChange={(e) => setF0Autotune(e.target.checked)}
                  />{" "}
                  Autotune
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={cleanAudio}
                    onChange={(e) => setCleanAudio(e.target.checked)}
                  />{" "}
                  Clean Audio
                </label>
              </div>
              <p className="muted">
                Full post-process chain (reverb, chorus, compressor, …) is accepted by{" "}
                <code>POST /api/inference</code> — UI controls land with the Extra/Presets migration. See{" "}
                <code>server/src/schemas.ts</code>.
              </p>
            </details>
          </div>

          <div className="card">
            <h2>Conversion</h2>
            <label className="terms">
              <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
              <span>I agree to the terms of use (inference is blocked until accepted).</span>
            </label>
            <div className="row" style={{ marginTop: 12 }}>
              <button type="submit" className="cta" disabled={submitting}>
                {submitting ? "Submitting…" : "Convert"}
              </button>
              {job && <span className={`badge ${job.status}`}>{job.status}</span>}
              {job && <span className="muted">job {job.id}</span>}
            </div>
            {submitError && <p style={{ color: "var(--err)" }}>{submitError}</p>}
            {job && directAudioUrl && job.status === "done" && (
              <div>
                {/* biome-ignore lint/a11y/useMediaCaption: converted audio has no caption track */}
                <audio controls src={directAudioUrl} />
                <p>
                  <a href={directAudioUrl} download>
                    Download output
                  </a>{" "}
                  <span className="muted">{job.outputFile}</span>
                </p>
              </div>
            )}
            {job && job.status === "error" && <p style={{ color: "var(--err)" }}>{job.error}</p>}
            {job && job.logs.length > 0 && (
              <div>
                <p className="muted">Engine logs (streams from core.py subprocess)</p>
                <div className="log">{job.logs.slice(-60).join("\n")}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </form>
  );
}
