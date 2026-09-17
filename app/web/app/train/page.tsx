"use client";

import { FolderUp, Layers, StopCircle, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, errMsg, submitJob } from "../../lib/api";

type TrainMode = "pipeline" | "steps" | "uploads";

export default function TrainPage() {
  const [trainMode, setTrainMode] = useState<TrainMode>("pipeline");
  const [modelName, setModelName] = useState("my-project");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [pretG, setPretG] = useState<string[]>([]);
  const [pretD, setPretD] = useState<string[]>([]);
  const [gpuInfo, setGpuInfo] = useState("");
  const [gpuCount, setGpuCount] = useState("0");

  const [datasetPath, setDatasetPath] = useState("");
  const [sampleRate, setSampleRate] = useState("40000");
  const [cut, setCut] = useState("Automatic");
  const [chunk, setChunk] = useState(3.0);
  const [overlap, setOverlap] = useState(0.3);
  const [noiseReduction, setNoiseReduction] = useState(false);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedder, setEmbedder] = useState("contentvec");
  const [vocoder, setVocoder] = useState("HiFi-GAN");
  const [totalEpoch, setTotalEpoch] = useState(200);
  const [batchSize, setBatchSize] = useState(4);
  const [saveEvery, setSaveEvery] = useState(10);
  const [indexAlgo, setIndexAlgo] = useState("Auto");
  const [customPre, setCustomPre] = useState(false);
  const [gPath, setGPath] = useState("");
  const [dPath, setDPath] = useState("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopTarget, setStopTarget] = useState("");

  useEffect(() => {
    apiGet<{ datasets: string[] }>("/api/train/datasets")
      .then((d) => {
        setDatasets(d.datasets);
        if (d.datasets[0]) setDatasetPath(d.datasets[0]);
      })
      .catch(() => {});
    apiGet<{ g: string[]; d: string[] }>("/api/train/pretraineds")
      .then((p) => {
        setPretG(p.g);
        setPretD(p.d);
      })
      .catch(() => {});
    apiGet<{ count: number; info: string }>("/api/train/gpus")
      .then((g) => {
        setGpuInfo(g.info);
        setGpuCount(g.count > 0 ? "0" : "-");
      })
      .catch(() => setGpuInfo("GPU query failed (CPU-only host)"));
  }, []);

  async function run(path: string, body: unknown) {
    setError("");
    setBusy(true);
    try {
      const { jobId: id } = await submitJob(path, body);
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function runPipeline() {
    if (!modelName.trim()) {
      setError("Please enter a model name.");
      return;
    }
    if (!datasetPath) {
      setError("Please select or upload a dataset.");
      return;
    }
    setError("");
    setBusy(true);
    try {
      const { jobId: id } = await submitJob("/api/train/pipeline", {
        modelName: modelName.trim(),
        datasetPath,
        sampleRate,
        f0Method,
        embedderModel: embedder,
        vocoder,
        totalEpoch,
        batchSize,
        saveEveryEpoch: saveEvery,
        gpu: gpuCount,
        indexAlgorithm: indexAlgo,
        noiseReduction,
      });
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setError("");
    try {
      await submitJob("/api/train/stop", jobId ? { jobId } : { modelName: stopTarget || modelName });
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  }

  return (
    <div>
      <PageHeader
        title="Training Studio"
        description="Train custom RVC voice models from audio datasets with automated 1-click pipeline or step-by-step control."
      >
        <div className="row">
          <button
            type="button"
            className={
              trainMode === "pipeline" ? "cta flex items-center gap-1.5" : "ghost flex items-center gap-1.5"
            }
            onClick={() => setTrainMode("pipeline")}
          >
            <Zap size={14} />
            <span>1-Click Pipeline</span>
          </button>
          <button
            type="button"
            className={
              trainMode === "steps" ? "cta flex items-center gap-1.5" : "ghost flex items-center gap-1.5"
            }
            onClick={() => setTrainMode("steps")}
          >
            <Layers size={14} />
            <span>Step-by-Step</span>
          </button>
          <button
            type="button"
            className={
              trainMode === "uploads" ? "cta flex items-center gap-1.5" : "ghost flex items-center gap-1.5"
            }
            onClick={() => setTrainMode("uploads")}
          >
            <FolderUp size={14} />
            <span>Uploads</span>
          </button>
        </div>
      </PageHeader>

      {/* Global Model Name & Hardware Config Bar */}
      <div className="card mb-4">
        <div className="grid2">
          <div>
            <label>Model Project Name</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. vocal-model"
            />
          </div>
          <div>
            <label>Compute Hardware (GPU)</label>
            <input
              type="text"
              value={gpuCount}
              onChange={(e) => setGpuCount(e.target.value)}
              placeholder="0 (or - for CPU)"
            />
          </div>
        </div>
        <div className="flex items-center justify-between text-xs text-neutral-400 mt-2">
          <span>{gpuInfo || "Detecting GPU acceleration…"}</span>
          <span className="text-neutral-500">
            Output saved to <code>logs/{modelName || "…"}/</code>
          </span>
        </div>
        {error && (
          <p className="mt-2" style={{ color: "var(--err)" }}>
            {error}
          </p>
        )}
      </div>

      {/* 1. AUTOMATED 1-CLICK PIPELINE VIEW */}
      {trainMode === "pipeline" && (
        <div className="space-y-4">
          <div className="card border border-white/20">
            <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-3 mb-4">
              <div>
                <h2 className="text-lg font-bold text-white m-0 flex items-center gap-2">
                  <Zap size={18} className="text-amber-400" />
                  <span>1-Click Complete Pipeline</span>
                </h2>
                <p className="text-xs text-neutral-400 m-0 mt-0.5">
                  Runs Preprocess, Feature Extraction, Model Training, and Feature Indexing in a single
                  automated flow.
                </p>
              </div>
            </div>

            {/* Stepper overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-6">
              {[
                { step: "1", name: "Preprocess", desc: "Slice & normalize audio" },
                { step: "2", name: "Extract", desc: "F0 pitch & embeddings" },
                { step: "3", name: "Train", desc: "Generator & Discriminator" },
                { step: "4", name: "Index", desc: "Faiss feature retrieval" },
              ].map((s) => (
                <div key={s.step} className="bg-white/5 border border-white/5 rounded-lg p-3">
                  <span className="text-xs font-bold text-neutral-400 block">Step {s.step}</span>
                  <span className="text-sm font-semibold text-white block">{s.name}</span>
                  <span className="text-xs text-neutral-500 block">{s.desc}</span>
                </div>
              ))}
            </div>

            <div className="grid2">
              <div>
                <label>Dataset Folder (in assets/datasets)</label>
                <input
                  type="text"
                  list="datasets"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  placeholder="assets/datasets/my_vocal_data"
                />
                <datalist id="datasets">
                  {datasets.map((d) => (
                    <option key={d} value={d} />
                  ))}
                </datalist>
              </div>

              <div>
                <label>Target Sampling Rate</label>
                <select value={sampleRate} onChange={(e) => setSampleRate(e.target.value)}>
                  {["32000", "40000", "48000"].map((s) => (
                    <option key={s} value={s}>
                      {s} Hz
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label>Total Epochs: {totalEpoch}</label>
                <input
                  type="range"
                  min={10}
                  max={1000}
                  step={10}
                  value={totalEpoch}
                  onChange={(e) => setTotalEpoch(Number(e.target.value))}
                />
              </div>

              <div>
                <label>Batch Size: {batchSize}</label>
                <input
                  type="range"
                  min={1}
                  max={32}
                  step={1}
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                />
              </div>

              <div>
                <label>Pitch Extraction (F0)</label>
                <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
                  {["rmvpe", "crepe", "crepe-tiny"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label>Vocoder Architecture</label>
                <select value={vocoder} onChange={(e) => setVocoder(e.target.value)}>
                  {["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex items-center gap-4 mt-4 pt-3 border-t border-white/10">
              <label className="flex items-center gap-2 cursor-pointer m-0">
                <input
                  type="checkbox"
                  checked={noiseReduction}
                  onChange={(e) => setNoiseReduction(e.target.checked)}
                />
                <span>Enable Audio Noise Reduction</span>
              </label>
            </div>

            <div className="row mt-5 pt-3 border-t border-white/10">
              <button
                type="button"
                className="cta flex items-center gap-2 px-6 py-2.5 text-base"
                disabled={busy}
                onClick={runPipeline}
              >
                <Zap size={16} />
                <span>{busy ? "Pipeline Running…" : "Start 1-Click Pipeline"}</span>
              </button>

              {busy && (
                <button
                  type="button"
                  className="ghost text-red-400 hover:text-red-300 flex items-center gap-1.5"
                  onClick={stop}
                >
                  <StopCircle size={16} />
                  <span>Stop Pipeline</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 2. STEP-BY-STEP TRAINING VIEW */}
      {trainMode === "steps" && (
        <div className="space-y-4">
          {/* Step 1: Preprocess */}
          <div className="card">
            <h2>1 · Preprocess Dataset</h2>
            <div className="grid2">
              <div>
                <label>Dataset (assets/datasets)</label>
                <input
                  type="text"
                  list="datasets"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                />
                <datalist id="datasets">
                  {datasets.map((d) => (
                    <option key={d} value={d} />
                  ))}
                </datalist>
              </div>
              <div>
                <label>Sample Rate</label>
                <select value={sampleRate} onChange={(e) => setSampleRate(e.target.value)}>
                  {["32000", "40000", "48000"].map((s) => (
                    <option key={s} value={s}>
                      {s} Hz
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Cut Method</label>
                <select value={cut} onChange={(e) => setCut(e.target.value)}>
                  {["Skip", "Simple", "Automatic"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>
                  Chunk {chunk}s · Overlap {overlap}s
                </label>
                <div className="row">
                  <input
                    type="range"
                    min={0.5}
                    max={5}
                    step={0.1}
                    value={chunk}
                    onChange={(e) => setChunk(Number(e.target.value))}
                  />
                  <input
                    type="range"
                    min={0}
                    max={0.4}
                    step={0.1}
                    value={overlap}
                    onChange={(e) => setOverlap(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>
            <label className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                type="checkbox"
                checked={noiseReduction}
                onChange={(e) => setNoiseReduction(e.target.checked)}
              />
              <span>Noise reduction</span>
            </label>
            <div className="row mt-4">
              <button
                type="button"
                className="cta"
                disabled={busy}
                onClick={() =>
                  run("/api/train/preprocess", {
                    modelName,
                    datasetPath,
                    sampleRate,
                    cutPreprocess: cut,
                    chunkLen: chunk,
                    overlapLen: overlap,
                    noiseReduction,
                  })
                }
              >
                Run Preprocess
              </button>
            </div>
          </div>

          {/* Step 2: Feature Extraction */}
          <div className="card">
            <h2>2 · Extract Features</h2>
            <div className="grid2">
              <div>
                <label>Pitch Method (F0)</label>
                <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
                  {["crepe", "crepe-tiny", "rmvpe"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Embedder Model</label>
                <select value={embedder} onChange={(e) => setEmbedder(e.target.value)}>
                  {["contentvec", "spin-v2", "custom"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="row mt-4">
              <button
                type="button"
                className="cta"
                disabled={busy}
                onClick={() =>
                  run("/api/train/extract", {
                    modelName,
                    f0Method,
                    gpu: gpuCount,
                    sampleRate,
                    embedderModel: embedder,
                  })
                }
              >
                Run Extract
              </button>
            </div>
          </div>

          {/* Step 3: Train */}
          <div className="card">
            <h2>3 · Model Training</h2>
            <div className="grid2">
              <div>
                <label>Vocoder</label>
                <select value={vocoder} onChange={(e) => setVocoder(e.target.value)}>
                  {["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Total Epochs: {totalEpoch}</label>
                <input
                  type="range"
                  min={1}
                  max={1000}
                  step={1}
                  value={totalEpoch}
                  onChange={(e) => setTotalEpoch(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Batch Size: {batchSize}</label>
                <input
                  type="range"
                  min={1}
                  max={64}
                  step={1}
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Save Every N Epochs: {saveEvery}</label>
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={saveEvery}
                  onChange={(e) => setSaveEvery(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Index Algorithm</label>
                <select value={indexAlgo} onChange={(e) => setIndexAlgo(e.target.value)}>
                  {["Auto", "Faiss", "KMeans"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer mt-3">
              <input type="checkbox" checked={customPre} onChange={(e) => setCustomPre(e.target.checked)} />
              <span>Custom pretrained G/D</span>
            </label>
            {customPre && (
              <div className="grid2 mt-2">
                <div>
                  <label>G path</label>
                  <input type="text" list="preG" value={gPath} onChange={(e) => setGPath(e.target.value)} />
                  <datalist id="preG">
                    {pretG.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </div>
                <div>
                  <label>D path</label>
                  <input type="text" list="preD" value={dPath} onChange={(e) => setDPath(e.target.value)} />
                  <datalist id="preD">
                    {pretD.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </div>
              </div>
            )}

            <div className="row mt-4">
              <button
                type="button"
                className="cta"
                disabled={busy}
                onClick={() =>
                  run("/api/train/train", {
                    modelName,
                    vocoder,
                    totalEpoch,
                    batchSize,
                    saveEveryEpoch: saveEvery,
                    gpu: gpuCount,
                    sampleRate,
                    indexAlgorithm: indexAlgo,
                    customPretrained: customPre,
                    gPretrainedPath: gPath || undefined,
                    dPretrainedPath: dPath || undefined,
                  })
                }
              >
                Start Training
              </button>
              <button
                type="button"
                className="ghost"
                onClick={() => run("/api/train/index", { modelName, indexAlgorithm: indexAlgo })}
              >
                Generate Index Only
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 3. UPLOADS VIEW */}
      {trainMode === "uploads" && (
        <div className="card space-y-4">
          <h2>Dataset & Checkpoint Uploads</h2>
          <UploadBox
            path="/api/train/upload-dataset"
            fields={[{ name: "datasetName", label: "Dataset name (e.g. my_vocals)" }]}
            files="files"
            multiple
            label="Dataset Audio Files (WAV/MP3/FLAC) → assets/datasets/<name>/"
          />
          <UploadBox
            path="/api/train/upload-pretrained"
            fields={[]}
            files="file"
            label="Custom Pretrained Weights (.pth) → rvc/models/pretraineds/custom/"
          />
          <UploadBox
            path="/api/train/upload-embedder"
            fields={[{ name: "folderName", label: "Folder name" }]}
            files="bin"
            extra="config"
            label="Custom Embedder (.bin + .json)"
          />
        </div>
      )}

      {/* Stop Controller Card */}
      <div className="card mt-4">
        <h2>Stop Training Process</h2>
        <div className="row">
          <input
            type="text"
            placeholder="model name (fallback)"
            value={stopTarget}
            onChange={(e) => setStopTarget(e.target.value)}
            style={{ maxWidth: 240 }}
          />
          <button type="button" className="ghost text-red-400 hover:text-red-300" onClick={stop}>
            Stop Running Job
          </button>
        </div>
      </div>

      <JobPanel jobId={jobId} />
    </div>
  );
}

function UploadBox({
  path,
  fields,
  files,
  extra,
  multiple,
  label,
}: {
  path: string;
  fields: Array<{ name: string; label: string }>;
  files: string;
  extra?: string;
  multiple?: boolean;
  label: string;
}) {
  const [vals, setVals] = useState<Record<string, string>>({});
  const [picked, setPicked] = useState<FileList | null>(null);
  const [picked2, setPicked2] = useState<FileList | null>(null);
  const [msg, setMsg] = useState("");
  async function send() {
    setMsg("");
    const fd = new FormData();
    for (const f of fields) fd.append(f.name, vals[f.name] || "");
    if (picked) for (const f of Array.from(picked)) fd.append(files, f);
    if (extra && picked2) for (const f of Array.from(picked2)) fd.append(extra, f);
    try {
      const r = await fetch(path, { method: "POST", body: fd });
      const b = await r.json();
      if (!r.ok) throw new Error(b?.error || "Upload failed");
      setMsg("Uploaded ✓");
    } catch (e) {
      setMsg(errMsg(e));
    }
  }
  return (
    <div className="bg-white/5 border border-white/5 rounded-lg p-3">
      <p className="text-xs font-medium text-neutral-300 mb-2">{label}</p>
      <div className="row flex-wrap gap-2">
        {fields.map((f) => (
          <input
            key={f.name}
            type="text"
            placeholder={f.label}
            value={vals[f.name] || ""}
            onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}
            style={{ maxWidth: 200 }}
          />
        ))}
        <input type="file" multiple={multiple} onChange={(e) => setPicked(e.target.files)} />
        {extra && <input type="file" onChange={(e) => setPicked2(e.target.files)} />}
        <button type="button" className="ghost text-xs" onClick={send}>
          Upload
        </button>
        <span className="text-xs text-emerald-400">{msg}</span>
      </div>
    </div>
  );
}
