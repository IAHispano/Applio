"use client";

import { Activity, Cpu, Download, Flame, FolderUp, Layers, Sliders, StopCircle, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import TrainingConsole from "../../components/train/TrainingConsole";
import SegmentedControl from "../../components/ui/SegmentedControl";
import SliderField from "../../components/ui/SliderField";
import { apiGet, errMsg, submitJob } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { toast } from "../../lib/toast";

type TrainMode = "pipeline" | "steps" | "uploads";

export default function TrainPage() {
  const { t } = useI18n();
  const [trainMode, setTrainMode] = useState<TrainMode>("pipeline");
  const [modelName, setModelName] = useState("my-project");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [pretG, setPretG] = useState<string[]>([]);
  const [pretD, setPretD] = useState<string[]>([]);
  const [gpuInfo, setGpuInfo] = useState("");
  const [gpuCount, setGpuCount] = useState("0");

  const [datasetPath, setDatasetPath] = useState("");
  const [sampleRate, setSampleRate] = useState("40000");
  const [cpuCores, setCpuCores] = useState("");
  const [cut, setCut] = useState("Automatic");
  const [chunk, setChunk] = useState(3.0);
  const [overlap, setOverlap] = useState(0.3);
  const [noiseReduction, setNoiseReduction] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.7);
  const [processEffects, setProcessEffects] = useState(false);
  const [normalizationMode, setNormalizationMode] = useState("none");
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedder, setEmbedder] = useState("contentvec");
  const [embedderCustom, setEmbedderCustom] = useState("");
  const [includeMutes, setIncludeMutes] = useState(2);
  const [vocoder, setVocoder] = useState("HiFi-GAN");
  const [totalEpoch, setTotalEpoch] = useState(200);
  const [batchSize, setBatchSize] = useState(4);
  const [saveEvery, setSaveEvery] = useState(10);
  const [pretrained, setPretrained] = useState(true);
  const [saveOnlyLatest, setSaveOnlyLatest] = useState(true);
  const [saveEveryWeights, setSaveEveryWeights] = useState(true);
  const [cleanup, setCleanup] = useState(false);
  const [cacheGpu, setCacheGpu] = useState(false);
  const [checkpointing, setCheckpointing] = useState(false);
  const [indexAlgo, setIndexAlgo] = useState("Auto");
  const [customPre, setCustomPre] = useState(false);
  const [gPath, setGPath] = useState("");
  const [dPath, setDPath] = useState("");
  const [expModels, setExpModels] = useState<string[]>([]);
  const [expIndexes, setExpIndexes] = useState<string[]>([]);
  const [expModel, setExpModel] = useState("");
  const [expIndex, setExpIndex] = useState("");

  const srOptions = vocoder === "RefineGAN" ? ["24000", "32000"] : ["32000", "40000", "48000"];

  function pickVocoder(v: string) {
    setVocoder(v);
    if (v === "RefineGAN" && (sampleRate === "40000" || sampleRate === "48000")) setSampleRate("32000");
    if (v !== "RefineGAN" && sampleRate === "24000") setSampleRate("40000");
  }

  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopTarget, setStopTarget] = useState("");

  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-time fetch only; t is a stable dictionary lookup
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
      .catch(() => setGpuInfo(t("GPU query failed (CPU-only host)")));
    apiGet<{ models: string[]; indexes: string[] }>("/api/train/exports")
      .then((e) => {
        setExpModels(e.models || []);
        setExpIndexes(e.indexes || []);
        if (e.models?.[0]) setExpModel(e.models[0]);
        if (e.indexes?.[0]) setExpIndex(e.indexes[0]);
      })
      .catch(() => {});
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
      setError(t("Please enter a model name."));
      return;
    }
    if (!datasetPath) {
      setError(t("Please select or upload a dataset."));
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

  async function downloadExport(file: string) {
    if (!file) return;
    setError("");
    try {
      const r = await fetch(`/api/train/export-file?file=${encodeURIComponent(file)}`);
      if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.error || t("Download failed"));
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.split(/[\\/]/).pop() || "export";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function stop() {
    try {
      await fetch("/api/train/stop", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ modelName: stopTarget || modelName }),
      });
      toast(t("Training stopped."));
    } catch (e) {
      setError(errMsg(e));
    }
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Training")}
        description={t(
          "Train custom RVC voice models from audio datasets with automated 1-click pipeline or step-by-step control.",
        )}
      >
        <SegmentedControl
          value={trainMode}
          onChange={setTrainMode}
          ariaLabel={t("Training mode")}
          tabPanels
          options={[
            { value: "pipeline", label: t("1-Click Pipeline"), icon: Zap },
            { value: "steps", label: t("Step-by-Step"), icon: Layers },
            { value: "uploads", label: t("Uploads"), icon: FolderUp },
          ]}
        />
      </PageHeader>

      {/* Global Model Name & Hardware Config Bar */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">
                {t("Model & Compute Hardware")}
              </h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Define project identity and target compute device configuration.")}
          </p>
        </div>
        <div className="grid2">
          <div>
            <label htmlFor="train-model-name">{t("Model Project Name")}</label>
            <input
              id="train-model-name"
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder={t("e.g. vocal-model")}
            />
          </div>
          <div>
            <label htmlFor="train-gpu-count">{t("Compute Hardware (GPU)")}</label>
            <input
              id="train-gpu-count"
              type="text"
              value={gpuCount}
              onChange={(e) => setGpuCount(e.target.value)}
              placeholder={t("0 (or - for CPU)")}
            />
          </div>
          <div>
            <label htmlFor="train-cpu-cores">{t("CPU Cores")}</label>
            <input
              id="train-cpu-cores"
              type="number"
              min={1}
              max={64}
              value={cpuCores}
              onChange={(e) => setCpuCores(e.target.value)}
              placeholder={t("auto")}
            />
          </div>
        </div>
        <div className="flex items-center justify-between text-xs text-neutral-400 mt-2">
          <span>{gpuInfo || t("Detecting GPU acceleration…")}</span>
          <span className="text-neutral-500">
            {t("Output saved to")} <code>logs/{modelName || "…"}/</code>
          </span>
        </div>
        {error && (
          <div
            role="alert"
            aria-live="assertive"
            className="mt-2 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
          >
            {error}
          </div>
        )}
      </div>

      {/* 1. AUTOMATED 1-CLICK PIPELINE VIEW */}
      {trainMode === "pipeline" && (
        <div id="panel-pipeline" role="tabpanel" aria-labelledby="tab-pipeline" className="space-y-4">
          <div className="card border border-white/20">
            <div className="border-b border-white/10 pb-3.5 space-y-1 mb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("1-Click Complete Pipeline")}
                  </h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t(
                  "Runs Preprocess, Feature Extraction, Model Training, and Feature Indexing in a single automated flow.",
                )}
              </p>
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
                  <span className="text-sm font-semibold text-white block">{t(s.name)}</span>
                  <span className="text-xs text-neutral-500 block">{t(s.desc)}</span>
                </div>
              ))}
            </div>

            <div className="grid2">
              <div>
                <label htmlFor="pipeline-dataset-path">{t("Dataset Folder (in assets/datasets)")}</label>
                <input
                  id="pipeline-dataset-path"
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
                <label htmlFor="pipeline-sample-rate">{t("Target Sampling Rate")}</label>
                <select
                  id="pipeline-sample-rate"
                  value={sampleRate}
                  onChange={(e) => setSampleRate(e.target.value)}
                >
                  {srOptions.map((s) => (
                    <option key={s} value={s}>
                      {s} Hz
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <SliderField
                  id="pipeline-total-epoch"
                  label={t("Total Epochs")}
                  value={totalEpoch}
                  min={10}
                  max={1000}
                  step={10}
                  onChange={setTotalEpoch}
                />
              </div>

              <div>
                <SliderField
                  id="pipeline-batch-size"
                  label={t("Batch Size")}
                  value={batchSize}
                  min={1}
                  max={32}
                  step={1}
                  onChange={setBatchSize}
                />
              </div>

              <div>
                <label htmlFor="pipeline-f0-method">{t("Pitch Extraction (F0)")}</label>
                <select
                  id="pipeline-f0-method"
                  value={f0Method}
                  onChange={(e) => setF0Method(e.target.value)}
                >
                  {["rmvpe", "crepe", "crepe-tiny"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="pipeline-vocoder">{t("Vocoder Architecture")}</label>
                <select id="pipeline-vocoder" value={vocoder} onChange={(e) => pickVocoder(e.target.value)}>
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
                <span>{t("Enable Audio Noise Reduction")}</span>
              </label>
            </div>

            <div className="row mt-5 pt-3 border-t border-white/10">
              <button
                type="button"
                className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
                disabled={busy}
                onClick={runPipeline}
              >
                <Zap size={16} className="shrink-0" />
                <span>{busy ? t("Pipeline Running…") : t("Start 1-Click Pipeline")}</span>
              </button>

              {busy && (
                <button
                  type="button"
                  className="ghost h-10 px-4 text-red-400 hover:text-red-300 border-red-500/30 flex items-center gap-2 text-sm font-medium rounded-xl"
                  onClick={stop}
                >
                  <StopCircle size={16} className="shrink-0" />
                  <span>{t("Stop Pipeline")}</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 2. STEP-BY-STEP TRAINING VIEW */}
      {trainMode === "steps" && (
        <div id="panel-steps" role="tabpanel" aria-labelledby="tab-steps" className="space-y-4">
          {/* Step 1: Preprocess */}
          <div className="card space-y-4">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-white/10 text-xs font-bold text-white flex items-center justify-center">
                    1
                  </span>
                  <Sliders size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Preprocess Dataset")}
                  </h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t("Slice, clean, and normalize raw dataset audio samples for model ingestion.")}
              </p>
            </div>
            <div className="grid2">
              <div>
                <label htmlFor="prep-dataset-path">{t("Dataset (assets/datasets)")}</label>
                <input
                  id="prep-dataset-path"
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
                <label htmlFor="prep-sample-rate">{t("Sample Rate")}</label>
                <select
                  id="prep-sample-rate"
                  value={sampleRate}
                  onChange={(e) => setSampleRate(e.target.value)}
                >
                  {srOptions.map((s) => (
                    <option key={s} value={s}>
                      {s} Hz
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="prep-cut-method">{t("Cut Method")}</label>
                <select id="prep-cut-method" value={cut} onChange={(e) => setCut(e.target.value)}>
                  {["Skip", "Simple", "Automatic"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <SliderField
                  id="prep-chunk"
                  label={t("Chunk length")}
                  value={chunk}
                  min={0.5}
                  max={5}
                  step={0.1}
                  unit="s"
                  onChange={setChunk}
                />
              </div>
              <div>
                <SliderField
                  id="prep-overlap"
                  label={t("Overlap length")}
                  value={overlap}
                  min={0}
                  max={0.4}
                  step={0.1}
                  unit="s"
                  onChange={setOverlap}
                />
              </div>
            </div>
            <label htmlFor="prep-noise-reduction" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="prep-noise-reduction"
                type="checkbox"
                checked={noiseReduction}
                onChange={(e) => setNoiseReduction(e.target.checked)}
              />
              <span>{t("Noise Reduction")}</span>
            </label>
            {noiseReduction && (
              <div className="mt-2">
                <SliderField
                  id="prep-clean-strength"
                  label={t("Clean strength")}
                  value={cleanStrength}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setCleanStrength}
                />
              </div>
            )}
            <label htmlFor="prep-process-effects" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="prep-process-effects"
                type="checkbox"
                checked={processEffects}
                onChange={(e) => setProcessEffects(e.target.checked)}
              />
              <span>{t("Process effects (disable filters during preprocessing)")}</span>
            </label>
            <div className="mt-2">
              <label htmlFor="prep-norm-mode">{t("Normalization mode")}</label>
              <select
                id="prep-norm-mode"
                value={normalizationMode}
                onChange={(e) => setNormalizationMode(e.target.value)}
              >
                {["none", "pre", "post"].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
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
                    ...(cpuCores ? { cpuCores: Number(cpuCores) } : {}),
                    cutPreprocess: cut,
                    chunkLen: chunk,
                    overlapLen: overlap,
                    noiseReduction,
                    cleanStrength,
                    processEffects,
                    normalizationMode,
                  })
                }
              >
                {t("Run Preprocess")}
              </button>
            </div>
          </div>

          {/* Step 2: Feature Extraction */}
          <div className="card space-y-4">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-white/10 text-xs font-bold text-white flex items-center justify-center">
                    2
                  </span>
                  <Activity size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Extract Features")}
                  </h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t("Extract pitch contours and speech representations with your chosen embedder.")}
              </p>
            </div>
            <div className="grid2">
              <div>
                <label htmlFor="ext-pitch-method">{t("Pitch Method (F0)")}</label>
                <select id="ext-pitch-method" value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
                  {["crepe", "crepe-tiny", "rmvpe"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="ext-embedder-model">{t("Embedder Model")}</label>
                <select
                  id="ext-embedder-model"
                  value={embedder}
                  onChange={(e) => setEmbedder(e.target.value)}
                >
                  {["contentvec", "spin-v2", "custom"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              {embedder === "custom" && (
                <div>
                  <label htmlFor="ext-custom-embedder">{t("Custom embedder path")}</label>
                  <input
                    id="ext-custom-embedder"
                    type="text"
                    value={embedderCustom}
                    onChange={(e) => setEmbedderCustom(e.target.value)}
                    placeholder="rvc/models/embedders/embedders_custom/my-embedder"
                  />
                </div>
              )}
              <div>
                <SliderField
                  id="ext-include-mutes"
                  label={t("Include mutes")}
                  value={includeMutes}
                  min={0}
                  max={10}
                  step={1}
                  onChange={setIncludeMutes}
                />
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
                    ...(cpuCores ? { cpuCores: Number(cpuCores) } : {}),
                    embedderModel: embedder,
                    ...(embedder === "custom" && embedderCustom
                      ? { embedderModelCustom: embedderCustom }
                      : {}),
                    includeMutes,
                  })
                }
              >
                {t("Run Extract")}
              </button>
            </div>
          </div>

          {/* Step 3: Train */}
          <div className="card space-y-4">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-white/10 text-xs font-bold text-white flex items-center justify-center">
                    3
                  </span>
                  <Flame size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Model Training")}
                  </h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t("Train generator and discriminator weights and compile the feature index.")}
              </p>
            </div>
            <div className="grid2">
              <div>
                <label htmlFor="train-step-vocoder">{t("Vocoder")}</label>
                <select id="train-step-vocoder" value={vocoder} onChange={(e) => pickVocoder(e.target.value)}>
                  {["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <SliderField
                  id="train-step-total-epoch"
                  label={t("Total Epochs")}
                  value={totalEpoch}
                  min={1}
                  max={10000}
                  step={1}
                  onChange={setTotalEpoch}
                />
              </div>
              <div>
                <SliderField
                  id="train-step-batch-size"
                  label={t("Batch Size")}
                  value={batchSize}
                  min={1}
                  max={64}
                  step={1}
                  onChange={setBatchSize}
                />
              </div>
              <div>
                <SliderField
                  id="train-step-save-every"
                  label={t("Save Every N Epochs")}
                  value={saveEvery}
                  min={1}
                  max={100}
                  step={1}
                  onChange={setSaveEvery}
                />
              </div>
              <div>
                <label htmlFor="train-step-index-algo">{t("Index Algorithm")}</label>
                <select
                  id="train-step-index-algo"
                  value={indexAlgo}
                  onChange={(e) => setIndexAlgo(e.target.value)}
                >
                  {["Auto", "Faiss", "KMeans"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <label htmlFor="train-step-pretrained" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-pretrained"
                type="checkbox"
                checked={pretrained}
                onChange={(e) => setPretrained(e.target.checked)}
              />
              <span>{t("Use pretrained model")}</span>
            </label>
            <label htmlFor="train-step-save-latest" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-save-latest"
                type="checkbox"
                checked={saveOnlyLatest}
                onChange={(e) => setSaveOnlyLatest(e.target.checked)}
              />
              <span>{t("Save only latest checkpoint")}</span>
            </label>
            <label htmlFor="train-step-save-weights" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-save-weights"
                type="checkbox"
                checked={saveEveryWeights}
                onChange={(e) => setSaveEveryWeights(e.target.checked)}
              />
              <span>{t("Save model weights every checkpoint")}</span>
            </label>
            <label htmlFor="train-step-cleanup" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-cleanup"
                type="checkbox"
                checked={cleanup}
                onChange={(e) => setCleanup(e.target.checked)}
              />
              <span>{t("Fresh start (clean up previous attempt)")}</span>
            </label>
            <label htmlFor="train-step-cache-gpu" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-cache-gpu"
                type="checkbox"
                checked={cacheGpu}
                onChange={(e) => setCacheGpu(e.target.checked)}
              />
              <span>{t("Cache Dataset in GPU")}</span>
            </label>
            <label htmlFor="train-step-checkpointing" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-checkpointing"
                type="checkbox"
                checked={checkpointing}
                onChange={(e) => setCheckpointing(e.target.checked)}
              />
              <span>{t("Memory-efficient checkpointing")}</span>
            </label>
            <label htmlFor="train-step-custom-pre" className="flex items-center gap-2 cursor-pointer mt-3">
              <input
                id="train-step-custom-pre"
                type="checkbox"
                checked={customPre}
                onChange={(e) => setCustomPre(e.target.checked)}
              />
              <span>{t("Custom pretrained G/D")}</span>
            </label>
            {customPre && (
              <div className="grid2 mt-2">
                <div>
                  <label htmlFor="train-step-gpath">{t("G path")}</label>
                  <input
                    id="train-step-gpath"
                    type="text"
                    list="preG"
                    value={gPath}
                    onChange={(e) => setGPath(e.target.value)}
                  />
                  <datalist id="preG">
                    {pretG.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </div>
                <div>
                  <label htmlFor="train-step-dpath">{t("D path")}</label>
                  <input
                    id="train-step-dpath"
                    type="text"
                    list="preD"
                    value={dPath}
                    onChange={(e) => setDPath(e.target.value)}
                  />
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
                onClick={() => {
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
                    pretrained,
                    saveOnlyLatest,
                    saveEveryWeights,
                    cleanup,
                    cacheDataInGpu: cacheGpu,
                    checkpointing,
                  });
                }}
              >
                {t("Start Training")}
              </button>
              <button
                type="button"
                className="ghost"
                onClick={() => run("/api/train/index", { modelName, indexAlgorithm: indexAlgo })}
              >
                {t("Generate Index Only")}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 3. UPLOADS VIEW */}
      {trainMode === "uploads" && (
        <div id="panel-uploads" role="tabpanel" aria-labelledby="tab-uploads" className="card space-y-4">
          <div className="border-b border-white/10 pb-3.5 space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FolderUp size={18} className="text-white" />
                <h2 className="text-base font-bold text-white m-0">
                  {t("Dataset & Checkpoint Uploads")}
                </h2>
              </div>
            </div>
            <p className="text-xs text-neutral-400 m-0 leading-relaxed">
              {t("Upload local dataset files or pretrained generator checkpoints directly.")}
            </p>
          </div>
          <UploadBox
            path="/api/train/upload-dataset"
            fields={[{ name: "datasetName", label: t("Dataset name (e.g. my_vocals)") }]}
            files="files"
            multiple
            label={t("Dataset Audio Files (WAV/MP3/FLAC) → assets/datasets/<name>/")}
          />
          <UploadBox
            path="/api/train/upload-pretrained"
            fields={[]}
            files="file"
            label={t("Custom Pretrained Weights (.pth) → rvc/models/pretraineds/custom/")}
          />
          <UploadBox
            path="/api/train/upload-embedder"
            fields={[{ name: "folderName", label: t("Folder Name") }]}
            files="bin"
            extra="config"
            label={t("Custom Embedder (.bin + .json)")}
          />
        </div>
      )}

      {/* Export Model */}
      <div className="card space-y-4 mt-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <Download size={18} className="text-white" />
            <h2 className="text-base font-bold text-white m-0">
              {t("Export Model")}
            </h2>
          </div>
        </div>
        <p className="muted text-sm m-0">{t("Download a trained .pth and its .index from logs/.")}</p>
        <div className="grid2">
          <div>
            <label htmlFor="train-exp-model">{t("Model (.pth)")}</label>
            <select id="train-exp-model" value={expModel} onChange={(e) => setExpModel(e.target.value)}>
              <option value="">—</option>
              {expModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="train-exp-index">{t("Index (.index)")}</label>
            <select id="train-exp-index" value={expIndex} onChange={(e) => setExpIndex(e.target.value)}>
              <option value="">—</option>
              {expIndexes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row mt-4">
          <button
            type="button"
            className="ghost"
            onClick={() => downloadExport(expModel)}
            disabled={!expModel}
          >
            {t("Download .pth")}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => downloadExport(expIndex)}
            disabled={!expIndex}
          >
            {t("Download .index")}
          </button>
        </div>
      </div>

      {/* Stop Controller Card */}
      <div className="card space-y-4 mt-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <StopCircle size={18} className="text-white" />
            <h2 className="text-base font-bold text-white m-0">
              {t("Stop Training Process")}
            </h2>
          </div>
        </div>
        <div className="row">
          <input
            type="text"
            placeholder={t("model name (fallback)")}
            aria-label={t("Model name (fallback)")}
            value={stopTarget}
            onChange={(e) => setStopTarget(e.target.value)}
            style={{ maxWidth: 240 }}
          />
          <button type="button" className="ghost" onClick={stop}>
            {t("Stop Training")}
          </button>
        </div>
      </div>

      <TrainingConsole
        jobId={jobId}
        modelName={modelName}
        totalEpochs={totalEpoch}
        onStop={stop}
      />
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
  const { t } = useI18n();
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
      if (!r.ok) throw new Error(b?.error || t("Upload failed."));
      setMsg(t("Uploaded."));
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
            aria-label={f.label}
            value={vals[f.name] || ""}
            onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}
            style={{ maxWidth: 200 }}
          />
        ))}
        <input
          type="file"
          aria-label={label}
          multiple={multiple}
          onChange={(e) => setPicked(e.target.files)}
        />
        {extra && (
          <input
            type="file"
            aria-label={`${label} (${t("extra config")})`}
            onChange={(e) => setPicked2(e.target.files)}
          />
        )}
        <button type="button" className="ghost text-xs" onClick={send}>
          {t("Upload")}
        </button>
        <span className="text-xs text-neutral-300">{msg}</span>
      </div>
    </div>
  );
}
