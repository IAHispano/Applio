"use client";

import { useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, errMsg, submitJob } from "../../lib/api";

export default function TrainPage() {
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
      .catch(() => setGpuInfo("GPU query failed (CPU-only host?)"));
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
        title="Training"
        description="Preprocess datasets, extract acoustic features, and train custom voice conversion models."
      />
      <div className="mb-4">
        <div className="grid2">
          <div>
            <label>Model Name (shared by all steps)</label>
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} />
          </div>
          <div>
            <label>GPU</label>
            <input
              type="text"
              value={gpuCount}
              onChange={(e) => setGpuCount(e.target.value)}
              placeholder="0, or - for CPU"
            />
          </div>
        </div>
        <p className="muted">{gpuInfo || "Querying GPUs…"}</p>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      </div>

      <div className="card">
        <h2>1 · Preprocess</h2>
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
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Cut method</label>
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
        <label>
          <input
            type="checkbox"
            checked={noiseReduction}
            onChange={(e) => setNoiseReduction(e.target.checked)}
          />{" "}
          Noise reduction
        </label>
        <div className="row" style={{ marginTop: 8 }}>
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

      <div className="card">
        <h2>2 · Extract Features</h2>
        <div className="grid2">
          <div>
            <label>Pitch method</label>
            <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {["crepe", "crepe-tiny", "rmvpe"].map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Embedder</label>
            <select value={embedder} onChange={(e) => setEmbedder(e.target.value)}>
              {["contentvec", "spin-v2", "custom"].map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
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

      <div className="card">
        <h2>3 · Train</h2>
        <div className="grid2">
          <div>
            <label>Vocoder</label>
            <select value={vocoder} onChange={(e) => setVocoder(e.target.value)}>
              {["HiFi-GAN", "RefineGAN"].map((s) => (
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
        <label>
          <input type="checkbox" checked={customPre} onChange={(e) => setCustomPre(e.target.checked)} />{" "}
          Custom pretrained G/D
        </label>
        {customPre && (
          <div className="grid2">
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
        <div className="row" style={{ marginTop: 8 }}>
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

      <div className="card">
        <h2>Stop</h2>
        <div className="row">
          <input
            type="text"
            placeholder="model name (legacy fallback)"
            value={stopTarget}
            onChange={(e) => setStopTarget(e.target.value)}
            style={{ maxWidth: 240 }}
          />
          <button type="button" className="ghost" onClick={stop}>
            Stop current job / model
          </button>
        </div>
        <p className="muted">
          Prefers the tracked job above; falls back to the{" "}
          <code>logs/&lt;model&gt;/config.json process_pids</code> kill.
        </p>
      </div>

      <div className="card">
        <h2>Uploads</h2>
        <UploadBox
          path="/api/train/upload-dataset"
          fields={[{ name: "datasetName", label: "Dataset name" }]}
          files="files"
          multiple
          label="Dataset audio files → assets/datasets/<name>/"
        />
        <UploadBox
          path="/api/train/upload-pretrained"
          fields={[]}
          files="file"
          label="Pretrained .pth → rvc/models/pretraineds/custom/"
        />
        <UploadBox
          path="/api/train/upload-embedder"
          fields={[{ name: "folderName", label: "Folder name" }]}
          files="bin"
          extra="config"
          label="Custom embedder .bin + .json"
        />
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
    <div style={{ marginBottom: 12 }}>
      <p className="muted">{label}</p>
      <div className="row">
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
        <button type="button" className="ghost" onClick={send}>
          Upload
        </button>
        <span className="muted">{msg}</span>
      </div>
    </div>
  );
}
