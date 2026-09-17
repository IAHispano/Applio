"use client";

import { useEffect, useState } from "react";
import { errMsg, fetchModels, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import JobPanel from "../JobPanel";
import SliderField from "../ui/SliderField";

export default function BlenderPanel() {
  const { t } = useI18n();
  const [models, setModels] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [p1, setP1] = useState("");
  const [p2, setP2] = useState("");
  const [f1, setF1] = useState<File | null>(null);
  const [f2, setF2] = useState<File | null>(null);
  const [ratio, setRatio] = useState(0.5);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        if (m.models[0]) setP1(m.models[0]);
        if (m.models[1]) setP2(m.models[1]);
      })
      .catch(() => {});
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!name || (!p1 && !f1) || (!p2 && !f2)) {
      setError(t("Name + two models are required (path or upload)."));
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("modelName", name);
      fd.append("pthPath1", p1);
      fd.append("pthPath2", p2);
      if (f1) fd.append("pth_file_1", f1);
      if (f2) fd.append("pth_file_2", f2);
      fd.append("ratio", String(ratio));
      const { jobId: id } = await postForm<{ jobId: string }>("/api/voice-blender", fd);
      setJobId(id);
    } catch (err) {
      setError(errMsg(err) || t("Submit failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mb-4 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
        >
          {error}
        </div>
      )}
      <form onSubmit={onSubmit}>
        <div className="card">
          <div className="grid2">
            <div>
              <label htmlFor="blend-model-name">{t("New model name")}</label>
              <input
                id="blend-model-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("my-fusion")}
              />
            </div>
            <div>
              <SliderField
                id="blend-ratio"
                label={`${t("Blend ratio")} (0 = ${t("Model 1")}, 1 = ${t("Model 2")})`}
                value={ratio}
                min={0}
                max={1}
                step={0.05}
                onChange={setRatio}
              />
            </div>
            <div>
              <label htmlFor="blend-model1-path">{t("Model 1 path")}</label>
              <input
                id="blend-model1-path"
                type="text"
                list="vmodels"
                value={p1}
                onChange={(e) => setP1(e.target.value)}
              />
              <label htmlFor="blend-model1-file" className="sr-only">
                {t("Upload Model 1 file")}
              </label>
              <input
                id="blend-model1-file"
                type="file"
                accept=".pth,.onnx"
                onChange={(e) => setF1(e.target.files?.[0] || null)}
                className="mt-2"
              />
            </div>
            <div>
              <label htmlFor="blend-model2-path">{t("Model 2 path")}</label>
              <input
                id="blend-model2-path"
                type="text"
                list="vmodels"
                value={p2}
                onChange={(e) => setP2(e.target.value)}
              />
              <label htmlFor="blend-model2-file" className="sr-only">
                {t("Upload Model 2 file")}
              </label>
              <input
                id="blend-model2-file"
                type="file"
                accept=".pth,.onnx"
                onChange={(e) => setF2(e.target.files?.[0] || null)}
                className="mt-2"
              />
            </div>
          </div>
          <datalist id="vmodels">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="submit" className="cta" disabled={busy}>
              {busy ? t("Blending…") : t("Fuse Models")}
            </button>
          </div>
        </div>
      </form>
      <JobPanel jobId={jobId} />
    </div>
  );
}
