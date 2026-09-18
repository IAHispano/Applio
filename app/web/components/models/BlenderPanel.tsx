"use client";

import { useEffect, useState } from "react";
import { Layers, Sparkles } from "lucide-react";
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
      setError(t("Name and two models are required (path or upload)."));
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
        <div className="card space-y-4">
          <div className="border-b border-white/10 pb-3.5 space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers size={18} className="text-white" />
                <h2 className="text-base font-bold text-white m-0">{t("Model Fusion & Blending")}</h2>
              </div>
              <span className="text-xs text-neutral-400">
                {models.length} {t("models detected")}
              </span>
            </div>
            <p className="text-xs text-neutral-400 m-0 leading-relaxed">
              {t("Merge and interpolate weights between two compatible voice model checkpoints.")}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-white/5">
            <div className="space-y-2">
              <label htmlFor="blend-model1-path">{t("Model 1 path")}</label>
              <input
                id="blend-model1-path"
                type="text"
                list="vmodels"
                value={p1}
                onChange={(e) => setP1(e.target.value)}
                placeholder="logs/model1.pth"
              />
              <label htmlFor="blend-model1-file" className="block text-xs text-neutral-400">
                {t("Or upload Model 1 file")}
              </label>
              <input
                id="blend-model1-file"
                type="file"
                accept=".pth,.onnx"
                onChange={(e) => setF1(e.target.files?.[0] || null)}
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="blend-model2-path">{t("Model 2 path")}</label>
              <input
                id="blend-model2-path"
                type="text"
                list="vmodels"
                value={p2}
                onChange={(e) => setP2(e.target.value)}
                placeholder="logs/model2.pth"
              />
              <label htmlFor="blend-model2-file" className="block text-xs text-neutral-400">
                {t("Or upload Model 2 file")}
              </label>
              <input
                id="blend-model2-file"
                type="file"
                accept=".pth,.onnx"
                onChange={(e) => setF2(e.target.files?.[0] || null)}
              />
            </div>
          </div>

          <datalist id="vmodels">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </div>

        {/* Action card */}
        <div className="card flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-neutral-400">
            <Sparkles size={16} className="text-white" />
            <span>{t("Interpolate weights between two checkpoint files.")}</span>
          </div>
          <button
            type="submit"
            className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
            disabled={busy}
          >
            <Layers size={16} className="shrink-0" />
            <span>{busy ? t("Blending…") : t("Fuse Models")}</span>
          </button>
        </div>
      </form>

      <JobPanel jobId={jobId} />
    </div>
  );
}
