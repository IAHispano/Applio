"use client";

import { useEffect, useState } from "react";
import { Link2, Upload, Database, Download } from "lucide-react";
import { apiGet, apiSend, errMsg, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import JobPanel from "../JobPanel";

export default function DownloadPanel() {
  const { t } = useI18n();
  const [link, setLink] = useState("");
  const [linkJob, setLinkJob] = useState<string | null>(null);
  const [dropFile, setDropFile] = useState<File | null>(null);
  const [dropMsg, setDropMsg] = useState("");
  const [pretrained, setPretrained] = useState<Array<{ name: string; sampleRates: string[] }>>([]);
  const [model, setModel] = useState("Titan");
  const [sr, setSr] = useState("40k");
  const [custom, setCustom] = useState(false);
  const [urlG, setUrlG] = useState("");
  const [urlD, setUrlD] = useState("");
  const [preJob, setPreJob] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<{ models: Array<{ name: string; sampleRates: string[] }> }>("/api/download/pretraineds")
      .then((p) => {
        setPretrained(p.models);
        if (p.models[0]) {
          setModel(p.models[0].name);
          if (p.models[0].sampleRates[0]) setSr(p.models[0].sampleRates[0]);
        }
      })
      .catch((e) => setError(errMsg(e)));
  }, []);

  async function downloadLink() {
    setError("");
    try {
      const { jobId } = await apiSend<{ jobId: string }>("/api/download", "POST", { modelLink: link });
      setLinkJob(jobId);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function drop() {
    setDropMsg("");
    if (!dropFile) return;
    const fd = new FormData();
    fd.append("file", dropFile);
    try {
      const r = await postForm<{ file: string; modelDir: string }>("/api/download/drop", fd);
      setDropMsg(`Saved ${r.file} → ${r.modelDir} ✓`);
    } catch (e) {
      setDropMsg(errMsg(e));
    }
  }

  async function downloadPretrained() {
    setError("");
    try {
      const body = custom ? { urlG, urlD } : { model, sampleRate: sr };
      const { jobId } = await apiSend<{ jobId: string }>("/api/download/pretraineds", "POST", body);
      setPreJob(jobId);
    } catch (e) {
      setError(errMsg(e));
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

      {/* Card 1: Download from URL */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Link2 size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Download from Link")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Paste a model URL (HuggingFace, Google Drive, Mega, or direct zip) to download weights.")}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 max-w-xl">
          <label htmlFor="dl-link-input" className="sr-only">
            {t("Model link")}
          </label>
          <input
            id="dl-link-input"
            type="text"
            value={link}
            onChange={(e) => setLink(e.target.value)}
            placeholder={t("Model link (Drive, HuggingFace, direct zip)…")}
            className="flex-1 h-10 px-3 text-sm rounded-xl bg-white/5 border border-white/10"
          />
          <button
            type="button"
            className="cta h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl shrink-0"
            onClick={downloadLink}
          >
            <Download size={16} className="shrink-0" />
            <span>{t("Download")}</span>
          </button>
        </div>
      </div>
      <JobPanel jobId={linkJob} compact />

      {/* Card 2: Upload Files */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Upload size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Upload Model Files")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Upload local .pth, .index, or .onnx files directly into your models directory.")}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 max-w-xl">
          <label htmlFor="dl-file-input" className="sr-only">
            {t("Upload model file")}
          </label>
          <input
            id="dl-file-input"
            type="file"
            accept=".pth,.index,.onnx"
            onChange={(e) => setDropFile(e.target.files?.[0] || null)}
            className="flex-1"
          />
          <button
            type="button"
            className="ghost h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl shrink-0"
            onClick={drop}
          >
            <Upload size={16} className="text-white shrink-0" />
            <span>{t("Save File")}</span>
          </button>
        </div>
        {dropMsg && (
          <p className="text-xs text-neutral-400 m-0" role="status" aria-live="polite">
            {dropMsg}
          </p>
        )}
      </div>

      {/* Card 3: Pretrained Base Models */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Pretrained Base Models")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Download generator and discriminator checkpoints for training custom voices.")}
          </p>
        </div>

        {!custom ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
            <div>
              <label htmlFor="dl-pretrained-select">{t("Pretrained Model")}</label>
              <select
                id="dl-pretrained-select"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  const m = pretrained.find((p) => p.name === e.target.value);
                  if (m?.sampleRates[0]) setSr(m.sampleRates[0]);
                }}
              >
                {pretrained.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="dl-sr-select">{t("Sampling Rate")}</label>
              <select id="dl-sr-select" value={sr} onChange={(e) => setSr(e.target.value)}>
                {(pretrained.find((p) => p.name === model)?.sampleRates || [sr]).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
            <div>
              <label htmlFor="dl-url-g">{t("Pretrained G URL")}</label>
              <input id="dl-url-g" type="text" value={urlG} onChange={(e) => setUrlG(e.target.value)} />
            </div>
            <div>
              <label htmlFor="dl-url-d">{t("Pretrained D URL")}</label>
              <input id="dl-url-d" type="text" value={urlD} onChange={(e) => setUrlD(e.target.value)} />
            </div>
          </div>
        )}

        <label
          htmlFor="dl-custom-checkbox"
          className="flex items-center gap-2 cursor-pointer text-sm text-neutral-300"
        >
          <input
            id="dl-custom-checkbox"
            type="checkbox"
            checked={custom}
            onChange={(e) => setCustom(e.target.checked)}
          />
          <span>{t("Custom Pretrained URLs")}</span>
        </label>

        <div className="pt-3.5 border-t border-white/5 flex justify-end">
          <button
            type="button"
            className="ghost h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl text-neutral-200 hover:text-white"
            onClick={downloadPretrained}
          >
            <Download size={16} className="text-white shrink-0" />
            <span>{t("Download Pretrained")}</span>
          </button>
        </div>
      </div>
      <JobPanel jobId={preJob} />
    </div>
  );
}
