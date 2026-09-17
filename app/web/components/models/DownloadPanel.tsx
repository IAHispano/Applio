"use client";

import { useEffect, useState } from "react";
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
      <div className="card">
        <h2>{t("From link")}</h2>
        <div className="row">
          <label htmlFor="dl-link-input" className="sr-only">
            {t("Model link")}
          </label>
          <input
            id="dl-link-input"
            type="text"
            value={link}
            onChange={(e) => setLink(e.target.value)}
            placeholder={t("Model link (Drive, HuggingFace, direct zip)…")}
            style={{ flex: 1 }}
          />
          <button type="button" className="cta" onClick={downloadLink}>
            {t("Download")}
          </button>
        </div>
      </div>
      <JobPanel jobId={linkJob} compact />

      <div className="card">
        <h2>{t("Drop files")}</h2>
        <div className="row">
          <label htmlFor="dl-file-input" className="sr-only">
            {t("Upload model file")}
          </label>
          <input
            id="dl-file-input"
            type="file"
            accept=".pth,.index,.onnx"
            onChange={(e) => setDropFile(e.target.files?.[0] || null)}
          />
          <button type="button" className="ghost" onClick={drop}>
            {t("Save File")}
          </button>
          {dropMsg && (
            <span className="muted" role="status" aria-live="polite">
              {dropMsg}
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <h2>{t("Pretrained models")}</h2>
        {!custom ? (
          <div className="grid2">
            <div>
              <label htmlFor="dl-pretrained-select">{t("Pretrained")}</label>
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
          <div className="grid2">
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
          className="checkbox-label flex items-center gap-2 cursor-pointer mt-3"
        >
          <input
            id="dl-custom-checkbox"
            type="checkbox"
            checked={custom}
            onChange={(e) => setCustom(e.target.checked)}
          />
          <span>{t("Custom Pretrained")}</span>
        </label>
        <div className="row" style={{ marginTop: 8 }}>
          <button type="button" className="ghost" onClick={downloadPretrained}>
            {t("Download")}
          </button>
        </div>
      </div>
      <JobPanel jobId={preJob} />
    </div>
  );
}
