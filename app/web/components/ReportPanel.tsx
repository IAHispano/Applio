"use client";

import { useEffect, useRef, useState } from "react";
import { apiGet, apiSend, errMsg, fileBasename, outputUrl } from "../lib/api";
import { useI18n } from "../lib/i18n";

interface SystemInfo {
  version: string;
  platform: string;
  node: string;
  python: string;
  cpus: number;
  totalMemGB: number;
  issueUrl: string;
}

export default function ReportPanel() {
  const { t } = useI18n();
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [recording, setRecording] = useState(false);
  const [clip, setClip] = useState("");
  const [msg, setMsg] = useState("");
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    apiGet<SystemInfo>("/api/report/info")
      .then(setInfo)
      .catch((e) => setMsg(errMsg(e)));
  }, []);

  async function toggleRecord() {
    setMsg("");
    if (recording) {
      recRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      const rec = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : undefined,
      });
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => {
          t.stop();
        });
        setRecording(false);
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        const dataUrl = await new Promise<string>((resolve) => {
          const fr = new FileReader();
          fr.onload = () => resolve(String(fr.result));
          fr.readAsDataURL(blob);
        });
        try {
          const r = await apiSend<{ file: string; issueUrl: string }>("/api/report/upload", "POST", {
            dataUrl,
          });
          setClip(r.file);
          setMsg(`Clip saved → ${r.file}. Attach it to your GitHub issue.`);
        } catch (e) {
          setMsg(errMsg(e));
        }
      };
      recRef.current = rec;
      rec.start();
      setRecording(true);
    } catch {
      setMsg(t("Screen capture cancelled or unsupported in this browser."));
    }
  }

  const issueBody = info
    ? `**Describe the bug**%0A%0A**System**%0A- Applio ${info.version} · ${info.platform}%0A- Node ${info.node} · ${info.python}%0A- CPUs ${info.cpus} · RAM ${info.totalMemGB}GB`
    : "";

  return (
    <div>
      {info ? (
        <section aria-label={t("System diagnostics")} className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
            <span className="text-neutral-400 block">{t("Version")}</span>
            <span className="font-medium text-white">{info.version}</span>
          </div>
          <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
            <span className="text-neutral-400 block">{t("Platform")}</span>
            <span className="font-medium text-white truncate block">{info.platform}</span>
          </div>
          <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
            <span className="text-neutral-400 block">{t("Node / Python")}</span>
            <span className="font-medium text-white truncate block">Node {info.node} · {info.python}</span>
          </div>
          <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
            <span className="text-neutral-400 block">{t("Hardware")}</span>
            <span className="font-medium text-white">{info.cpus} CPUs · {info.totalMemGB}GB RAM</span>
          </div>
        </section>
      ) : (
        <p className="muted">{t("Collecting system info…")}</p>
      )}
      <div className="row" style={{ marginTop: 12 }}>
        <button
          type="button"
          className={recording ? "cta" : "ghost"}
          onClick={toggleRecord}
          aria-pressed={recording}
        >
          {recording ? t("Stop Recording") : t("Record Screen")}
        </button>
        {info && (
          <button
            type="button"
            className="cta"
            onClick={() => window.open(`${info.issueUrl}?body=${issueBody}`, "_blank")}
          >
            {t("Open GitHub Issue")}
          </button>
        )}
      </div>
      {clip && (
        <p>
          <a href={outputUrl(clip)} download aria-label={`${t("Download clip")}: ${fileBasename(clip)}`}>
            {t("Download clip")}
          </a>{" "}
          <span className="muted">{clip}</span>
        </p>
      )}
      {msg && (
        <p role="status" aria-live="polite" className="muted">
          {msg}
        </p>
      )}
    </div>
  );
}
