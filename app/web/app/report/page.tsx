"use client";

import { useEffect, useRef, useState } from "react";
import { Bug, Check, Copy, Cpu, Download, ExternalLink, Video } from "lucide-react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg, outputUrl } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { toast } from "../../lib/toast";

interface SystemInfo {
  version: string;
  platform: string;
  node: string;
  python: string;
  cpus: number;
  totalMemGB: number;
  issueUrl: string;
}

export default function ReportPage() {
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
      // Capture the screen in-browser.
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
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Report a Bug")}
        description={t("Collect system diagnostics, record screen logs, and submit issue reports to GitHub.")}
      />

      {msg && (
        <div
          role="status"
          aria-live="polite"
          className="p-3.5 rounded-xl border border-white/10 text-neutral-300 bg-white/5 text-sm"
        >
          {msg}
        </div>
      )}

      {/* Guide Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bug size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("How to Report an Issue on GitHub")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Follow these steps to record a reproduction clip and submit a detailed bug report.")}
          </p>
        </div>
        <ol className="space-y-2 text-xs text-neutral-300 m-0 pl-4 leading-relaxed">
          <li>{t("Click on 'Record Screen' to start recording the issue you are experiencing.")}</li>
          <li>{t("Once you have finished reproducing the issue, click 'Stop Recording'.")}</li>
          <li>{t("Go to GitHub Issues and click on 'New Issue'.")}</li>
          <li>
            {t(
              "Complete the provided issue template, paste the diagnostic information below, and attach the recorded screen clip.",
            )}
          </li>
        </ol>

        <div className="flex items-center gap-3 pt-3.5 border-t border-white/5">
          <button
            type="button"
            className={
              recording
                ? "ghost h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl text-neutral-200 hover:text-white"
                : "cta h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl"
            }
            onClick={toggleRecord}
          >
            <Video size={16} className={recording ? "animate-pulse shrink-0" : "shrink-0"} />
            <span>{recording ? t("Stop Recording") : t("Record Screen")}</span>
          </button>
          {info && (
            <a
              href={`${info.issueUrl}?body=${issueBody}`}
              target="_blank"
              rel="noreferrer"
              className="ghost h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl text-neutral-200 hover:text-white"
            >
              <ExternalLink size={16} className="text-white shrink-0" />
              <span>{t("Open GitHub Issue")}</span>
            </a>
          )}
        </div>
      </div>

      {clip && (
        <div className="card space-y-3">
          {/* biome-ignore lint/performance/noImgElement: user-recorded screen capture has no caption track */}
          <video controls src={outputUrl(clip)} className="max-w-full rounded-xl border border-white/10" />
          <div className="flex items-center justify-between">
            <a
              href={outputUrl(clip)}
              download
              className="cta h-10 px-4 flex items-center gap-2 rounded-xl text-sm font-medium"
            >
              <Download size={16} className="shrink-0" />
              <span>{t("Download Video")}</span>
            </a>
            <span className="text-xs text-neutral-400">{clip}</span>
          </div>
        </div>
      )}

      {/* Diagnostics Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("System Diagnostics")}</h2>
            </div>
            {info && (
              <button
                type="button"
                className="ghost h-8 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 text-neutral-300 hover:text-white"
                onClick={() => {
                  const text = `Applio ${info.version}\n${info.platform}\nNode ${info.node}\n${info.python}\nCPUs: ${info.cpus} · RAM: ${info.totalMemGB}GB`;
                  navigator.clipboard.writeText(text);
                  toast(t("Diagnostics copied to clipboard"));
                }}
              >
                <Copy size={13} />
                <span>{t("Copy Diagnostics")}</span>
              </button>
            )}
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Environment details and system specifications to include in your issue report.")}
          </p>
        </div>

        {info ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-black/30 border border-white/5 space-y-1">
              <span className="text-xs text-neutral-400 block">{t("Applio Version")}</span>
              <span className="text-sm font-semibold text-white block">{info.version}</span>
            </div>
            <div className="p-3 rounded-xl bg-black/30 border border-white/5 space-y-1">
              <span className="text-xs text-neutral-400 block">{t("Platform")}</span>
              <span className="text-sm font-semibold text-white block truncate">{info.platform}</span>
            </div>
            <div className="p-3 rounded-xl bg-black/30 border border-white/5 space-y-1">
              <span className="text-xs text-neutral-400 block">{t("Engine Runtimes")}</span>
              <span className="text-sm font-semibold text-white block truncate">Node {info.node}</span>
              <span className="text-[10px] text-neutral-400 block truncate">{info.python}</span>
            </div>
            <div className="p-3 rounded-xl bg-black/30 border border-white/5 space-y-1">
              <span className="text-xs text-neutral-400 block">{t("Compute Resources")}</span>
              <span className="text-sm font-semibold text-white block">{info.cpus} CPU cores</span>
              <span className="text-[10px] text-neutral-400 block">{info.totalMemGB} GB RAM</span>
            </div>
          </div>
        ) : (
          <p className="text-xs text-neutral-400 m-0">{t("Collecting system info…")}</p>
        )}
      </div>
    </div>
  );
}
