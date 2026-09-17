"use client";

import { useEffect, useRef, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

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
    <div>
      <PageHeader
        title={t("Report a Bug")}
        description={t("Collect system diagnostics, record screen logs, and submit issue reports to GitHub.")}
      />
      <div className="card">
        <h2>{t("How to Report an Issue on GitHub")}</h2>
        <p className="muted">
          {t(
            "1. Click on the 'Record Screen' button below to start recording the issue you are experiencing.",
          )}
        </p>
        <p className="muted">
          {t(
            "2. Once you have finished recording the issue, click on the 'Stop Recording' button (the same button, but the label changes depending on whether you are actively recording or not).",
          )}
        </p>
        <p className="muted">{t("3. Go to GitHub Issues and click on the 'New Issue' button.")}</p>
        <p className="muted">
          {t(
            "4. Complete the provided issue template, ensuring to include details as needed, and utilize the assets section to upload the recorded file from the previous step.",
          )}
        </p>
      </div>
      {info ? (
        <div className="log">
          {`Applio ${info.version}\n${info.platform}\nNode ${info.node}\n${info.python}\nCPUs: ${info.cpus} · RAM: ${info.totalMemGB}GB`}
        </div>
      ) : (
        <p className="muted">{t("Collecting system info…")}</p>
      )}
      <div className="row" style={{ marginTop: 12 }}>
        <button type="button" className="ghost" onClick={toggleRecord}>
          {recording ? t("Stop Recording") : t("Record Screen")}
        </button>
        {info && (
          <a href={`${info.issueUrl}?body=${issueBody}`} target="_blank" rel="noreferrer">
            <button type="button">{t("Open GitHub Issue")}</button>
          </a>
        )}
      </div>
      {clip && (
        <div style={{ marginTop: 12 }}>
          {/* biome-ignore lint/a11y/useMediaCaption: user-recorded screen capture has no caption track */}
          <video
            controls
            src={`/outputs/${clip.split("/").pop()}`}
            style={{ maxWidth: "100%", borderRadius: 8 }}
          />
          <p>
            <a href={`/outputs/${clip.split("/").pop()}`} download>
              {t("Download clip")}
            </a>{" "}
            <span className="muted">{clip}</span>
          </p>
        </div>
      )}
      {msg && <p className="muted">{msg}</p>}
    </div>
  );
}
