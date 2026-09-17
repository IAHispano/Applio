"use client";

import { useEffect, useRef, useState } from "react";
import { apiGet, apiSend, errMsg } from "../lib/api";

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
      setMsg("Screen capture cancelled or unsupported in this browser.");
    }
  }

  const issueBody = info
    ? `**Describe the bug**%0A%0A**System**%0A- Applio ${info.version} · ${info.platform}%0A- Node ${info.node} · ${info.python}%0A- CPUs ${info.cpus} · RAM ${info.totalMemGB}GB`
    : "";

  return (
    <div>
      {info ? (
        <div className="log">
          {`Applio ${info.version}\n${info.platform}\nNode ${info.node}\n${info.python}\nCPUs: ${info.cpus} · RAM: ${info.totalMemGB}GB`}
        </div>
      ) : (
        <p className="muted">Collecting system info…</p>
      )}
      <div className="row" style={{ marginTop: 12 }}>
        <button type="button" className="ghost" onClick={toggleRecord}>
          {recording ? "Stop Recording" : "Record Screen"}
        </button>
        {info && (
          <a href={`${info.issueUrl}?body=${issueBody}`} target="_blank" rel="noreferrer">
            <button type="button">Open GitHub Issue</button>
          </a>
        )}
      </div>
      {clip && (
        <p>
          <a href={`/outputs/${clip.split("/").pop()}`} download>
            Download clip
          </a>{" "}
          <span className="muted">{clip}</span>
        </p>
      )}
      {msg && <p className="muted">{msg}</p>}
    </div>
  );
}
