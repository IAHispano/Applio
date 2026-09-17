"use client";

import { useEffect, useState } from "react";
import {
  errMsg,
  fetchJob,
  isAudioFile,
  isImageFile,
  type Job,
  outputUrl,
  pollJob,
  stopJob,
} from "../lib/api";
import { useI18n } from "../lib/i18n";
import AudioPlayer from "./AudioPlayer";

// Polls a job, shows status/logs, and renders its output file.
export default function JobPanel({ jobId, compact }: { jobId: string | null; compact?: boolean }) {
  const { t } = useI18n();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    setError("");
    let stop = () => {};
    fetchJob(jobId)
      .then(({ job: j }) => {
        setJob(j);
        if (j.status !== "done" && j.status !== "error") stop = pollJob(jobId, setJob);
      })
      .catch((e) => setError(String(e?.message || e)));
    return () => stop();
  }, [jobId]);

  if (!jobId) return null;
  if (error) return <p style={{ color: "var(--err)" }}>{error}</p>;
  if (!job) return <p className="muted">{t("Loading job…")}</p>;

  const out = job.outputFile;
  const sidecars: Array<{ label: string; file: string }> = [];
  if (job.result) {
    for (const [k, v] of Object.entries(job.result)) {
      if (
        typeof v === "string" &&
        (v.endsWith(".txt") ||
          v.endsWith(".png") ||
          v.endsWith(".pth") ||
          v.endsWith(".wav") ||
          v.endsWith(".mp4") ||
          v.endsWith(".webm")) &&
        v !== out
      ) {
        sidecars.push({ label: k, file: v });
      }
    }
  }

  const resultMsg = typeof job.result?.message === "string" ? job.result.message : null;
  const resultInfo = typeof job.result?.info === "string" ? job.result.info : null;

  return (
    <div className="card">
      <div className="row">
        <span className={`badge ${job.status}`}>{job.status}</span>
        <span className="muted">
          {t("job")} {job.id}
        </span>
        {(job.status === "queued" || job.status === "running") && (
          <button
            type="button"
            className="ghost"
            onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
          >
            {t("Stop")}
          </button>
        )}
      </div>
      {job.status === "error" && <p style={{ color: "var(--err)" }}>{job.error}</p>}
      {(job.status === "queued" || job.status === "running") && (
        <div className="loader" style={{ marginTop: 12 }}>
          <div className="loaderBar" />
        </div>
      )}
      {resultMsg && <p>{resultMsg}</p>}
      {resultInfo && (
        <div>
          <p className="muted">{t("Analysis result")}</p>
          <div className="log">{resultInfo}</div>
        </div>
      )}
      {out && isAudioFile(out) && <AudioPlayer src={outputUrl(out)} filename={out.split("/").pop()} />}
      {out && isImageFile(out) && (
        <div>
          {/* biome-ignore lint/performance/noImgElement: user-generated plot, no optimizer benefit */}
          <img src={outputUrl(out)} alt={t("output")} style={{ maxWidth: "100%", borderRadius: 8 }} />
          <p>
            <a href={outputUrl(out)} download>
              {t("Download image")}
            </a>{" "}
            <span className="muted">{out}</span>
          </p>
        </div>
      )}
      {out && !isAudioFile(out) && !isImageFile(out) && (
        <p>
          <a href={outputUrl(out)} download>
            {t("Download result")}
          </a>{" "}
          <span className="muted">{out}</span>
        </p>
      )}
      {sidecars.map((s) => (
        <p key={s.label}>
          <a href={outputUrl(s.file)} download>
            {t("Download")} {s.label}
          </a>{" "}
          <span className="muted">{s.file}</span>
        </p>
      ))}
      {!compact && job.logs.length > 0 && (
        <div>
          <p className="muted">{t("Engine logs")}</p>
          <div className="log">{job.logs.slice(-60).join("\n")}</div>
        </div>
      )}
    </div>
  );
}
