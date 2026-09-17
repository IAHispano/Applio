"use client";

import { useEffect, useState } from "react";
import {
  errMsg,
  fetchJob,
  fileBasename,
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
    <section className="card space-y-3" aria-label={`Job ${job.id} panel`}>
      <div className="row justify-between">
        <div className="flex items-center gap-2">
          <span className={`badge ${job.status}`} role="status" aria-label={`Job status: ${job.status}`}>
            {job.status}
          </span>
          <span className="muted font-mono text-xs">
            {t("job")} {job.id}
          </span>
        </div>
        {(job.status === "queued" || job.status === "running") && (
          <button
            type="button"
            className="ghost text-xs px-2.5 py-1 text-red-400 hover:text-red-300 border-red-500/30"
            onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
            aria-label={t("Stop job")}
          >
            {t("Stop")}
          </button>
        )}
      </div>

      {job.status === "error" && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs font-mono"
        >
          {job.error}
        </div>
      )}

      {(job.status === "queued" || job.status === "running") && (
        <div
          className="loader"
          role="progressbar"
          aria-label="Job execution in progress"
          aria-valuetext={job.status}
          style={{ marginTop: 8 }}
        >
          <div className="loaderBar" />
        </div>
      )}

      {resultMsg && <p className="text-sm font-medium text-white">{resultMsg}</p>}

      {resultInfo && (
        <div>
          <p className="muted text-xs mb-1 font-semibold">{t("Analysis result")}</p>
          <pre className="log">{resultInfo}</pre>
        </div>
      )}

      {out && isAudioFile(out) && <AudioPlayer src={outputUrl(out)} filename={fileBasename(out)} />}

      {out && isImageFile(out) && (
        <div className="space-y-2">
          {/* biome-ignore lint/performance/noImgElement: user-generated plot, no optimizer benefit */}
          <img
            src={outputUrl(out)}
            alt={`Analysis plot result for job ${job.id}`}
            style={{ maxWidth: "100%", borderRadius: 8 }}
          />
          <p className="m-0">
            <a
              href={outputUrl(out)}
              download
              aria-label={`${t("Download image")}: ${fileBasename(out)}`}
              className="text-xs font-semibold hover:underline"
            >
              {t("Download image")}
            </a>{" "}
            <span className="muted text-xs font-mono">({fileBasename(out)})</span>
          </p>
        </div>
      )}

      {out && !isAudioFile(out) && !isImageFile(out) && (
        <p className="m-0">
          <a
            href={outputUrl(out)}
            download
            aria-label={`${t("Download result")}: ${fileBasename(out)}`}
            className="text-xs font-semibold hover:underline"
          >
            {t("Download result")}
          </a>{" "}
          <span className="muted text-xs font-mono">({fileBasename(out)})</span>
        </p>
      )}

      {sidecars.map((s) => (
        <p key={s.label} className="m-0">
          <a
            href={outputUrl(s.file)}
            download
            aria-label={`${t("Download")} ${s.label}: ${fileBasename(s.file)}`}
            className="text-xs font-semibold hover:underline"
          >
            {t("Download")} {s.label}
          </a>{" "}
          <span className="muted text-xs font-mono">({fileBasename(s.file)})</span>
        </p>
      ))}

      {!compact && job.logs.length > 0 && (
        <div className="pt-2">
          <p className="muted text-xs mb-1 font-semibold">{t("Engine logs")}</p>
          <pre className="log" role="log" aria-live="polite">
            {job.logs.slice(-60).join("\n")}
          </pre>
        </div>
      )}
    </section>
  );
}
