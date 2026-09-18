"use client";

import { ChevronDown, Download, FileAudio, FileCheck, Image as ImageIcon, StopCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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

export interface JobPanelProps {
  jobId: string | null;
  compact?: boolean;
  showLogs?: boolean;
}

// Polls a job, shows status, and renders its output file cleanly without CLI clutter.
export default function JobPanel({ jobId, compact, showLogs = false }: JobPanelProps) {
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
      .catch((e) => setError(errMsg(e)));
    return () => stop();
  }, [jobId]);

  const cleanedLogs = useMemo(() => {
    if (!job?.logs) return [];
    return job.logs
      .map((l) =>
        l
          .replace(/^\$ python.*$/i, "")
          .replace(/^\[(stdout|stderr)\]\s*/i, "")
          .trim(),
      )
      .filter((l) => l.length > 0);
  }, [job?.logs]);

  if (!jobId) return null;
  if (error) return <p style={{ color: "var(--err)" }}>{error}</p>;
  if (!job) return <p className="muted text-xs">{t("Loading activity…")}</p>;

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
    <section className="card space-y-3 animate-in fade-in duration-200" aria-label={t("Task Activity")}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`badge ${job.status}`} role="status" aria-label={`Status: ${job.status}`}>
            {job.status === "done"
              ? t("Completed")
              : job.status === "running"
                ? t("In Progress")
                : job.status === "error"
                  ? t("Failed")
                  : t("Queued")}
          </span>
          <span className="text-neutral-400 text-xs font-medium">
            {t("Activity")}
          </span>
        </div>
        {(job.status === "queued" || job.status === "running") && (
          <button
            type="button"
            className="ghost text-xs h-7 px-2.5 text-red-400 hover:text-red-300 border-red-500/30 rounded-lg flex items-center gap-1"
            onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
            aria-label={t("Cancel")}
          >
            <StopCircle size={13} />
            <span>{t("Cancel")}</span>
          </button>
        )}
      </div>

      {job.status === "error" && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs"
        >
          {job.error || t("Operation failed.")}
        </div>
      )}

      {(job.status === "queued" || job.status === "running") && (
        <div
          className="loader"
          role="progressbar"
          aria-label="Execution in progress"
          aria-valuetext={job.status}
          style={{ marginTop: 8 }}
        >
          <div className="loaderBar" />
        </div>
      )}

      {resultMsg && <p className="text-xs font-medium text-white m-0">{resultMsg}</p>}

      {resultInfo && (
        <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-1">
          <p className="text-neutral-400 text-xs font-medium m-0">{t("Analysis details")}</p>
          <p className="text-xs text-neutral-200 leading-relaxed m-0 whitespace-pre-wrap">{resultInfo}</p>
        </div>
      )}

      {out && isAudioFile(out) && <AudioPlayer src={outputUrl(out)} filename={fileBasename(out)} />}

      {out && isImageFile(out) && (
        <div className="space-y-2">
          {/* biome-ignore lint/performance/noImgElement: user-generated plot */}
          <img
            src={outputUrl(out)}
            alt={`Analysis plot result for job ${job.id}`}
            className="w-full h-auto rounded-xl border border-white/10"
          />
          <div className="flex items-center justify-between pt-1">
            <span className="text-xs text-neutral-400">({fileBasename(out)})</span>
            <a
              href={outputUrl(out)}
              download
              className="cta h-8 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <Download size={13} />
              <span>{t("Download image")}</span>
            </a>
          </div>
        </div>
      )}

      {out && !isAudioFile(out) && !isImageFile(out) && (
        <div className="flex items-center justify-between p-3 rounded-xl bg-black/30 border border-white/5">
          <span className="text-xs text-neutral-300 font-medium truncate">{fileBasename(out)}</span>
          <a
            href={outputUrl(out)}
            download
            className="cta h-8 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 shrink-0"
          >
            <Download size={13} />
            <span>{t("Download")}</span>
          </a>
        </div>
      )}

      {sidecars.map((s) => (
        <div key={s.label} className="flex items-center justify-between p-2.5 rounded-xl bg-black/30 border border-white/5">
          <span className="text-xs text-neutral-300">{s.label}: {fileBasename(s.file)}</span>
          <a
            href={outputUrl(s.file)}
            download
            className="ghost h-7 px-2.5 rounded-lg text-xs font-medium flex items-center gap-1 text-neutral-300 hover:text-white shrink-0"
          >
            <Download size={12} />
            <span>{t("Download")}</span>
          </a>
        </div>
      ))}

      {showLogs && !compact && cleanedLogs.length > 0 && (
        <details className="pt-2 text-xs text-neutral-400 group border-t border-white/5">
          <summary className="cursor-pointer hover:text-white transition-colors py-1 flex items-center gap-1.5 select-none font-medium">
            <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
            <span>{t("Activity Details")}</span>
          </summary>
          <div className="mt-2 max-h-48 overflow-y-auto text-xs p-3 rounded-xl bg-black/50 border border-white/10 space-y-0.5" role="log">
            {cleanedLogs.slice(-60).map((l, idx) => (
              <p key={idx} className="m-0 leading-relaxed text-neutral-300">{l}</p>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
