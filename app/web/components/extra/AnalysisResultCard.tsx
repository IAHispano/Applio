"use client";

import { Download, FileText, Image as ImageIcon, LineChart, StopCircle, Waves } from "lucide-react";
import { useEffect, useState } from "react";
import {
  errMsg,
  fetchJob,
  fileBasename,
  type Job,
  outputUrl,
  pollJob,
  stopJob,
} from "../../lib/api";
import { useI18n } from "../../lib/i18n";

interface AnalysisResultCardProps {
  jobId: string | null;
  title: string;
  type: "analyzer" | "f0";
}

export default function AnalysisResultCard({ jobId, title, type }: AnalysisResultCardProps) {
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

  if (!jobId) return null;

  if (error) {
    return (
      <div className="card border-red-500/30 bg-red-500/10 text-red-400 text-xs p-4">
        {error}
      </div>
    );
  }

  if (!job) {
    return (
      <div className="card p-5 text-center text-xs text-neutral-400">
        {t("Loading analysis…")}
      </div>
    );
  }

  const isRunning = job.status === "running" || job.status === "queued";
  const out = job.outputFile;
  const curveFile = typeof job.result?.curveFile === "string" ? job.result.curveFile : null;

  return (
    <div className="card space-y-4 animate-in fade-in duration-200">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          {type === "analyzer" ? (
            <Waves size={18} className="text-white shrink-0" />
          ) : (
            <LineChart size={18} className="text-white shrink-0" />
          )}
          <h3 className="text-base font-bold text-white m-0">{title}</h3>
          <span className={`badge ${job.status} text-[10px] ml-1`} role="status">
            {job.status === "done"
              ? t("Ready")
              : isRunning
                ? t("Analyzing…")
                : job.status === "error"
                  ? t("Failed")
                  : t("Queued")}
          </span>
        </div>

        {isRunning && (
          <button
            type="button"
            className="ghost h-7 px-2.5 text-xs text-red-400 hover:text-red-300 border-red-500/30 rounded-lg flex items-center gap-1.5"
            onClick={() => stopJob(job.id).catch((e) => setError(errMsg(e)))}
          >
            <StopCircle size={13} />
            <span>{t("Cancel")}</span>
          </button>
        )}
      </div>

      {/* Error state */}
      {job.status === "error" && (
        <div
          role="alert"
          className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
        >
          {job.error || t("Analysis operation failed.")}
        </div>
      )}

      {/* Running Progress Bar */}
      {isRunning && (
        <div className="py-4 space-y-3">
          <div className="loader" role="progressbar" aria-label={t("Analyzing audio…")}>
            <div className="loaderBar" />
          </div>
          <p className="text-xs text-neutral-400 text-center m-0">
            {type === "analyzer"
              ? t("Computing multi-band acoustic spectrogram and energy profiles…")
              : t("Calculating pitch contours across audio frames…")}
          </p>
        </div>
      )}

      {/* Done State: Visual Display */}
      {job.status === "done" && out && (
        <div className="space-y-4">
          {/* Plot Image */}
          <div className="rounded-xl overflow-hidden border border-white/10 bg-black/60 p-2">
            {/* biome-ignore lint/performance/noImgElement: user-generated analysis plot */}
            <img
              src={outputUrl(out)}
              alt={`${title} visualization`}
              className="w-full h-auto rounded-lg object-contain max-h-[480px]"
            />
          </div>

          {/* Action and Download Bar */}
          <div className="flex items-center justify-between gap-3 flex-wrap pt-1">
            <span className="text-xs text-neutral-400">
              {fileBasename(out)}
            </span>

            <div className="flex items-center gap-2">
              {curveFile && (
                <a
                  href={outputUrl(curveFile)}
                  download
                  className="ghost h-9 px-3.5 rounded-xl text-xs font-medium flex items-center gap-1.5 text-neutral-200 hover:text-white"
                >
                  <FileText size={14} className="shrink-0" />
                  <span>{t("Download F0 CSV")}</span>
                </a>
              )}

              <a
                href={outputUrl(out)}
                download
                className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-1.5 shrink-0"
              >
                <Download size={14} className="shrink-0" />
                <span>{t("Download Plot")}</span>
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
