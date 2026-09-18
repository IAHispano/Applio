"use client";

import {
  AlertCircle,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Copy,
  Cpu,
  Download,
  Flame,
  Layers,
  Pause,
  Play,
  RotateCcw,
  Search,
  Sparkles,
  StopCircle,
  Terminal,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { errMsg, fetchJob, type Job, pollJob, stopJob } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { toast } from "../../lib/toast";

interface TrainingConsoleProps {
  jobId: string | null;
  modelName: string;
  totalEpochs?: number;
  onStop?: () => Promise<void> | void;
}

type LogLevel = "all" | "epochs" | "checkpoints" | "errors";

interface ParsedMetrics {
  currentEpoch: number | null;
  currentStep: number | null;
  loss: string | null;
  activePhase: 1 | 2 | 3 | 4 | 5; // 1: Preprocess, 2: Extract, 3: Train, 4: Index, 5: Done
}

export default function TrainingConsole({
  jobId,
  modelName,
  totalEpochs = 200,
  onStop,
}: TrainingConsoleProps) {
  const { t } = useI18n();
  const router = useRouter();

  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [filterText, setFilterText] = useState("");
  const [level, setLevel] = useState<LogLevel>("all");
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const logEndRef = useRef<HTMLDivElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Poll active training job
  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setElapsedSeconds(0);
      return;
    }
    setError("");
    let stop = () => {};
    fetchJob(jobId)
      .then(({ job: j }) => {
        setJob(j);
        if (j.status !== "done" && j.status !== "error") {
          stop = pollJob(jobId, setJob);
        }
      })
      .catch((e) => setError(errMsg(e)));
    return () => stop();
  }, [jobId]);

  // Elapsed timer while job is active
  useEffect(() => {
    if (!job || (job.status !== "running" && job.status !== "queued")) return;
    const interval = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [job?.status]);

  // Clean and filter logs
  const cleanedLogs = useMemo(() => {
    if (!job?.logs) return [];
    return job.logs
      .map((l) =>
        l
          .replace(/^\$ python core\.py.*$/i, "")
          .replace(/^\[(stdout|stderr)\]\s*/i, "")
          .trim(),
      )
      .filter((l) => l.length > 0);
  }, [job?.logs]);

  // Real-time metric parser from logs
  const metrics: ParsedMetrics = useMemo(() => {
    let currentEpoch: number | null = null;
    let currentStep: number | null = null;
    let loss: string | null = null;
    let activePhase: 1 | 2 | 3 | 4 | 5 = 1;

    if (job?.status === "done") {
      activePhase = 5;
    }

    for (let i = cleanedLogs.length - 1; i >= 0; i--) {
      const line = cleanedLogs[i];

      // Parse phase if not done
      if (activePhase !== 5) {
        if (line.includes("index") || line.includes("faiss") || line.includes("trained_IVF")) {
          activePhase = 4;
        } else if (line.includes("epoch=") || line.includes("step=") || line.includes("epoch:")) {
          activePhase = 3;
        } else if (
          line.includes("extract") ||
          line.includes("f0") ||
          line.includes("rmvpe") ||
          line.includes("contentvec")
        ) {
          activePhase = 2;
        } else if (line.includes("preprocess") || line.includes("sliced") || line.includes("audio")) {
          activePhase = 1;
        }
      }

      // Parse epoch
      if (currentEpoch === null) {
        const mEpoch = line.match(/epoch=(\d+)/i) || line.match(/epoch:\s*(\d+)/i);
        if (mEpoch) currentEpoch = Number.parseInt(mEpoch[1], 10);
      }

      // Parse step
      if (currentStep === null) {
        const mStep = line.match(/step=(\d+)/i) || line.match(/step:\s*(\d+)/i);
        if (mStep) currentStep = Number.parseInt(mStep[1], 10);
      }

      // Parse loss
      if (loss === null) {
        const mLoss =
          line.match(/lowest_value=([0-9\.]+)/i) ||
          line.match(/loss_gen_all=([0-9\.]+)/i) ||
          line.match(/loss:\s*([0-9\.]+)/i);
        if (mLoss) loss = mLoss[1];
      }

      if (currentEpoch !== null && currentStep !== null && loss !== null && activePhase !== 1) {
        break;
      }
    }

    return { currentEpoch, currentStep, loss, activePhase };
  }, [cleanedLogs, job?.status]);

  // Filter logs for user
  const displayedLogs = useMemo(() => {
    return cleanedLogs.filter((line) => {
      // Level filter
      if (level === "epochs" && !line.includes("epoch=") && !line.includes("epoch:")) {
        return false;
      }
      if (
        level === "checkpoints" &&
        !line.toLowerCase().includes("save") &&
        !line.toLowerCase().includes("checkpoint") &&
        !line.toLowerCase().includes(".pth")
      ) {
        return false;
      }
      if (
        level === "errors" &&
        !line.toLowerCase().includes("error") &&
        !line.toLowerCase().includes("fail") &&
        !line.toLowerCase().includes("exception")
      ) {
        return false;
      }
      // Text filter
      if (filterText && !line.toLowerCase().includes(filterText.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [cleanedLogs, level, filterText]);

  // Auto-scroll effect
  // biome-ignore lint/correctness/useExhaustiveDependencies: autoScroll toggle + new log updates
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [displayedLogs, autoScroll]);

  if (!jobId) return null;

  function copyAllLogs() {
    if (cleanedLogs.length === 0) return;
    navigator.clipboard.writeText(cleanedLogs.join("\n"));
    setCopied(true);
    toast(t("Console logs copied to clipboard"));
    setTimeout(() => setCopied(false), 2000);
  }

  async function handleStop() {
    if (onStop) {
      await onStop();
      return;
    }
    if (job?.id) {
      try {
        await stopJob(job.id);
        toast(t("Training job stop requested"));
      } catch (e) {
        setError(errMsg(e));
      }
    }
  }

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const progressPercent =
    metrics.currentEpoch && totalEpochs
      ? Math.min(100, Math.round((metrics.currentEpoch / totalEpochs) * 100))
      : null;

  return (
    <div className="card space-y-5 animate-in fade-in duration-200" aria-label={t("Training Activity Console")}>
      {/* 1. Header Bar */}
      <div className="flex items-center justify-between gap-3 border-b border-white/10 pb-4 flex-wrap">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
            <Cpu size={20} className="text-white" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-base font-bold text-white m-0 truncate">
                {modelName ? `${t("Training:")} ${modelName}` : t("Training Project")}
              </h3>
              <span
                className={`badge ${job?.status || "queued"} text-[10px] px-2 py-0.5 font-medium`}
                role="status"
              >
                {job?.status === "done"
                  ? t("Completed")
                  : job?.status === "running"
                    ? t("In Progress")
                    : job?.status === "error"
                      ? t("Failed")
                      : t("Queued")}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-neutral-400 mt-0.5">
              <span className="flex items-center gap-1">
                <Clock size={12} className="text-neutral-400" />
                <span>{formatTime(elapsedSeconds)}</span>
              </span>
              <span>•</span>
              <span>{t("Integrated Engine Activity")}</span>
            </div>
          </div>
        </div>

        {/* Header Action Controls */}
        <div className="flex items-center gap-2 shrink-0">
          {(job?.status === "running" || job?.status === "queued") && (
            <button
              type="button"
              className="ghost h-8 px-3 text-xs font-medium text-red-400 hover:text-red-300 border-red-500/30 rounded-xl flex items-center gap-1.5"
              onClick={handleStop}
              aria-label={t("Stop training job")}
            >
              <StopCircle size={14} className="shrink-0" />
              <span>{t("Stop Training")}</span>
            </button>
          )}

          <button
            type="button"
            className="ghost h-8 px-2.5 text-xs text-neutral-400 hover:text-white rounded-xl flex items-center gap-1"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? t("Expand console") : t("Collapse console")}
          >
            {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            <span className="hidden sm:inline">{collapsed ? t("Expand") : t("Collapse")}</span>
          </button>
        </div>
      </div>

      {/* Runtime Error alert */}
      {error && (
        <div
          role="alert"
          className="p-3.5 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs"
        >
          {error}
        </div>
      )}

      {/* 2. Pipeline Phase Stepper */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[
          { step: 1, title: t("Preprocessing"), desc: t("Slice & Normalize") },
          { step: 2, title: t("Feature Extraction"), desc: t("Pitch & Embeddings") },
          { step: 3, title: t("Model Training"), desc: t("Epochs & Loss") },
          { step: 4, title: t("Index Building"), desc: t("FAISS Feature Index") },
        ].map((phase) => {
          const isDone = metrics.activePhase > phase.step || job?.status === "done";
          const isCurrent = metrics.activePhase === phase.step && job?.status === "running";

          return (
            <div
              key={phase.step}
              className={`p-3 rounded-xl border transition-colors ${
                isDone
                  ? "bg-white/[0.03] border-white/20 text-white"
                  : isCurrent
                    ? "bg-white/10 border-white/40 text-white shadow-sm"
                    : "bg-black/20 border-white/5 text-neutral-500"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-semibold tracking-wider uppercase">
                  {t("Phase")} {phase.step}
                </span>
                {isDone ? (
                  <CheckCircle2 size={13} className="text-white" />
                ) : isCurrent ? (
                  <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-neutral-600" />
                )}
              </div>
              <p className="text-xs font-semibold m-0 text-white truncate">{phase.title}</p>
              <p className="text-[10px] text-neutral-400 m-0 mt-0.5 truncate">{phase.desc}</p>
            </div>
          );
        })}
      </div>

      {/* 3. Live KPI Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-neutral-400 text-xs block">{t("Epoch Progress")}</span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-base font-bold text-white">
              {metrics.currentEpoch !== null ? metrics.currentEpoch : "—"}
            </span>
            <span className="text-xs text-neutral-400">/ {totalEpochs}</span>
          </div>
          {progressPercent !== null && (
            <div className="w-full bg-white/10 rounded-full h-1 mt-2 overflow-hidden">
              <div
                className="bg-white h-full rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          )}
        </div>

        <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-neutral-400 text-xs block">{t("Training Steps")}</span>
          <span className="text-base font-bold text-white block">
            {metrics.currentStep !== null ? metrics.currentStep.toLocaleString() : "—"}
          </span>
          <span className="text-[10px] text-neutral-400 block">{t("Gradient updates")}</span>
        </div>

        <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-neutral-400 text-xs block">{t("Generator Loss")}</span>
          <span className="text-base font-bold text-white block">
            {metrics.loss !== null ? metrics.loss : "—"}
          </span>
          <span className="text-[10px] text-neutral-400 block">{t("Lowest rolling loss")}</span>
        </div>

        <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-neutral-400 text-xs block">{t("Active Status")}</span>
          <span className="text-base font-bold text-white block capitalize">
            {job?.status === "running" ? t("Training") : job?.status || t("Idle")}
          </span>
          <span className="text-[10px] text-neutral-400 block truncate">
            {metrics.activePhase === 1
              ? t("Slicing audio")
              : metrics.activePhase === 2
                ? t("Extracting pitch")
                : metrics.activePhase === 3
                  ? t("Training network")
                  : metrics.activePhase === 4
                    ? t("Building index")
                    : t("Ready")}
          </span>
        </div>
      </div>

      {/* 4. Celebratory Completion Banner */}
      {job?.status === "done" && (
        <div className="p-4 rounded-xl border border-white/20 bg-white/5 space-y-3">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-white flex items-center justify-center shrink-0">
              <Check size={16} className="text-black" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white m-0">{t("Training Completed Successfully")}</h4>
              <p className="text-xs text-neutral-400 m-0 mt-0.5">
                {t("Your voice model weights and feature index have been saved and are ready to use.")}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 pt-1">
            <button
              type="button"
              className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
              onClick={() => router.push(`/inference?model=${encodeURIComponent(`logs/${modelName}/${modelName}.pth`)}`)}
            >
              <span>{t("Test in Inference")}</span>
              <ArrowRight size={14} className="shrink-0" />
            </button>
            <Link
              href="/models"
              className="ghost h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-1.5 text-neutral-300 hover:text-white"
            >
              <Layers size={14} />
              <span>{t("View in Models Library")}</span>
            </Link>
          </div>
        </div>
      )}

      {/* 5. Embedded App Console Output */}
      {!collapsed && (
        <div className="space-y-2 pt-1">
          {/* Console Controls Bar */}
          <div className="flex items-center justify-between gap-2 flex-wrap text-xs text-neutral-400">
            <div className="flex items-center gap-1.5">
              <Terminal size={14} className="text-white" />
              <span className="font-semibold text-white">{t("Activity Log")}</span>
              <span className="text-[11px] text-neutral-400">({displayedLogs.length} events)</span>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              {/* Level Filter Pills */}
              <div className="flex items-center bg-black/40 p-0.5 rounded-lg border border-white/5 text-[11px]">
                {(["all", "epochs", "checkpoints", "errors"] as LogLevel[]).map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    className={`px-2 py-0.5 rounded-md capitalize transition-colors ${
                      level === lvl ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-neutral-200"
                    }`}
                    onClick={() => setLevel(lvl)}
                  >
                    {lvl === "all" ? t("All") : lvl === "epochs" ? t("Epochs") : lvl === "checkpoints" ? t("Saves") : t("Errors")}
                  </button>
                ))}
              </div>

              {/* Filter search input */}
              <div className="relative">
                <input
                  type="text"
                  placeholder={t("Filter logs…")}
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  className="h-7 w-32 sm:w-44 text-xs px-2 rounded-lg bg-black/40 border border-white/5 text-white placeholder:text-neutral-400"
                />
              </div>

              {/* Auto scroll toggle */}
              <button
                type="button"
                className={`ghost h-7 px-2 text-[11px] rounded-lg flex items-center gap-1 ${
                  autoScroll ? "text-white" : "text-neutral-400"
                }`}
                onClick={() => setAutoScroll(!autoScroll)}
                title={t("Toggle auto-scroll")}
              >
                <span>{t("Auto-scroll")}</span>
                <div className={`w-1.5 h-1.5 rounded-full ${autoScroll ? "bg-white" : "bg-neutral-600"}`} />
              </button>

              {/* Copy logs */}
              <button
                type="button"
                className="ghost h-7 px-2 text-[11px] text-neutral-300 hover:text-white rounded-lg flex items-center gap-1"
                onClick={copyAllLogs}
                aria-label={t("Copy logs")}
              >
                {copied ? <Check size={12} className="text-white" /> : <Copy size={12} />}
                <span>{copied ? t("Copied") : t("Copy")}</span>
              </button>
            </div>
          </div>

          {/* Console Output Window */}
          <div
            ref={logContainerRef}
            className="h-64 sm:h-72 overflow-y-auto rounded-xl bg-black/60 border border-white/10 p-3 space-y-1 text-xs text-neutral-300 select-text"
            role="log"
            aria-live="polite"
          >
            {displayedLogs.length === 0 ? (
              <p className="text-neutral-400 text-xs italic m-0 p-2">
                {job?.status === "queued"
                  ? t("Queued for training execution…")
                  : t("Waiting for activity stream…")}
              </p>
            ) : (
              displayedLogs.map((line, idx) => {
                const isEpoch = line.includes("epoch=") || line.includes("epoch:");
                const isSave =
                  line.toLowerCase().includes("save") ||
                  line.toLowerCase().includes("checkpoint") ||
                  line.toLowerCase().includes(".pth");
                const isErr =
                  line.toLowerCase().includes("error") ||
                  line.toLowerCase().includes("fail") ||
                  line.toLowerCase().includes("traceback");

                return (
                  <div
                    key={`${idx}-${line.slice(0, 20)}`}
                    className={`leading-relaxed break-words py-0.5 flex items-start gap-2 ${
                      isErr
                        ? "text-red-400 font-medium"
                        : isSave
                          ? "text-neutral-100 font-semibold bg-white/5 px-2 rounded"
                          : isEpoch
                            ? "text-white"
                            : "text-neutral-300"
                    }`}
                  >
                    <span className="text-[10px] text-neutral-400 select-none shrink-0 w-7 text-right">
                      {idx + 1}
                    </span>
                    <span className="flex-1">{line}</span>
                  </div>
                );
              })
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
