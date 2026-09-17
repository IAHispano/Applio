"use client";

import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Loader2,
  RefreshCw,
  Sparkles,
  Terminal,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { apiGet, apiSend, errMsg, fetchJob, type Job } from "../../lib/api";

interface SetupCheck {
  id: string;
  label: string;
  status: "ok" | "missing" | "warn";
  detail: string;
}

interface SetupStatus {
  ready: boolean;
  checks: SetupCheck[];
  checkedAt: string;
}

interface FirstRunSetupProps {
  onComplete: () => void;
}

export default function FirstRunSetup({ onComplete }: FirstRunSetupProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [showLogs, setShowLogs] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  const autoStartedRef = useRef(false);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-start installation on mount
  useEffect(() => {
    if (autoStartedRef.current) return;
    autoStartedRef.current = true;

    async function start() {
      setError("");
      try {
        const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/install", "POST");
        setJobId(id);
      } catch (err) {
        setError(errMsg(err));
      }
    }

    start();
  }, []);

  // Poll setup job progress
  useEffect(() => {
    if (!jobId) return;
    let active = true;

    const timer = setInterval(async () => {
      try {
        const res = await fetchJob(jobId);
        if (!active) return;
        setJob(res.job);

        if (res.job.status === "done") {
          clearInterval(timer);
          // Verify final status
          apiGet<SetupStatus>("/api/setup/status?refresh=1")
            .then((s) => {
              if (s.ready) {
                setCountdown(2);
              } else {
                setError("Setup finished, but some required checks are incomplete.");
              }
            })
            .catch(() => setCountdown(2));
        } else if (res.job.status === "error") {
          clearInterval(timer);
          setError(res.job.error || "Setup failed. Please check the console log below.");
        }
      } catch {
        /* retry next tick */
      }
    }, 1000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [jobId]);

  // Auto-scroll logs when drawer is open
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll when logs length changes
  useEffect(() => {
    if (showLogs) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [job?.logs.length, showLogs]);

  // Countdown to enter studio
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      onComplete();
      return;
    }
    const t = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown, onComplete]);

  // Retry setup
  async function retry() {
    setError("");
    setJob(null);
    setCountdown(null);
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/install", "POST");
      setJobId(id);
    } catch (err) {
      setError(errMsg(err));
    }
  }

  // Derive step progress and active phase from logs
  const logsText = useMemo(() => job?.logs.join("\n") || "", [job?.logs]);

  const steps = useMemo(() => {
    const isDone = job?.status === "done";
    const hasPy = logsText.includes("App virtualenv") || logsText.includes("Creating app virtualenv");
    const hasTorch = logsText.includes("Installing engine packages") || logsText.includes("torch");
    const hasDeps = logsText.includes("Using Python env") || logsText.includes("web dependencies");
    const hasModels =
      logsText.includes("Downloading base voice models") || logsText.includes("prerequisites");
    const hasVerified = isDone || logsText.includes("Setup complete") || logsText.includes("checks passed");

    return [
      {
        id: "env",
        title: "Python Virtual Environment",
        desc: "Isolated Python runtime (.venv)",
        status: hasPy ? (hasTorch ? "done" : "running") : "running",
      },
      {
        id: "torch",
        title: "PyTorch & GPU Acceleration",
        desc: "CUDA, MPS, or high-performance CPU backend",
        status: hasTorch ? (hasDeps ? "done" : "running") : "pending",
      },
      {
        id: "engine",
        title: "RVC Engine Packages",
        desc: "Faiss indexer, FCPE, Crepe, and audio processors",
        status: hasDeps ? (hasModels ? "done" : "running") : "pending",
      },
      {
        id: "models",
        title: "Base Acoustic Checkpoints",
        desc: "HuBERT feature extractor & RMVPE pitch models",
        status: hasModels ? (hasVerified ? "done" : "running") : "pending",
      },
      {
        id: "ready",
        title: "Finalizing Applio Studio",
        desc: "Configuring workspaces and studio pipelines",
        status: isDone ? "done" : hasVerified ? "running" : "pending",
      },
    ];
  }, [job?.status, logsText]);

  // Estimated progress percentage
  const progressPercent = useMemo(() => {
    if (job?.status === "done") return 100;
    const completedCount = steps.filter((s) => s.status === "done").length;
    const hasRunning = steps.some((s) => s.status === "running");
    const base = completedCount * 20;
    return Math.min(95, Math.max(8, base + (hasRunning ? 10 : 0)));
  }, [job?.status, steps]);

  return (
    <div className="min-h-full flex flex-col items-center justify-center p-4 sm:p-6 max-w-3xl mx-auto">
      <div className="w-full space-y-6">
        {/* Brand Header */}
        <div className="text-center space-y-3">
          <div className="relative inline-flex items-center justify-center">
            <div className="w-16 h-16 rounded-2xl bg-white/10 border border-white/15 flex items-center justify-center shadow-xl">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            {job?.status !== "done" && !error && (
              <span className="absolute -top-1 -right-1 flex h-4 w-4">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white/40 opacity-75" />
                <span className="relative inline-flex rounded-full h-4 w-4 bg-white" />
              </span>
            )}
          </div>

          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white m-0">
              {job?.status === "done" ? "Applio Studio is Ready" : "Preparing Applio for First Use"}
            </h1>
            <p className="text-sm text-neutral-400 max-w-md mx-auto leading-relaxed m-0">
              {job?.status === "done"
                ? "Your environment, voice conversion engine, and acoustic models are prepared."
                : "Automatically setting up dependencies, neural models, and audio engines. This only happens on first launch."}
            </p>
          </div>
        </div>

        {/* Progress Bar Card */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between text-xs text-neutral-300">
            <span className="font-medium flex items-center gap-2">
              {job?.status === "done" ? (
                <CheckCircle2 size={16} className="text-emerald-400" />
              ) : (
                <Loader2 size={16} className="animate-spin text-white" />
              )}
              <span>
                {job?.status === "done"
                  ? "Installation Complete"
                  : `Automated Setup in Progress (${progressPercent}%)`}
              </span>
            </span>
            <span className="font-mono text-neutral-400">
              {steps.filter((s) => s.status === "done").length} / {steps.length} Steps
            </span>
          </div>

          <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden relative">
            <div
              className="h-full bg-white transition-all duration-500 rounded-full"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          {/* Checklist */}
          <div className="space-y-2 pt-2">
            {steps.map((step) => {
              const isDone = step.status === "done";
              const isRunning = step.status === "running";

              return (
                <div
                  key={step.id}
                  className={`flex items-center justify-between p-3 rounded-xl border transition-all duration-200 ${
                    isDone
                      ? "bg-emerald-500/[0.05] border-emerald-500/20 text-neutral-200"
                      : isRunning
                        ? "bg-white/[0.06] border-white/20 text-white shadow-sm"
                        : "bg-white/[0.02] border-white/5 text-neutral-500"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="shrink-0">
                      {isDone ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : isRunning ? (
                        <Loader2 className="w-4 h-4 text-white animate-spin" />
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-neutral-600" />
                      )}
                    </div>
                    <div>
                      <p className="text-xs font-semibold m-0">{step.title}</p>
                      <p className="text-[11px] text-neutral-400 m-0">{step.desc}</p>
                    </div>
                  </div>

                  <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded">
                    {isDone ? (
                      <span className="text-emerald-400">Ready</span>
                    ) : isRunning ? (
                      <span className="text-white animate-pulse">Installing…</span>
                    ) : (
                      <span className="text-neutral-500">Queued</span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="space-y-2 flex-1">
              <p className="text-xs font-semibold text-red-300 m-0">Setup Encountered an Issue</p>
              <p className="text-xs text-red-200/80 m-0 leading-relaxed">{error}</p>
              <button
                type="button"
                className="cta text-xs px-3 py-1.5 flex items-center gap-1.5"
                onClick={retry}
              >
                <RefreshCw size={13} />
                <span>Retry Automated Setup</span>
              </button>
            </div>
          </div>
        )}

        {/* Action Controls */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-1">
          <button
            type="button"
            className="ghost text-xs px-3 py-1.5 flex items-center gap-1.5 text-neutral-400 hover:text-white"
            onClick={() => setShowLogs(!showLogs)}
          >
            <Terminal size={14} />
            <span>{showLogs ? "Hide Console Output" : "View Live Console Output"}</span>
            {showLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {job?.status === "done" && (
            <button
              type="button"
              className="cta px-6 py-2.5 text-sm flex items-center gap-2 shadow-lg"
              onClick={onComplete}
            >
              <span>{countdown !== null ? `Entering Studio (${countdown}s)…` : "Launch Studio"}</span>
              <ArrowRight size={15} />
            </button>
          )}
        </div>

        {/* Live Terminal Log Drawer */}
        {showLogs && (
          <div className="rounded-xl border border-white/10 bg-black/70 p-4 font-mono text-xs text-neutral-300 max-h-64 overflow-y-auto space-y-1 shadow-inner">
            {job?.logs.length ? (
              job.logs.map((log, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: append-only terminal logs
                <p key={`log-${i}`} className="m-0 leading-relaxed text-[11px] break-all">
                  <span className="text-neutral-500 mr-2">&gt;</span>
                  {log}
                </p>
              ))
            ) : (
              <p className="text-neutral-500 m-0 italic">Initializing setup stream…</p>
            )}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
