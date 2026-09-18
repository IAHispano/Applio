"use client";

import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Database,
  Download,
  FileAudio,
  Radio,
  RefreshCw,
  Sparkles,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import JobPanel from "../components/JobPanel";
import FirstRunSetup from "../components/setup/FirstRunSetup";
import { apiGet, apiSend, errMsg } from "../lib/api";
import { useI18n } from "../lib/i18n";

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

interface ModelsSummary {
  models: string[];
  indexes: string[];
  audios: string[];
}

export default function Home() {
  const router = useRouter();
  const { t } = useI18n();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [modelsData, setModelsData] = useState<ModelsSummary | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);

  const refresh = useCallback(async (force = false) => {
    try {
      const [setupRes, modelsRes] = await Promise.all([
        apiGet<SetupStatus>(`/api/setup/status${force ? "?refresh=1" : ""}`),
        apiGet<ModelsSummary>("/api/models").catch(() => null),
      ]);
      setStatus(setupRes);
      if (modelsRes) setModelsData(modelsRes);
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function prerequisites() {
    setError("");
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/prerequisites", "POST");
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  const passedChecks = status?.checks.filter((c) => c.status === "ok").length ?? 0;
  const totalChecks = status?.checks.length ?? 0;
  const modelCount = modelsData?.models.length ?? 0;
  const audioCount = modelsData?.audios.length ?? 0;

  if (!status) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="flex flex-col items-center gap-4 text-neutral-400 w-full max-w-xs text-center">
          <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden relative">
            <div className="h-full bg-white rounded-full animate-pulse w-3/4" />
          </div>
          <span className="text-xs font-medium text-neutral-300">{t("Connecting to engine…")}</span>
          {error && (
            <div className="text-center mt-2">
              <p className="text-xs text-red-400 mb-2">{error}</p>
              <button type="button" className="ghost text-xs py-1 px-3" onClick={() => refresh(true)}>
                {t("Retry Connection")}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (!status.ready) {
    return <FirstRunSetup onComplete={() => refresh(true)} />;
  }

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      {/* Applio Header Card */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] p-6 sm:p-7">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2.5">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-xs font-medium text-neutral-300">
                <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
                {t("Ready")}
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white m-0">Applio</h1>
            <p className="text-neutral-400 text-xs sm:text-sm leading-relaxed m-0">
              {t("High-performance AI voice cloning, real-time conversion, and model training.")}
            </p>
          </div>

          <div className="flex items-center gap-2.5 shrink-0">
            <button
              type="button"
              className="cta flex items-center gap-2 text-xs sm:text-sm py-2 px-4"
              onClick={() => router.push("/inference")}
            >
              <Sparkles className="w-4 h-4" />
              <span>{t("Open Inference")}</span>
            </button>
            <button
              type="button"
              className="ghost flex items-center gap-2 text-xs sm:text-sm py-2 px-3.5"
              onClick={() => router.push("/realtime")}
            >
              <Radio className="w-4 h-4" />
              <span>{t("Realtime")}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Glanceable Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Link
          href="/models"
          className="p-4 rounded-xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20 transition-all flex items-center justify-between group"
        >
          <div className="space-y-1">
            <span className="text-xs text-neutral-400 font-medium">{t("Installed Models")}</span>
            <p className="text-xl font-bold text-white m-0 tracking-tight">
              {modelCount} {modelCount === 1 ? t("Model") : t("Models")}
            </p>
          </div>
          <div className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center text-neutral-400 group-hover:text-white transition-colors">
            <Database className="w-4 h-4" />
          </div>
        </Link>

        <Link
          href="/inference"
          className="p-4 rounded-xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20 transition-all flex items-center justify-between group"
        >
          <div className="space-y-1">
            <span className="text-xs text-neutral-400 font-medium">{t("Audio Outputs")}</span>
            <p className="text-xl font-bold text-white m-0 tracking-tight">
              {audioCount} {audioCount === 1 ? t("File") : t("Files")}
            </p>
          </div>
          <div className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center text-neutral-400 group-hover:text-white transition-colors">
            <FileAudio className="w-4 h-4" />
          </div>
        </Link>

        <div className="p-4 rounded-xl border border-white/10 bg-white/[0.02] flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-xs text-neutral-400 font-medium">{t("Engine Status")}</span>
            <p className="text-xl font-bold text-white m-0 tracking-tight">
              {passedChecks}/{totalChecks} {t("Checks OK")}
            </p>
          </div>
          <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center text-white">
            <CheckCircle2 className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Focused Workflows (Clean 2-card layout) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link
          href="/inference"
          className="group p-5 rounded-2xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.06] hover:border-white/20 transition-all flex flex-col justify-between"
        >
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center text-white">
                <Sparkles className="w-4 h-4" />
              </div>
              <span className="text-[11px] font-semibold text-neutral-400">
                {t("Audio Conversion")}
              </span>
            </div>
            <h2 className="text-base font-bold text-neutral-100 group-hover:text-white transition-colors m-0">
              {t("Voice Inference")}
            </h2>
            <p className="text-xs text-neutral-400 leading-relaxed m-0">
              {t(
                "Transform audio files or batches with your custom voice models, pitch, and timbre controls.",
              )}
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-xs font-medium text-neutral-300 group-hover:text-white pt-4 mt-3 border-t border-white/5">
            <span>{t("Start Converting")}</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link
          href="/realtime"
          className="group p-5 rounded-2xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.06] hover:border-white/20 transition-all flex flex-col justify-between"
        >
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center text-white">
                <Radio className="w-4 h-4" />
              </div>
              <span className="text-[11px] font-semibold text-neutral-400">
                {t("Live Audio")}
              </span>
            </div>
            <h2 className="text-base font-bold text-neutral-100 group-hover:text-white transition-colors m-0">
              {t("Realtime")}
            </h2>
            <p className="text-xs text-neutral-400 leading-relaxed m-0">
              {t("Low-latency microphone voice conversion for live streams, voice calls, and monitoring.")}
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-xs font-medium text-neutral-300 group-hover:text-white pt-4 mt-3 border-t border-white/5">
            <span>{t("Launch Live Audio")}</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>
      </div>

      {/* Clean Collapsible Diagnostics Section */}
      <section className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2.5">
            <Activity size={16} className="text-white" />
            <h3 className="text-sm font-semibold text-neutral-200 m-0">{t("System Diagnostics")}</h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-neutral-300 border border-white/5">
              {passedChecks}/{totalChecks} {t("passed")}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="ghost text-xs py-1 px-2.5 flex items-center gap-1.5"
              onClick={prerequisites}
              title={t("Download base models & checkpoints")}
            >
              <Download className="w-3 h-3" />
              <span>{t("Prerequisites")}</span>
            </button>
            <button
              type="button"
              className="ghost text-xs py-1 px-2.5 flex items-center gap-1.5"
              onClick={() => refresh(true)}
              title={t("Re-run diagnostic checks")}
            >
              <RefreshCw className="w-3 h-3" />
              <span>{t("Re-check")}</span>
            </button>
            <button
              type="button"
              className="ghost text-xs py-1 px-2.5 flex items-center gap-1"
              onClick={() => setShowDetails(!showDetails)}
            >
              <span>{showDetails ? t("Hide Details") : t("Show Details")}</span>
              {showDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            aria-live="assertive"
            className="mx-4 mb-4 flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs"
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {showDetails && (
          <div className="border-t border-white/5 p-4 bg-black/20 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {status.checks.map((c) => {
              const isOk = c.status === "ok";
              const isWarn = c.status === "warn";
              return (
                <div
                  key={c.id}
                  className="flex items-start gap-2 p-2.5 rounded-lg border border-white/5 bg-white/[0.01]"
                >
                  <div className="mt-0.5 shrink-0">
                    {isOk ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-white" />
                    ) : isWarn ? (
                      <AlertCircle className="w-3.5 h-3.5 text-neutral-300" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-neutral-400" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-medium text-neutral-200 truncate">{c.label}</span>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded ${
                          isOk
                            ? "text-white bg-white/10"
                            : isWarn
                              ? "text-neutral-300 bg-white/10"
                              : "text-neutral-400 bg-white/5"
                        }`}
                      >
                        {c.status}
                      </span>
                    </div>
                    <p className="text-[10px] text-neutral-500 mt-0.5 truncate m-0">{c.detail}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* Active Setup / Prerequisite Job */}
      <JobPanel jobId={jobId} />
    </div>
  );
}
