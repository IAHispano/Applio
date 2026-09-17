"use client";

import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Layers,
  Mic,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Wrench,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import JobPanel from "../components/JobPanel";
import { apiGet, apiSend, errMsg } from "../lib/api";

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

export default function Home() {
  const router = useRouter();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async (force = false) => {
    try {
      setStatus(await apiGet<SetupStatus>(`/api/setup/status${force ? "?refresh=1" : ""}`));
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function install() {
    setError("");
    setBusy(true);
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/install", "POST");
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function prerequisites() {
    setError("");
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/prerequisites", "POST");
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  const ready = status?.ready ?? false;
  const passedChecks = status?.checks.filter((c) => c.status === "ok").length ?? 0;
  const totalChecks = status?.checks.length ?? 0;

  // 4 Primary Studio Workflows
  const FEATURED_WORKFLOWS = [
    {
      title: "Inference",
      tag: "Core Studio",
      description: "Transform voice in recorded audio or songs with single and batch file conversion.",
      icon: Sparkles,
      href: "/inference",
      actionText: "Open Inference",
    },
    {
      title: "Realtime",
      tag: "Live Audio",
      description: "Low-latency real-time voice conversion from microphone to virtual audio cables.",
      icon: Radio,
      href: "/realtime",
      actionText: "Launch Realtime",
    },
    {
      title: "Training",
      tag: "Model Lab",
      description: "Slice audio datasets, extract pitch contours, and train your own custom voice models.",
      icon: Cpu,
      href: "/train",
      actionText: "Start Training",
    },
    {
      title: "TTS",
      tag: "Speech Synthesis",
      description: "Generate realistic speech in dozens of languages and instantly clone target voices.",
      icon: Mic,
      href: "/tts",
      actionText: "Synthesize Voice",
    },
  ];

  // Secondary Tools
  const SECONDARY_TOOLS = [
    { label: "Model Library", href: "/models", icon: Database },
    { label: "Voice Blender", href: "/voice-blender", icon: Layers },
    { label: "Download Models", href: "/download", icon: Download },
    { label: "Audio Tools & F0", href: "/extra", icon: SlidersHorizontal },
    { label: "TensorBoard", href: "/tensorboard", icon: Activity },
  ];

  return (
    <div className="h-full flex flex-col gap-6 overflow-y-auto pb-8 pr-1">
      {/* Hero Showcase Card */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.07] via-white/[0.02] to-transparent p-6 sm:p-8">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-xl">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold tracking-wide uppercase bg-white/10 text-neutral-300 border border-white/10">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    ready ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
                  }`}
                />
                {ready ? "Studio Ready" : "Setup Required"}
              </span>
              <span className="text-xs text-neutral-400">Applio v3.6</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white m-0">Applio</h1>
            <p className="text-neutral-300 text-sm sm:text-base leading-relaxed m-0">
              High-performance AI voice cloning, real-time audio morphing, and neural model training right on
              your local machine.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 shrink-0">
            {ready ? (
              <>
                <button
                  type="button"
                  className="cta flex items-center justify-center gap-2"
                  onClick={() => router.push("/inference")}
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Start Converting</span>
                </button>
                <button
                  type="button"
                  className="ghost flex items-center justify-center gap-2"
                  onClick={() => router.push("/realtime")}
                >
                  <Radio className="w-4 h-4" />
                  <span>Live Studio</span>
                </button>
              </>
            ) : (
              <button
                type="button"
                className="cta flex items-center justify-center gap-2"
                onClick={install}
                disabled={busy}
              >
                <Wrench className="w-4 h-4" />
                <span>{busy ? "Setting up engine…" : "Install / Repair"}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Primary Workflows Grid (Clean 4-card layout) */}
      <section className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <h2 className="title text-lg font-bold text-neutral-200 tracking-tight m-0">Primary Workflows</h2>
          <span className="text-xs text-neutral-400">Select a studio tool to get started</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURED_WORKFLOWS.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="group relative flex flex-col justify-between p-5 rounded-2xl border border-white/10 bg-white/[0.03] hover:bg-white/[0.08] hover:border-white/20 transition-all duration-200"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center text-white group-hover:scale-105 transition-transform duration-200">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded bg-white/5 text-neutral-400 border border-white/5">
                      {item.tag}
                    </span>
                  </div>
                  <h3 className="title text-base font-bold text-neutral-100 group-hover:text-white transition-colors mb-1.5">
                    {item.title}
                  </h3>
                  <p className="text-xs text-neutral-400 leading-relaxed m-0">{item.description}</p>
                </div>

                <div className="flex items-center gap-1 text-xs font-semibold text-neutral-300 group-hover:text-white pt-4 mt-2 border-t border-white/5">
                  <span>{item.actionText}</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Secondary Quick Access Bar */}
      <div className="flex items-center gap-2 p-2 rounded-xl border border-white/10 bg-white/[0.02] flex-wrap">
        <span className="text-xs font-medium text-neutral-400 px-2 py-1">Quick Access:</span>
        {SECONDARY_TOOLS.map((tool) => {
          const Icon = tool.icon;
          return (
            <Link
              key={tool.href}
              href={tool.href}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium text-neutral-300 hover:text-white hover:bg-white/10 transition-colors"
            >
              <Icon className="w-3.5 h-3.5 text-neutral-400" />
              <span>{tool.label}</span>
            </Link>
          );
        })}
      </div>

      {/* System Status & Environment Diagnostics */}
      <section className="card !mb-0 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="title text-base font-bold text-neutral-100 m-0">System Diagnostics</h2>
              {status && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-neutral-300 border border-white/5 font-mono">
                  {passedChecks}/{totalChecks} checks passed
                </span>
              )}
            </div>
            <p className="text-xs text-neutral-400 mt-1 mb-0">
              Hardware acceleration, Python dependencies, and pretrained base checkpoints.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="ghost text-xs py-1.5 px-3 flex items-center gap-1.5"
              onClick={() => refresh(true)}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Re-check</span>
            </button>
            <button
              type="button"
              className="ghost text-xs py-1.5 px-3 flex items-center gap-1.5"
              onClick={prerequisites}
            >
              <Download className="w-3.5 h-3.5" />
              <span>Engine models</span>
            </button>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {!status && !error && <p className="muted text-xs py-2">Contacting the Applio engine API…</p>}

        {status && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {status.checks.map((c) => {
              const isOk = c.status === "ok";
              const isWarn = c.status === "warn";
              return (
                <div
                  key={c.id}
                  className="flex items-start gap-2.5 p-3 rounded-xl border border-white/5 bg-black/20"
                >
                  <div className="mt-0.5 shrink-0">
                    {isOk ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : isWarn ? (
                      <AlertCircle className="w-4 h-4 text-amber-400" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-semibold text-neutral-200 truncate">{c.label}</span>
                      <span
                        className={`text-[10px] font-mono px-1.5 py-0.2 rounded uppercase ${
                          isOk
                            ? "text-emerald-400 bg-emerald-400/10"
                            : isWarn
                              ? "text-amber-400 bg-amber-400/10"
                              : "text-red-400 bg-red-400/10"
                        }`}
                      >
                        {c.status}
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-400 mt-0.5 truncate m-0">{c.detail}</p>
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
