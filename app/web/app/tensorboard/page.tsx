"use client";

import { ExternalLink, RefreshCw, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

export default function TensorboardPage() {
  const { t } = useI18n();
  const [status, setStatus] = useState<{
    running: boolean;
    starting?: boolean;
    url: string;
    startedAt: string | null;
  } | null>(null);
  const [error, setError] = useState("");
  const [restarting, setRestarting] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const refresh = useCallback(async () => {
    try {
      // Status polls must bypass the apiGet cache or engine state freezes.
      const res = await apiGet<{
        running: boolean;
        starting?: boolean;
        url: string;
        startedAt: string | null;
      }>("/api/tensorboard/status", { ttlMs: 0 });
      setStatus(res);
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // Poll more frequently if starting up, then ease into 5s interval
    const t = setInterval(refresh, status?.running ? 5000 : 2000);
    return () => clearInterval(t);
  }, [refresh, status?.running]);

  async function handleRestart() {
    setRestarting(true);
    setError("");
    try {
      await apiSend("/api/tensorboard/stop", "POST");
      await apiSend("/api/tensorboard/start", "POST");
      await refresh();
      setIframeKey((k) => k + 1);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setRestarting(false);
    }
  }

  function handleReloadIframe() {
    setIframeKey((k) => k + 1);
  }

  // The iframe points at the API host directly; on Colab/Kaggle expose TB_PORT like :3000.
  const tbHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
  const tbPort = status?.url?.match(/:(\d+)$/)?.[1] || "6007";
  const iframeUrl = `http://${tbHost}:${tbPort}/`;

  const isRunning = status?.running ?? false;

  return (
    <div className="max-w-7xl mx-auto w-full h-full flex flex-col min-h-0 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 shrink-0">
        <PageHeader
          title={t("TensorBoard")}
          description={t(
            "Monitor loss curves, spectrograms, and training metrics live during model training.",
          )}
        />
        <div className="flex items-center gap-2 self-start sm:self-auto">
          {isRunning && (
            <>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-white/10 text-white border border-white/10">
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                <span>{`:${tbPort}`}</span>
              </span>
              <button
                type="button"
                className="ghost text-xs py-1.5 px-2.5 flex items-center gap-1.5"
                onClick={handleReloadIframe}
                title={t("Reload TensorBoard view")}
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span className="hidden md:inline">{t("Reload")}</span>
              </button>
              <a
                href={iframeUrl}
                target="_blank"
                rel="noreferrer"
                className="ghost text-xs py-1.5 px-2.5 flex items-center gap-1.5"
                title={t("Open in browser tab")}
              >
                <ExternalLink className="w-3.5 h-3.5" />
                <span className="hidden md:inline">{t("Open Tab")}</span>
              </a>
            </>
          )}
          <button
            type="button"
            className="ghost text-xs py-1.5 px-2.5 flex items-center gap-1.5 text-neutral-400 hover:text-white"
            onClick={handleRestart}
            disabled={restarting}
            title={t("Restart TensorBoard backend service")}
          >
            <RotateCcw className={`w-3.5 h-3.5 ${restarting ? "animate-spin" : ""}`} />
            <span className="hidden md:inline">{restarting ? t("Restarting…") : t("Restart")}</span>
          </button>
        </div>
      </div>

      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3 rounded-lg border border-red-500/20 text-red-400 bg-red-500/10 text-xs shrink-0"
        >
          {error}
        </div>
      )}

      <div className="flex-1 min-h-[500px] relative rounded-xl border border-white/10 overflow-hidden bg-black/40">
        {isRunning ? (
          <iframe
            key={iframeKey}
            ref={iframeRef}
            src={iframeUrl}
            title={t("TensorBoard")}
            className="w-full h-full border-0 absolute inset-0"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-4 text-neutral-400 p-8">
            <div className="w-56 h-1.5 rounded-full bg-white/10 overflow-hidden relative">
              <div className="h-full bg-white rounded-full animate-pulse w-3/4" />
            </div>
            <div className="text-center space-y-1">
              <p className="text-sm font-medium text-neutral-200">
                {restarting ? t("Restarting TensorBoard…") : t("Starting TensorBoard…")}
              </p>
              <p className="text-xs text-neutral-500">
                {t("The service is initializing in the backend and will display automatically.")}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
