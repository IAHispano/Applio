"use client";

import { ExternalLink, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { apiGet, errMsg } from "../lib/api";
import { useI18n } from "../lib/i18n";

export default function TensorboardPanel() {
  const { t } = useI18n();
  const [status, setStatus] = useState<{
    running: boolean;
    starting?: boolean;
    url: string;
    startedAt: string | null;
  } | null>(null);
  const [error, setError] = useState("");
  const [iframeKey, setIframeKey] = useState(0);

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
    const t = setInterval(refresh, status?.running ? 5000 : 2000);
    return () => clearInterval(t);
  }, [refresh, status?.running]);

  const tbHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
  const tbPort = status?.url?.match(/:(\d+)$/)?.[1] || "6007";
  const iframeUrl = `http://${tbHost}:${tbPort}/`;
  const isRunning = status?.running ?? false;

  return (
    <div className="space-y-3">
      {error && (
        <p role="alert" aria-live="assertive" className="text-xs text-red-400">
          {error}
        </p>
      )}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isRunning ? (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-white/10 text-white border border-white/10">
              <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              <span>{`:${tbPort}`}</span>
            </span>
          ) : (
            <span className="text-xs text-neutral-400">{t("Starting background service…")}</span>
          )}
        </div>
        {isRunning && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="ghost text-xs py-1 px-2 flex items-center gap-1"
              onClick={() => setIframeKey((k) => k + 1)}
              title={t("Reload")}
            >
              <RefreshCw className="w-3 h-3" />
              <span>{t("Reload")}</span>
            </button>
            <a
              href={iframeUrl}
              target="_blank"
              rel="noreferrer"
              className="ghost text-xs py-1 px-2 flex items-center gap-1"
              title={t("Open in browser tab")}
            >
              <ExternalLink className="w-3 h-3" />
              <span>{t("Open Tab")}</span>
            </a>
          </div>
        )}
      </div>

      <div className="w-full h-[600px] relative rounded-xl border border-white/10 overflow-hidden bg-black/40">
        {isRunning ? (
          <iframe
            key={iframeKey}
            src={iframeUrl}
            title={t("TensorBoard")}
            className="w-full h-full border-0 absolute inset-0"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-neutral-400 p-6">
            <div className="w-48 h-1.5 rounded-full bg-white/10 overflow-hidden relative">
              <div className="h-full bg-white rounded-full animate-pulse w-2/3" />
            </div>
            <span className="text-xs text-neutral-400">{t("Loading TensorBoard…")}</span>
          </div>
        )}
      </div>
    </div>
  );
}
