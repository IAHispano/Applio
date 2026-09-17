"use client";

import { useCallback, useEffect, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

export default function TensorboardPage() {
  const { t } = useI18n();
  const [status, setStatus] = useState<{ running: boolean; url: string; startedAt: string | null } | null>(
    null,
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      // Status polls must bypass the apiGet cache or engine state freezes.
      setStatus(await apiGet("/api/tensorboard/status", { ttlMs: 0 }));
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  async function start() {
    setError("");
    setBusy(true);
    try {
      const r = await apiSend<{ url: string }>("/api/tensorboard/start", "POST");
      setStatus({ running: true, url: r.url, startedAt: new Date().toISOString() });
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  // The iframe points at the API host directly; on Colab/Kaggle expose TB_PORT like :3000.
  const tbHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
  const tbPort = status?.url?.match(/:(\d+)$/)?.[1] || "6007";
  const iframeUrl = `http://${tbHost}:${tbPort}/`;

  return (
    <div>
      <PageHeader
        title={t("TensorBoard")}
        description={t("Monitor loss curves, spectrograms, and training metrics live during model training.")}
      />
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <div className="row mb-4">
        <button type="button" onClick={start} disabled={busy || status?.running}>
          {busy ? t("Starting…") : status?.running ? t("Running ✓") : t("Launch TensorBoard")}
        </button>
        {status?.running && (
          <button
            type="button"
            className="ghost"
            onClick={() => apiSend("/api/tensorboard/stop", "POST").then(refresh)}
          >
            {t("Stop")}
          </button>
        )}
        <span className="muted">
          {status ? (status.running ? `live at ${iframeUrl}` : t("stopped")) : t("checking…")}
        </span>
      </div>
      {status?.running && (
        <iframe
          src={iframeUrl}
          title={t("TensorBoard")}
          width="100%"
          height={800}
          style={{ border: "1px solid var(--border)", borderRadius: 8, marginTop: 12 }}
        />
      )}
    </div>
  );
}
