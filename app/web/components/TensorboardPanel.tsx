"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiSend, errMsg } from "../lib/api";

export default function TensorboardPanel() {
  const [status, setStatus] = useState<{ running: boolean; url: string; startedAt: string | null } | null>(
    null,
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await apiGet("/api/tensorboard/status"));
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

  const tbHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
  const tbPort = status?.url?.match(/:(\d+)$/)?.[1] || "6007";
  const iframeUrl = `http://${tbHost}:${tbPort}/`;

  return (
    <div>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      <div className="row">
        <button type="button" onClick={start} disabled={busy || status?.running}>
          {busy ? "Starting…" : status?.running ? "Running ✓" : "Launch TensorBoard"}
        </button>
        {status?.running && (
          <button
            type="button"
            className="ghost"
            onClick={() => apiSend("/api/tensorboard/stop", "POST").then(refresh)}
          >
            Stop
          </button>
        )}
        <span className="muted">
          {status ? (status.running ? `live at ${iframeUrl}` : "stopped") : "checking…"}
        </span>
      </div>
      {status?.running && (
        <iframe
          src={iframeUrl}
          title="TensorBoard"
          width="100%"
          height={600}
          style={{ border: "1px solid var(--border)", borderRadius: 8, marginTop: 12 }}
        />
      )}
    </div>
  );
}
