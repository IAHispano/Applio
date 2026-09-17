"use client";

import { useCallback, useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg, postForm } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

interface Plugin {
  name: string;
  enabled: boolean;
  hasEntrypoint: boolean;
}

export default function PluginsPage() {
  const { t } = useI18n();
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [msg, setMsg] = useState("");

  const refresh = useCallback(async () => {
    try {
      const p = await apiGet<{ plugins: Plugin[] }>("/api/plugins");
      setPlugins(p.plugins);
    } catch (e) {
      setMsg(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function install() {
    setMsg("");
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const { jobId: id } = await postForm<{ jobId: string }>("/api/plugins/install", fd);
      setJobId(id);
      setMsg(t("Installing… restart the app when done."));
    } catch (e) {
      setMsg(errMsg(e));
    }
  }

  async function toggle(p: Plugin) {
    try {
      await apiSend("/api/plugins/toggle", "POST", { name: p.name, enabled: !p.enabled });
      refresh();
    } catch (e) {
      setMsg(errMsg(e));
    }
  }

  return (
    <div>
      <PageHeader
        title={t("Plugins")}
        description={t("Manage installed plugins and extend Applio with custom functionality.")}
      />
      <div className="mb-4">
        {msg && (
          <p role="status" aria-live="polite" className="muted">
            {msg}
          </p>
        )}
        {plugins.length === 0 && <p className="muted">{t("No plugins installed yet.")}</p>}
        {plugins.map((p) => (
          <div className="row" key={p.name} style={{ marginBottom: 8 }}>
            <strong>{p.name}</strong>
            <span
              className={`badge ${p.enabled ? "done" : "queued"}`}
              role="status"
              aria-label={`Plugin ${p.name} status: ${p.enabled ? t("enabled") : t("disabled")}`}
            >
              {p.enabled ? t("enabled") : t("disabled")}
            </span>
            {!p.hasEntrypoint && <span className="muted">{t("no plugin.py entrypoint")}</span>}
            <button
              type="button"
              className="ghost"
              onClick={() => toggle(p)}
              aria-label={`${p.enabled ? t("Disable") : t("Enable")} ${p.name}`}
            >
              {p.enabled ? t("Disable") : t("Enable")}
            </button>
          </div>
        ))}
      </div>
      <div className="card">
        <h2>{t("Install Plugin (.zip)")}</h2>
        <div className="row">
          <input
            id="plugin-file-input"
            type="file"
            accept=".zip"
            aria-label={t("Select plugin zip file")}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button
            type="button"
            className="cta"
            onClick={install}
            disabled={!file}
            aria-label={t("Install selected plugin")}
          >
            {t("Install")}
          </button>
        </div>
      </div>
      <JobPanel jobId={jobId} />
    </div>
  );
}
