"use client";

import { useCallback, useEffect, useState } from "react";
import { Blocks, PackagePlus, Upload, Check, Power } from "lucide-react";
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
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Plugins")}
        description={t("Manage installed plugins and extend Applio with custom functionality.")}
      />

      {msg && (
        <div
          role="status"
          aria-live="polite"
          className="p-3.5 rounded-xl border border-white/10 text-neutral-300 bg-white/5 text-sm"
        >
          {msg}
        </div>
      )}

      {/* Installed Plugins Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Blocks size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Installed Plugins")}</h2>
            </div>
            <span className="text-xs text-neutral-400">
              {plugins.length} {t("installed")}
            </span>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("View, enable, or disable extensions installed in your Applio environment.")}
          </p>
        </div>

        {plugins.length === 0 ? (
          <p className="text-xs text-neutral-400 m-0 py-2">{t("No plugins installed yet.")}</p>
        ) : (
          <div className="space-y-2">
            {plugins.map((p) => (
              <div
                key={p.name}
                className="flex items-center justify-between p-3 rounded-xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.05] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="font-medium text-sm text-neutral-200">{p.name}</div>
                  <span
                    className={`badge ${p.enabled ? "done" : "queued"}`}
                    role="status"
                    aria-label={`Plugin ${p.name} status: ${p.enabled ? t("enabled") : t("disabled")}`}
                  >
                    {p.enabled ? t("enabled") : t("disabled")}
                  </span>
                  {!p.hasEntrypoint && (
                    <span className="text-xs text-neutral-400">{t("Missing plugin entry")}</span>
                  )}
                </div>
                <button
                  type="button"
                  className="ghost flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-neutral-300 hover:text-white"
                  onClick={() => toggle(p)}
                  aria-label={`${p.enabled ? t("Disable") : t("Enable")} ${p.name}`}
                >
                  <Power size={13} className="text-white" />
                  <span>{p.enabled ? t("Disable") : t("Enable")}</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Install Plugin Card */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <PackagePlus size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Install Plugin")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Upload a ZIP archive containing a compatible Applio plugin.")}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 max-w-xl">
          <input
            id="plugin-file-input"
            type="file"
            accept=".zip"
            aria-label={t("Select plugin zip file")}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="flex-1"
          />
          <button
            type="button"
            className="cta h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl shrink-0"
            onClick={install}
            disabled={!file}
            aria-label={t("Install selected plugin")}
          >
            <Upload size={16} className="shrink-0" />
            <span>{t("Install Plugin")}</span>
          </button>
        </div>
      </div>

      <JobPanel jobId={jobId} />
    </div>
  );
}
