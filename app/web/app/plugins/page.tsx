"use client";

import { useCallback, useEffect, useState } from "react";
import JobPanel from "../../components/JobPanel";
import { apiGet, apiSend, errMsg, postForm } from "../../lib/api";

interface Plugin {
  name: string;
  enabled: boolean;
  hasEntrypoint: boolean;
}

export default function PluginsPage() {
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
      setMsg("Installing… restart the app when done.");
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
      <h2 className="title mb-4">Plugins</h2>
      <div className="mb-4">
        {msg && <p className="muted">{msg}</p>}
        {plugins.length === 0 && <p className="muted">No plugins installed yet.</p>}
        {plugins.map((p) => (
          <div className="row" key={p.name} style={{ marginBottom: 8 }}>
            <strong>{p.name}</strong>
            <span className={`badge ${p.enabled ? "done" : "queued"}`}>
              {p.enabled ? "enabled" : "disabled"}
            </span>
            {!p.hasEntrypoint && <span className="muted">no plugin.py entrypoint</span>}
            <button type="button" className="ghost" onClick={() => toggle(p)}>
              {p.enabled ? "Disable" : "Enable"}
            </button>
          </div>
        ))}
      </div>
      <div className="card">
        <h2>Install Plugin (.zip)</h2>
        <div className="row">
          <input type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button type="button" className="ghost" onClick={install}>
            Install
          </button>
        </div>
      </div>
      <JobPanel jobId={jobId} />
    </div>
  );
}
