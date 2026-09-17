"use client";

import { useCallback, useEffect, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg } from "../../lib/api";

interface AppConfig {
  model_index_filter?: boolean;
  discord_presence?: boolean;
  lang?: { override: boolean; selected_lang: string };
  model_author?: string | null;
  precision?: string;
  rmvpe_high_register?: { enabled: boolean; mode: string; f0_ceil: number };
  version?: string;
  [key: string]: unknown;
}

interface VersionCheck {
  local?: string;
  latest?: string;
  status?: string;
  error?: string;
}

export default function SettingsPage() {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [langs, setLangs] = useState<string[]>([]);
  const [ver, setVer] = useState<VersionCheck | null>(null);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");

  const load = useCallback(async () => {
    try {
      const c = await apiGet<{ config: AppConfig }>("/api/settings");
      setCfg(c.config);
      const l = await apiGet<{ languages: string[] }>("/api/settings/languages");
      setLangs(l.languages);
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function save(patch: unknown) {
    setError("");
    setSaved("");
    try {
      const r = await apiSend<{ config: AppConfig }>("/api/settings", "PUT", patch);
      setCfg(r.config);
      setSaved("Saved ✓");
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function checkVersion() {
    try {
      setVer(await apiGet<VersionCheck>("/api/settings/version-check"));
    } catch (e) {
      setVer({ error: errMsg(e) });
    }
  }

  if (!cfg)
    return (
      <div className="card">
        <p className="muted">Loading settings…</p>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      </div>
    );
  const set = (path: string[], value: unknown) => {
    const next = structuredClone(cfg);
    let o: Record<string, unknown> = next;
    for (let i = 0; i < path.length - 1; i++) o = o[path[i]] as Record<string, unknown>;
    o[path[path.length - 1]] = value;
    setCfg(next);
  };

  return (
    <div>
      <PageHeader
        title="Settings"
        description="Configure application preferences, audio engine settings, precision, and language."
      />
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      {saved && <p style={{ color: "var(--ok)" }}>{saved}</p>}

      <div className="card">
        <h2>General</h2>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.model_index_filter}
            onChange={(e) => set(["model_index_filter"], e.target.checked)}
          />
          <span>Model & index filter box</span>
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.discord_presence}
            onChange={(e) => set(["discord_presence"], e.target.checked)}
          />
          <span>Discord Rich Presence</span>
        </label>
        <div className="grid2">
          <div>
            <label>Language ({langs.length} available)</label>
            <select
              value={cfg.lang?.override ? cfg.lang.selected_lang : ""}
              onChange={(e) => {
                if (!e.target.value)
                  setCfg({
                    ...cfg,
                    lang: { override: false, selected_lang: cfg.lang?.selected_lang || "en_US" },
                  });
                else setCfg({ ...cfg, lang: { override: true, selected_lang: e.target.value } });
              }}
            >
              <option value="">Language automatically detected…</option>
              {langs.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="cta"
            onClick={() =>
              save({
                model_index_filter: cfg.model_index_filter,
                discord_presence: cfg.discord_presence,
                lang: cfg.lang,
              })
            }
          >
            Save General
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Training</h2>
        <div className="grid2">
          <div>
            <label>Model author</label>
            <input
              type="text"
              value={cfg.model_author || ""}
              onChange={(e) => set(["model_author"], e.target.value || null)}
            />
          </div>
          <div>
            <label>Precision</label>
            <select value={cfg.precision} onChange={(e) => set(["precision"], e.target.value)}>
              {["fp32", "fp16", "bf16"].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="cta"
            onClick={() => save({ model_author: cfg.model_author, precision: cfg.precision })}
          >
            Save Training
          </button>
        </div>
      </div>

      <div className="card">
        <h2>RMVPE High Register</h2>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.rmvpe_high_register?.enabled}
            onChange={(e) => set(["rmvpe_high_register", "enabled"], e.target.checked)}
          />
          <span>Enable High Register</span>
        </label>
        <div className="grid2">
          <div>
            <label>Mode</label>
            <select
              value={cfg.rmvpe_high_register?.mode}
              onChange={(e) => set(["rmvpe_high_register", "mode"], e.target.value)}
            >
              <option value="true_pitch">true_pitch</option>
              <option value="fold">fold</option>
            </select>
          </div>
          <div>
            <label>F0 ceiling: {cfg.rmvpe_high_register?.f0_ceil}</label>
            <input
              type="range"
              min={1000}
              max={2000}
              step={10}
              value={cfg.rmvpe_high_register?.f0_ceil || 1250}
              onChange={(e) => set(["rmvpe_high_register", "f0_ceil"], Number(e.target.value))}
            />
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="cta"
            onClick={() => save({ rmvpe_high_register: cfg.rmvpe_high_register })}
          >
            Save RMVPE
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Version</h2>
        <p className="muted">Local: {cfg.version}</p>
        <div className="row">
          <button type="button" className="ghost" onClick={checkVersion}>
            Check for updates
          </button>
          {ver && <span className="muted">{ver.error || `${ver.latest} — ${ver.status}`}</span>}
        </div>
      </div>
    </div>
  );
}
