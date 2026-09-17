"use client";

import { useCallback, useEffect, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import { apiGet, apiSend, errMsg } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

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
  const { t } = useI18n();
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [langs, setLangs] = useState<Array<{ code: string; name: string }>>([]);
  const [themes, setThemes] = useState<Array<{ id: string; name: string; description: string; example: boolean }>>([]);
  const [ver, setVer] = useState<VersionCheck | null>(null);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const [presenceRunning, setPresenceRunning] = useState<boolean | null>(null);
  const [restartMsg, setRestartMsg] = useState("");

  const load = useCallback(async () => {
    try {
      const c = await apiGet<{ config: AppConfig }>("/api/settings");
      setCfg(c.config);
      const l = await apiGet<{ languages: string[]; named?: Array<{ code: string; name: string }> }>(
        "/api/settings/languages",
      );
      const named = l.named || l.languages.map((code) => ({ code, name: code }));
      setLangs(named);
      try {
        const th = await apiGet<{
          themes: Array<{ id: string; name: string; description: string; example: boolean }>;
        }>("/api/settings/themes");
        setThemes(th.themes);
      } catch {
        /* themes unavailable */
      }
      try {
        const p = await apiGet<{ running: boolean }>("/api/settings/presence");
        setPresenceRunning(p.running);
      } catch {
        /* presence unavailable (Discord closed?) */
      }
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
      if (typeof (patch as Record<string, unknown>).lang !== "undefined") {
        window.dispatchEvent(new Event("applio:language-changed"));
      }
      // Discord presence takes effect immediately (Gradio presence.py parity).
      if (typeof (patch as Record<string, unknown>).discord_presence === "boolean") {
        try {
          const p = await apiSend<{ running: boolean }>("/api/settings/presence", "POST", {
            enabled: (patch as Record<string, unknown>).discord_presence,
          });
          setPresenceRunning(p.running);
        } catch (e) {
          setError(errMsg(e));
          return;
        }
      }
      setSaved(t("Saved ✓"));
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

  async function restartApi() {
    setRestartMsg("");
    setError("");
    try {
      const r = await apiSend<{ message: string }>("/api/settings/restart", "POST");
      setRestartMsg(r.message);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  if (!cfg)
    return (
      <div className="card">
        <p className="muted">{t("Loading settings…")}</p>
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
        title={t("Settings")}
        description={t("Configure application preferences, audio engine settings, precision, and language.")}
      />
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      {saved && <p style={{ color: "var(--ok)" }}>{saved}</p>}

      <div className="card">
        <h2>{t("General")}</h2>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.model_index_filter}
            onChange={(e) => set(["model_index_filter"], e.target.checked)}
          />
          <span>{t("Model & index filter box")}</span>
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.discord_presence}
            onChange={(e) => set(["discord_presence"], e.target.checked)}
          />
          <span>{t("Discord Rich Presence")}</span>
          {presenceRunning !== null && (
            <span className="muted"> ({presenceRunning ? t("running") : t("stopped")})</span>
          )}
        </label>
        <div className="grid2">
          <div>
            <label>
              {t("Language")} ({langs.length} {t("available")})
            </label>
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
              <option value="">{t("Language automatically detected…")}</option>
              {langs.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name} ({l.code})
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
            {t("Save General")}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>{t("Appearance")}</h2>
        <p className="muted">
          {t("Pick a theme from assets/themes/. Copy custom.example.json to create your own — see docs/themes.md.")}
        </p>
        <div className="grid2">
          <div>
            <label>{t("Theme")}</label>
            <select
              value={(cfg.theme as { file?: string } | undefined)?.file || ""}
              onChange={(e) => set(["theme", "file"], e.target.value)}
            >
              <option value="">{t("Default")}</option>
              {themes.map((th) => (
                <option key={th.id} value={th.id}>
                  {th.name}
                  {th.description ? ` — ${th.description.slice(0, 60)}` : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="ghost"
            onClick={() =>
              save({ theme: { file: (cfg.theme as { file?: string } | undefined)?.file || "" } }).then(
                () => window.dispatchEvent(new Event("applio:theme-changed")),
              )
            }
          >
            {t("Save Appearance")}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>{t("Training")}</h2>
        <div className="grid2">
          <div>
            <label>{t("Model Author Name")}</label>
            <input
              type="text"
              value={cfg.model_author || ""}
              onChange={(e) => set(["model_author"], e.target.value || null)}
            />
          </div>
          <div>
            <label>{t("Precision")}</label>
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
            {t("Save Training")}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>{t("RMVPE High Register")}</h2>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!cfg.rmvpe_high_register?.enabled}
            onChange={(e) => set(["rmvpe_high_register", "enabled"], e.target.checked)}
          />
          <span>{t("Enable High Register")}</span>
        </label>
        <div className="grid2">
          <div>
            <label>{t("Mode")}</label>
            <select
              value={cfg.rmvpe_high_register?.mode}
              onChange={(e) => set(["rmvpe_high_register", "mode"], e.target.value)}
            >
              <option value="true_pitch">true_pitch</option>
              <option value="fold">fold</option>
            </select>
          </div>
          <div>
            <label>
              {t("F0 ceiling")}: {cfg.rmvpe_high_register?.f0_ceil}
            </label>
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
            {t("Save RMVPE")}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>{t("Version Checker")}</h2>
        <p className="muted">
          {t("Local")}: {cfg.version}
        </p>
        <div className="row">
          <button type="button" className="ghost" onClick={checkVersion}>
            {t("Check for updates")}
          </button>
          {ver && <span className="muted">{ver.error || `${ver.latest} — ${ver.status}`}</span>}
        </div>
      </div>

      <div className="card">
        <h2>{t("Restart")}</h2>
        <p className="muted">{t("Restarts the API process (the dev watcher respawns it automatically).")}</p>
        <div className="row">
          <button type="button" className="ghost" onClick={restartApi}>
            {t("Restart API")}
          </button>
          {restartMsg && <span className="muted">{restartMsg}</span>}
        </div>
      </div>
    </div>
  );
}
