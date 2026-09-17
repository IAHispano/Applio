"use client";

import { useCallback, useEffect, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
import SliderField from "../../components/ui/SliderField";
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

interface DesktopUpdaterState {
  status:
    | "idle"
    | "checking"
    | "available"
    | "not-available"
    | "downloading"
    | "downloaded"
    | "error"
    | "dev-mode";
  version?: string;
  percent?: number;
  message?: string;
  releaseNotes?: string;
}

export default function SettingsPage() {
  const { t } = useI18n();
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [langs, setLangs] = useState<Array<{ code: string; name: string }>>([]);
  const [themes, setThemes] = useState<
    Array<{ id: string; name: string; description: string; example: boolean }>
  >([]);
  const [ver, setVer] = useState<VersionCheck | null>(null);
  const [updaterState, setUpdaterState] = useState<DesktopUpdaterState | null>(null);
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
    if (typeof window !== "undefined") {
      const bridge = (
        window as unknown as {
          applio?: {
            updater?: {
              getStatus: () => Promise<DesktopUpdaterState>;
              onStatusChange: (cb: (s: DesktopUpdaterState) => void) => () => void;
            };
          };
        }
      ).applio;
      if (bridge?.updater) {
        bridge.updater
          .getStatus()
          .then((st) => {
            if (st && st.status !== "idle") setUpdaterState(st);
          })
          .catch(() => {});
        const unsub = bridge.updater.onStatusChange((st) => {
          setUpdaterState(st);
        });
        return unsub;
      }
    }
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
    setError("");
    const bridge =
      typeof window !== "undefined"
        ? (
            window as unknown as {
              applio?: {
                updater?: { check: () => Promise<{ status?: string }>; quitAndInstall: () => void };
              };
            }
          ).applio
        : undefined;

    if (bridge?.updater) {
      setVer(null);
      setUpdaterState({ status: "checking" });
      try {
        const res = await bridge.updater.check();
        if (res?.status === "dev-mode") {
          const v = await apiGet<VersionCheck>("/api/settings/version-check");
          setVer(v);
        }
      } catch (e) {
        setUpdaterState({ status: "error", message: errMsg(e) });
      }
      return;
    }

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
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mb-4 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
        >
          {error}
        </div>
      )}
      {saved && (
        <div
          role="status"
          aria-live="polite"
          className="mb-4 p-3 rounded-lg border border-[var(--ok)] text-[var(--ok)] bg-[color-mix(in_srgb,var(--ok)_10%,transparent)]"
        >
          {saved}
        </div>
      )}

      <div className="card">
        <h2>{t("General")}</h2>
        <label
          htmlFor="settings-filter-checkbox"
          className="checkbox-label flex items-center gap-2 cursor-pointer"
        >
          <input
            id="settings-filter-checkbox"
            type="checkbox"
            checked={!!cfg.model_index_filter}
            onChange={(e) => set(["model_index_filter"], e.target.checked)}
          />
          <span>{t("Model & index filter box")}</span>
        </label>
        <label
          htmlFor="settings-discord-checkbox"
          className="checkbox-label flex items-center gap-2 cursor-pointer mt-2"
        >
          <input
            id="settings-discord-checkbox"
            type="checkbox"
            checked={!!cfg.discord_presence}
            onChange={(e) => set(["discord_presence"], e.target.checked)}
          />
          <span>{t("Discord Rich Presence")}</span>
          {presenceRunning !== null && (
            <span className="muted" role="status">
              {" "}
              ({presenceRunning ? t("running") : t("stopped")})
            </span>
          )}
        </label>
        <div className="grid2 mt-3">
          <div>
            <label htmlFor="settings-lang">
              {t("Language")} ({langs.length} {t("available")})
            </label>
            <select
              id="settings-lang"
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
          {t(
            "Pick a theme from assets/themes/. Copy custom.example.json to create your own — see docs/themes.md.",
          )}
        </p>
        <div className="grid2">
          <div>
            <label htmlFor="settings-theme">{t("Theme")}</label>
            <select
              id="settings-theme"
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
              save({ theme: { file: (cfg.theme as { file?: string } | undefined)?.file || "" } }).then(() =>
                window.dispatchEvent(new Event("applio:theme-changed")),
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
            <label htmlFor="settings-model-author">{t("Model Author Name")}</label>
            <input
              id="settings-model-author"
              type="text"
              value={cfg.model_author || ""}
              onChange={(e) => set(["model_author"], e.target.value || null)}
            />
          </div>
          <div>
            <label htmlFor="settings-precision">{t("Precision")}</label>
            <select
              id="settings-precision"
              value={cfg.precision}
              onChange={(e) => set(["precision"], e.target.value)}
            >
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
        <label
          htmlFor="settings-rmvpe-enabled"
          className="checkbox-label flex items-center gap-2 cursor-pointer"
        >
          <input
            id="settings-rmvpe-enabled"
            type="checkbox"
            checked={!!cfg.rmvpe_high_register?.enabled}
            onChange={(e) => set(["rmvpe_high_register", "enabled"], e.target.checked)}
          />
          <span>{t("Enable High Register")}</span>
        </label>
        <div className="grid2 mt-3">
          <div>
            <label htmlFor="settings-rmvpe-mode">{t("Mode")}</label>
            <select
              id="settings-rmvpe-mode"
              value={cfg.rmvpe_high_register?.mode}
              onChange={(e) => set(["rmvpe_high_register", "mode"], e.target.value)}
            >
              <option value="true_pitch">true_pitch</option>
              <option value="fold">fold</option>
            </select>
          </div>
          <div>
            <SliderField
              id="settings-rmvpe-ceil"
              label={t("F0 ceiling")}
              value={cfg.rmvpe_high_register?.f0_ceil || 1250}
              min={1000}
              max={2000}
              step={10}
              unit="Hz"
              onChange={(v) => set(["rmvpe_high_register", "f0_ceil"], v)}
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
        <div className="row" style={{ alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
          <button
            type="button"
            className="ghost"
            onClick={checkVersion}
            disabled={updaterState?.status === "checking" || updaterState?.status === "downloading"}
          >
            {updaterState?.status === "checking" ? t("Checking…") : t("Check for updates")}
          </button>
          {updaterState && updaterState.status !== "idle" && updaterState.status !== "dev-mode" && (
            <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 1, minWidth: "240px" }}>
              {updaterState.status === "downloading" && (
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "12px",
                      color: "var(--text-muted)",
                      marginBottom: "4px",
                    }}
                  >
                    <span>{t("Downloading update…")}</span>
                    <span>{updaterState.percent ?? 0}%</span>
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: "6px",
                      background: "rgba(255,255,255,0.1)",
                      borderRadius: "999px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${updaterState.percent ?? 0}%`,
                        background: "#ffffff",
                        borderRadius: "999px",
                        transition: "width 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                      }}
                    />
                  </div>
                </div>
              )}
              {updaterState.status === "downloaded" && (
                <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                  <span className="muted" role="status" aria-live="polite">
                    {t("Update downloaded (v")}
                    {updaterState.version}
                    {")"}
                  </span>
                  <button
                    type="button"
                    className="cta"
                    onClick={() => {
                      (
                        window as unknown as { applio?: { updater?: { quitAndInstall: () => void } } }
                      ).applio?.updater?.quitAndInstall();
                    }}
                  >
                    {t("Restart and Update")}
                  </button>
                </div>
              )}
              {updaterState.status === "available" && (
                <span className="muted" role="status" aria-live="polite">
                  {t("Update available (v")}
                  {updaterState.version}
                  {t("). Downloading…")}
                </span>
              )}
              {updaterState.status === "not-available" && (
                <span className="muted" role="status" aria-live="polite">
                  {t("Applio is up to date (v")}
                  {updaterState.version || cfg.version}
                  {")"}
                </span>
              )}
              {updaterState.status === "error" && (
                <span className="muted" style={{ color: "var(--err)" }} role="status" aria-live="polite">
                  {updaterState.message}
                </span>
              )}
            </div>
          )}
          {ver && (!updaterState || updaterState.status === "dev-mode") && (
            <span className="muted" role="status" aria-live="polite">
              {ver.error || `${ver.latest} — ${ver.status}`}
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <h2>{t("Restart")}</h2>
        <p className="muted">{t("Restarts the API process (the dev watcher respawns it automatically).")}</p>
        <div className="row">
          <button type="button" className="ghost" onClick={restartApi}>
            {t("Restart API")}
          </button>
          {restartMsg && (
            <span className="muted" role="status" aria-live="polite">
              {restartMsg}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
