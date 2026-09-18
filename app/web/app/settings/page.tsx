"use client";

import { useCallback, useEffect, useState } from "react";
import { Sliders, Palette, Cpu, Activity, RefreshCw, Power, Save } from "lucide-react";
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
      setSaved(t("Settings saved successfully."));
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
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="card">
          <p className="text-neutral-400">{t("Loading settings…")}</p>
          {error && <p className="text-neutral-300">{error}</p>}
        </div>
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
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Settings")}
        description={t("Configure application preferences, audio engine settings, precision, and language.")}
      />

      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-3.5 rounded-xl border border-white/10 text-neutral-200 bg-white/5 text-sm"
        >
          {error}
        </div>
      )}
      {saved && (
        <div
          role="status"
          aria-live="polite"
          className="p-3.5 rounded-xl border border-white/10 text-white bg-white/10 text-sm"
        >
          {saved}
        </div>
      )}

      {/* 1. General Preferences */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("General Preferences")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Configure search filter visibility, Discord Rich Presence, and UI locale.")}
          </p>
        </div>

        <div className="space-y-3.5">
          <label
            htmlFor="settings-filter-checkbox"
            className="flex items-center gap-2.5 cursor-pointer text-xs font-medium text-neutral-200"
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
            className="flex items-center gap-2.5 cursor-pointer text-xs font-medium text-neutral-200"
          >
            <input
              id="settings-discord-checkbox"
              type="checkbox"
              checked={!!cfg.discord_presence}
              onChange={(e) => set(["discord_presence"], e.target.checked)}
            />
            <span>{t("Discord Rich Presence")}</span>
            {presenceRunning !== null && (
              <span className="text-xs text-neutral-400" role="status">
                ({presenceRunning ? t("running") : t("stopped")})
              </span>
            )}
          </label>

          <div className="max-w-md pt-1">
            <label htmlFor="settings-lang" className="block text-xs font-medium text-neutral-300 mb-1.5">
              {t("Interface Language")} ({langs.length} {t("available")})
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

        <div className="pt-3.5 border-t border-white/5 flex items-center justify-end">
          <button
            type="button"
            className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
            onClick={() =>
              save({
                model_index_filter: cfg.model_index_filter,
                discord_presence: cfg.discord_presence,
                lang: cfg.lang,
              })
            }
          >
            <Save size={14} className="shrink-0" />
            <span>{t("Save General")}</span>
          </button>
        </div>
      </div>

      {/* 2. Appearance & Themes */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Palette size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Appearance & Themes")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Customize visual theme styling from assets/themes/ or load custom theme palettes.")}
          </p>
        </div>

        <div className="max-w-md">
          <label htmlFor="settings-theme" className="block text-xs font-medium text-neutral-300 mb-1.5">
            {t("Theme")}
          </label>
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

        <div className="pt-3.5 border-t border-white/5 flex items-center justify-end">
          <button
            type="button"
            className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
            onClick={() =>
              save({ theme: { file: (cfg.theme as { file?: string } | undefined)?.file || "" } }).then(() =>
                window.dispatchEvent(new Event("applio:theme-changed")),
              )
            }
          >
            <Save size={14} className="shrink-0" />
            <span>{t("Save Appearance")}</span>
          </button>
        </div>
      </div>

      {/* 3. Training Engine */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Training Engine")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Set default model author metadata and floating-point computation precision for training.")}
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
          <div>
            <label htmlFor="settings-model-author" className="block text-xs font-medium text-neutral-300 mb-1.5">
              {t("Model Author Name")}
            </label>
            <input
              id="settings-model-author"
              type="text"
              value={cfg.model_author || ""}
              onChange={(e) => set(["model_author"], e.target.value || null)}
              placeholder="Applio"
            />
          </div>
          <div>
            <label htmlFor="settings-precision" className="block text-xs font-medium text-neutral-300 mb-1.5">
              {t("Precision")}
            </label>
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

        <div className="pt-3.5 border-t border-white/5 flex items-center justify-end">
          <button
            type="button"
            className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
            onClick={() => save({ model_author: cfg.model_author, precision: cfg.precision })}
          >
            <Save size={14} className="shrink-0" />
            <span>{t("Save Training")}</span>
          </button>
        </div>
      </div>

      {/* 4. RMVPE High Register */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("RMVPE High Register")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Adjust pitch detection algorithm behavior and frequency ceiling for higher vocal registers.")}
          </p>
        </div>

        <div className="space-y-4">
          <label
            htmlFor="settings-rmvpe-enabled"
            className="flex items-center gap-2.5 cursor-pointer text-xs font-medium text-neutral-200"
          >
            <input
              id="settings-rmvpe-enabled"
              type="checkbox"
              checked={!!cfg.rmvpe_high_register?.enabled}
              onChange={(e) => set(["rmvpe_high_register", "enabled"], e.target.checked)}
            />
            <span>{t("Enable High Register")}</span>
          </label>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
            <div>
              <label htmlFor="settings-rmvpe-mode" className="block text-xs font-medium text-neutral-300 mb-1.5">
                {t("Mode")}
              </label>
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
        </div>

        <div className="pt-3.5 border-t border-white/5 flex items-center justify-end">
          <button
            type="button"
            className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
            onClick={() => save({ rmvpe_high_register: cfg.rmvpe_high_register })}
          >
            <Save size={14} className="shrink-0" />
            <span>{t("Save RMVPE")}</span>
          </button>
        </div>
      </div>

      {/* 5. Version & Updates */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RefreshCw size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Version & Updates")}</h2>
            </div>
            <span className="text-xs text-neutral-400 tabular-nums px-2 py-0.5 rounded-full bg-white/5 border border-white/10">
              v{cfg.version}
            </span>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Check for official Applio updates and apply package releases.")}
          </p>
        </div>

        {/* Status display section */}
        <div className="space-y-3">
          {updaterState && updaterState.status === "downloading" && (
            <div className="max-w-md space-y-1.5">
              <div className="flex justify-between text-xs text-neutral-400 tabular-nums">
                <span>{t("Downloading update…")}</span>
                <span>{updaterState.percent ?? 0}%</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all duration-300"
                  style={{ width: `${updaterState.percent ?? 0}%` }}
                />
              </div>
            </div>
          )}

          {updaterState?.status === "available" && (
            <p className="text-xs text-neutral-300 m-0" role="status" aria-live="polite">
              {t("Update available: v")}{updaterState.version}{t(". Ready to download.")}
            </p>
          )}

          {updaterState?.status === "not-available" && (
            <p className="text-xs text-neutral-400 m-0" role="status" aria-live="polite">
              {t("Applio is up to date (v")}{updaterState.version || cfg.version}{")"}
            </p>
          )}

          {updaterState?.status === "error" && (
            <p className="text-xs text-neutral-300 m-0" role="status" aria-live="polite">
              {updaterState.message}
            </p>
          )}

          {ver && (!updaterState || updaterState.status === "dev-mode") && (
            <p className="text-xs text-neutral-400 m-0" role="status" aria-live="polite">
              {ver.error || `${ver.latest} — ${ver.status}`}
            </p>
          )}
        </div>

        {/* Action row with clean breathing room */}
        <div className="pt-3.5 border-t border-white/5 flex items-center justify-between gap-4 flex-wrap">
          <div className="text-xs text-neutral-400">
            {updaterState?.status === "downloaded" && (
              <span>{t("Update downloaded (v")}{updaterState.version}{")"}</span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {updaterState?.status === "downloaded" && (
              <button
                type="button"
                className="cta h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2"
                onClick={() => {
                  (
                    window as unknown as { applio?: { updater?: { quitAndInstall: () => void } } }
                  ).applio?.updater?.quitAndInstall();
                }}
              >
                <RefreshCw size={14} className="shrink-0" />
                <span>{t("Restart and Update")}</span>
              </button>
            )}

            <button
              type="button"
              className="ghost h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2 text-neutral-200 hover:text-white"
              onClick={checkVersion}
              disabled={updaterState?.status === "checking" || updaterState?.status === "downloading"}
            >
              <RefreshCw size={14} className={`text-white shrink-0 ${updaterState?.status === "checking" ? "animate-spin" : ""}`} />
              <span>{updaterState?.status === "checking" ? t("Checking…") : t("Check for Updates")}</span>
            </button>
          </div>
        </div>
      </div>

      {/* 6. Restart API */}
      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Power size={18} className="text-white shrink-0" />
              <h2 className="text-base font-bold text-white m-0">{t("Restart API")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Restarts the backend API service to apply system changes.")}
          </p>
        </div>

        <div className="pt-1 flex items-center justify-between gap-4">
          <div>
            {restartMsg && (
              <span className="text-xs text-neutral-300" role="status" aria-live="polite">
                {restartMsg}
              </span>
            )}
          </div>

          <button
            type="button"
            className="ghost h-9 px-4 rounded-xl text-xs font-medium flex items-center gap-2 text-neutral-200 hover:text-white"
            onClick={restartApi}
          >
            <Power size={14} className="text-white shrink-0" />
            <span>{t("Restart API")}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
