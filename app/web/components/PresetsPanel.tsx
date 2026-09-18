"use client";

import { Bookmark, Download, RefreshCw, Save } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { apiGet, apiSend, errMsg } from "../lib/api";
import { useI18n } from "../lib/i18n";
import { toast } from "../lib/toast";

interface Preset {
  name: string;
  values: { pitch: number; index_rate: number; rms_mix_rate: number; protect: number } | null;
}

export default function PresetsPanel() {
  const { t } = useI18n();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [formant, setFormant] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const p = await apiGet<{ presets: Preset[] }>("/api/presets");
      setPresets(p.presets);
      const f = await apiGet<{ presets: string[] }>("/api/presets/formant");
      setFormant(f.presets);
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function apply(p: Preset, target: "single" | "batch" = "single") {
    if (!p.values) return;
    window.dispatchEvent(
      new CustomEvent(target === "batch" ? "applio:apply-preset-batch" : "applio:apply-preset", {
        detail: p.values,
      }),
    );
  }

  async function importFile(file: File | null) {
    setError("");
    if (!file) return;
    try {
      const raw = await file.text();
      const values = JSON.parse(raw) as Preset["values"];
      if (
        !values ||
        typeof values.pitch !== "number" ||
        typeof values.index_rate !== "number" ||
        typeof values.rms_mix_rate !== "number" ||
        typeof values.protect !== "number"
      ) {
        throw new Error(t("Not a valid preset file (needs pitch, index_rate, rms_mix_rate, protect)."));
      }
      const name = file.name.replace(/\.json$/i, "") || t("Imported Preset");
      await apiSend("/api/presets", "POST", { name, values });
      refresh();
    } catch (err) {
      setError(errMsg(err));
    }
  }

  async function saveCurrent() {
    setError("");
    const handler = (e: Event) => {
      const values = (e as CustomEvent).detail;
      apiSend("/api/presets", "POST", { name: name || t("My Preset"), values })
        .then(() => {
          setName("");
          refresh();
          toast(t("Preset saved."));
        })
        .catch((err) => setError(errMsg(err)));
    };
    window.addEventListener("applio:read-preset", handler, { once: true });
    window.dispatchEvent(new Event("applio:request-preset"));
    setTimeout(() => window.removeEventListener("applio:read-preset", handler), 2000);
  }

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Bookmark size={18} className="text-white" />
          <h2 className="text-base font-bold text-white m-0">{t("Presets")}</h2>
        </div>
        <button type="button" className="ghost" onClick={refresh}>
          <RefreshCw size={14} />
          {t("Refresh Presets")}
        </button>
      </div>
      <p className="muted text-xs m-0">
        {t("Stored in")} <code>assets/presets/*.json</code>
        {t(": pitch, search-feature-ratio, volume-envelope, protect.")}
      </p>
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mb-3 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
        >
          {error}
        </div>
      )}
      {presets.length === 0 ? (
        <p className="muted text-[13px] m-0">{t("No presets saved yet.")}</p>
      ) : (
        <div className="preset-grid">
          {presets.map((p) => (
            <div className="preset-tile" key={p.name}>
              <strong className="text-[13px] truncate" title={p.name}>
                {p.name}
              </strong>
              {p.values ? (
                <>
                  <div className="preset-stats">
                    <span className="preset-stat">
                      {t("Pitch")} <b>{p.values.pitch}</b>
                    </span>
                    <span className="preset-stat">
                      {t("Ratio")} <b>{p.values.index_rate}</b>
                    </span>
                    <span className="preset-stat">
                      {t("Envelope")} <b>{p.values.rms_mix_rate}</b>
                    </span>
                    <span className="preset-stat">
                      {t("Protect")} <b>{p.values.protect}</b>
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <button type="button" className="ghost preset-apply" onClick={() => apply(p, "single")}>
                      {t("Single")}
                    </button>
                    <button type="button" className="ghost preset-apply" onClick={() => apply(p, "batch")}>
                      {t("Batch")}
                    </button>
                  </div>
                </>
              ) : (
                <span className="muted text-xs">{t("unreadable")}</span>
              )}
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center gap-2 flex-wrap pt-3 border-t border-white/10">
        <input
          type="text"
          placeholder={t("Preset Name")}
          aria-label={t("Preset Name")}
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="max-w-[240px]"
        />
        <button type="button" className="ghost" onClick={saveCurrent}>
          <Save size={14} />
          {t("Save current Single settings")}
        </button>
        <label className="file-import">
          <Download size={14} />
          <span>{t("Import file")}</span>
          <input
            type="file"
            accept=".json"
            aria-label={t("Import preset JSON file")}
            className="sr-only"
            onChange={(e) => {
              importFile(e.target.files?.[0] || null);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      {formant.length > 0 && (
        <p className="muted text-xs m-0">
          {t("Formant presets in assets/formant_shift:")} {formant.join(", ")}
        </p>
      )}
    </div>
  );
}
