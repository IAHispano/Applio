"use client";

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
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>{t("Presets")}</h2>
        <button type="button" className="ghost" onClick={refresh}>
          {t("Refresh Presets")}
        </button>
      </div>
      <p className="muted">
        {t("Stored in")} <code>assets/presets/*.json</code>
        {t(": pitch, search-feature-ratio, volume-envelope, protect.")}
      </p>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      {presets.map((p) => (
        <div className="row" key={p.name} style={{ marginBottom: 8 }}>
          <strong>{p.name}</strong>
          <span className="muted">
            {p.values
              ? `pitch ${p.values.pitch} · ratio ${p.values.index_rate} · envelope ${p.values.rms_mix_rate} · protect ${p.values.protect}`
              : t("unreadable")}
          </span>
          {p.values && (
            <>
              <button type="button" className="ghost" onClick={() => apply(p, "single")}>
                {t("Apply to Single")}
              </button>
              <button type="button" className="ghost" onClick={() => apply(p, "batch")}>
                {t("Apply to Batch")}
              </button>
            </>
          )}
        </div>
      ))}
      <div className="row" style={{ marginTop: 12 }}>
        <input
          type="text"
          placeholder={t("Preset Name")}
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ maxWidth: 240 }}
        />
        <button type="button" className="ghost" onClick={saveCurrent}>
          {t("Save current Single settings")}
        </button>
        <label className="ghost" style={{ cursor: "pointer" }}>
          {t("Import file")}
          <input
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(e) => {
              importFile(e.target.files?.[0] || null);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      {formant.length > 0 && (
        <p className="muted" style={{ marginTop: 12 }}>
          {t("Formant presets in assets/formant_shift:")} {formant.join(", ")}
        </p>
      )}
    </div>
  );
}
