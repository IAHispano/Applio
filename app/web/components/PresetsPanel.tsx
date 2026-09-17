"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiSend, errMsg } from "../lib/api";

interface Preset {
  name: string;
  values: { pitch: number; index_rate: number; rms_mix_rate: number; protect: number } | null;
}

export default function PresetsPanel() {
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

  function apply(p: Preset) {
    if (!p.values) return;
    window.dispatchEvent(new CustomEvent("applio:apply-preset", { detail: p.values }));
  }

  async function saveCurrent() {
    setError("");
    const handler = (e: Event) => {
      const values = (e as CustomEvent).detail;
      apiSend("/api/presets", "POST", { name: name || "My Preset", values })
        .then(() => {
          setName("");
          refresh();
        })
        .catch((err) => setError(errMsg(err)));
    };
    window.addEventListener("applio:read-preset", handler, { once: true });
    window.dispatchEvent(new Event("applio:request-preset"));
    setTimeout(() => window.removeEventListener("applio:read-preset", handler), 2000);
  }

  return (
    <div className="card">
      <h2>Presets</h2>
      <p className="muted">
        Stored in <code>assets/presets/*.json</code>: pitch, search-feature-ratio, volume-envelope, protect.
      </p>
      {error && <p style={{ color: "var(--err)" }}>{error}</p>}
      {presets.map((p) => (
        <div className="row" key={p.name} style={{ marginBottom: 8 }}>
          <strong>{p.name}</strong>
          <span className="muted">
            {p.values
              ? `pitch ${p.values.pitch} · ratio ${p.values.index_rate} · envelope ${p.values.rms_mix_rate} · protect ${p.values.protect}`
              : "unreadable"}
          </span>
          {p.values && (
            <button type="button" className="ghost" onClick={() => apply(p)}>
              Apply to Single
            </button>
          )}
        </div>
      ))}
      <div className="row" style={{ marginTop: 12 }}>
        <input
          type="text"
          placeholder="Preset name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ maxWidth: 240 }}
        />
        <button type="button" className="ghost" onClick={saveCurrent}>
          Save current Single settings
        </button>
      </div>
      {formant.length > 0 && (
        <p className="muted" style={{ marginTop: 12 }}>
          Formant presets in assets/formant_shift: {formant.join(", ")}
        </p>
      )}
    </div>
  );
}
