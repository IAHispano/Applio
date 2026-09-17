"use client";

import { useI18n } from "../lib/i18n";

// Radio group with Gradio gr.Radio look-and-feel (inline options).
export default function RadioRow({
  label,
  name,
  options,
  value,
  onChange,
}: {
  label: string;
  name: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  const { t } = useI18n();
  return (
    <div>
      <span className="muted">{t(label)}</span>
      <div className="row" role="radiogroup" aria-label={t(label)} style={{ flexWrap: "wrap" }}>
        {options.map((o) => (
          <label key={o} style={{ fontWeight: value === o ? 700 : 400 }}>
            <input type="radio" name={name} value={o} checked={value === o} onChange={() => onChange(o)} />{" "}
            {o}
          </label>
        ))}
      </div>
    </div>
  );
}
