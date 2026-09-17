"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import TtsForm from "../../components/convert/TtsForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
import { useI18n } from "../../lib/i18n";

type Mode = "single" | "batch" | "tts";

const MODES: Array<{ id: Mode; label: string }> = [
  { id: "single", label: "Single" },
  { id: "batch", label: "Batch" },
  { id: "tts", label: "Text to Speech" },
];

export default function ConvertPage() {
  const [mode, setMode] = useState<Mode>("single");
  const { t } = useI18n();
  return (
    <div>
      <PageHeader
        title={t("Convert")}
        description={t("Turn any voice into another — single file, batch folder, or text-to-speech.")}
      >
        <div className="row" role="tablist" aria-label={t("Conversion mode")}>
          {MODES.map((m) => (
            <button
              key={m.id}
              id={`tab-${m.id}`}
              type="button"
              role="tab"
              aria-selected={mode === m.id}
              aria-controls={`panel-${m.id}`}
              className={mode === m.id ? "cta" : "ghost"}
              onClick={() => setMode(m.id)}
            >
              {t(m.label)}
            </button>
          ))}
        </div>
      </PageHeader>
      <div id={`panel-${mode}`} role="tabpanel" aria-labelledby={`tab-${mode}`}>
        {mode === "single" && <InferenceForm />}
        {mode === "batch" && <BatchForm />}
        {mode === "tts" && <TtsForm />}
      </div>
    </div>
  );
}
