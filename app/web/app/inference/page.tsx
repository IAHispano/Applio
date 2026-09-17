"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
import PresetsPanel from "../../components/PresetsPanel";
import { useI18n } from "../../lib/i18n";

export default function InferencePage() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const { t } = useI18n();
  return (
    <div>
      <PageHeader
        title={t("Inference")}
        description={t("Convert audio files using trained voice models with single and batch processing.")}
      >
        <button
          type="button"
          className={mode === "single" ? "cta" : "ghost"}
          onClick={() => setMode("single")}
        >
          {t("Single")}
        </button>
        <button type="button" className={mode === "batch" ? "cta" : "ghost"} onClick={() => setMode("batch")}>
          {t("Batch")}
        </button>
      </PageHeader>
      {mode === "single" ? <InferenceForm /> : <BatchForm />}
      <PresetsPanel />
    </div>
  );
}
