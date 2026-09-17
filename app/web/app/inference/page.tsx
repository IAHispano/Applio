"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
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
        <div className="row" role="tablist" aria-label={t("Inference mode")}>
          <button
            id="tab-single"
            type="button"
            role="tab"
            aria-selected={mode === "single"}
            aria-controls="panel-single"
            className={mode === "single" ? "cta" : "ghost"}
            onClick={() => setMode("single")}
          >
            {t("Single")}
          </button>
          <button
            id="tab-batch"
            type="button"
            role="tab"
            aria-selected={mode === "batch"}
            aria-controls="panel-batch"
            className={mode === "batch" ? "cta" : "ghost"}
            onClick={() => setMode("batch")}
          >
            {t("Batch")}
          </button>
        </div>
      </PageHeader>
      <div id={`panel-${mode}`} role="tabpanel" aria-labelledby={`tab-${mode}`}>
        {mode === "single" ? <InferenceForm /> : <BatchForm />}
      </div>
    </div>
  );
}
