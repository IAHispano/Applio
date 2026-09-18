"use client";

import { Layers, Music } from "lucide-react";
import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
import SegmentedControl from "../../components/ui/SegmentedControl";
import { useI18n } from "../../lib/i18n";

export default function InferencePage() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const { t } = useI18n();

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Inference")}
        description={t("Convert audio files using trained voice models with single and batch processing.")}
      >
        <SegmentedControl
          value={mode}
          onChange={setMode}
          ariaLabel={t("Inference mode")}
          tabPanels
          options={[
            { value: "single", label: t("Single"), icon: Music },
            { value: "batch", label: t("Batch"), icon: Layers },
          ]}
        />
      </PageHeader>
      <div id={`panel-${mode}`} role="tabpanel" aria-labelledby={`tab-${mode}`}>
        {mode === "single" ? <InferenceForm /> : <BatchForm />}
      </div>
    </div>
  );
}
