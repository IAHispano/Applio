"use client";

import { useState } from "react";
import { Mic, Layers, FileText } from "lucide-react";
import BatchForm from "../../components/BatchForm";
import TtsForm from "../../components/convert/TtsForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
import SegmentedControl from "../../components/ui/SegmentedControl";
import { useI18n } from "../../lib/i18n";

type Mode = "single" | "batch" | "tts";

export default function ConvertPage() {
  const [mode, setMode] = useState<Mode>("single");
  const { t } = useI18n();

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Convert")}
        description={t("Turn any voice into another — single file, batch folder, or text-to-speech.")}
      >
        <SegmentedControl<Mode>
          value={mode}
          onChange={setMode}
          ariaLabel={t("Conversion mode")}
          tabPanels
          options={[
            { value: "single", label: t("Single Audio"), icon: Mic },
            { value: "batch", label: t("Batch Conversion"), icon: Layers },
            { value: "tts", label: t("Text-to-Speech"), icon: FileText },
          ]}
        />
      </PageHeader>
      <div id={`panel-${mode}`} role="tabpanel" aria-labelledby={`tab-${mode}`}>
        {mode === "single" && <InferenceForm />}
        {mode === "batch" && <BatchForm />}
        {mode === "tts" && <TtsForm />}
      </div>
    </div>
  );
}
