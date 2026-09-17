"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import InferenceForm from "../../components/InferenceForm";
import PageHeader from "../../components/layout/PageHeader";
import PresetsPanel from "../../components/PresetsPanel";

export default function InferencePage() {
  const [mode, setMode] = useState<"single" | "batch">("single");

  return (
    <div>
      <PageHeader
        title="Inference"
        description="Convert audio files using trained voice models with single and batch processing."
      >
        <button
          type="button"
          className={mode === "single" ? "cta" : "ghost"}
          onClick={() => setMode("single")}
        >
          Single
        </button>
        <button type="button" className={mode === "batch" ? "cta" : "ghost"} onClick={() => setMode("batch")}>
          Batch
        </button>
      </PageHeader>
      {mode === "single" ? <InferenceForm /> : <BatchForm />}
      <PresetsPanel />
    </div>
  );
}
