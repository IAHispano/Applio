"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import InferenceForm from "../../components/InferenceForm";
import PresetsPanel from "../../components/PresetsPanel";

export default function InferencePage() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="title" style={{ margin: 0 }}>
          Inference
        </h2>
        <div className="row">
          <button
            type="button"
            className={mode === "single" ? "" : "ghost"}
            onClick={() => setMode("single")}
          >
            Single
          </button>
          <button type="button" className={mode === "batch" ? "" : "ghost"} onClick={() => setMode("batch")}>
            Batch
          </button>
        </div>
      </div>
      {mode === "single" ? <InferenceForm /> : <BatchForm />}
      <PresetsPanel />
    </div>
  );
}
