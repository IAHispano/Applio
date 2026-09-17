"use client";

import { useState } from "react";
import BatchForm from "../../components/BatchForm";
import PresetsPanel from "../../components/PresetsPanel";
import InferenceForm from "../../components/InferenceForm";
import TtsForm from "../../components/convert/TtsForm";

type Mode = "single" | "batch" | "tts";

const MODES: Array<{ id: Mode; label: string }> = [
  { id: "single", label: "Single" },
  { id: "batch", label: "Batch" },
  { id: "tts", label: "Text to Speech" },
];

export default function ConvertPage() {
  const [mode, setMode] = useState<Mode>("single");
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="title" style={{ margin: 0 }}>
            Convert
          </h2>
          <p className="muted" style={{ margin: "4px 0 0" }}>
            Turn any voice into another — a file, a folder, or written text.
          </p>
        </div>
        <div className="row">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={mode === m.id ? "" : "ghost"}
              onClick={() => setMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      {mode === "single" && <InferenceForm />}
      {mode === "batch" && <BatchForm />}
      {mode === "tts" && <TtsForm />}
      {mode === "single" && <PresetsPanel />}
    </div>
  );
}
