"use client";

import { useState } from "react";
import JobPanel from "../../components/JobPanel";
import PageHeader from "../../components/layout/PageHeader";
import BlenderPanel from "../../components/models/BlenderPanel";
import DownloadPanel from "../../components/models/DownloadPanel";
import { errMsg, submitJob } from "../../lib/api";

type Section = "download" | "blend" | "inspect";

export default function ModelsPage() {
  const [section, setSection] = useState<Section>("download");
  const [pth, setPth] = useState("");
  const [infoJob, setInfoJob] = useState<string | null>(null);
  const [error, setError] = useState("");

  async function inspect() {
    setError("");
    try {
      const { jobId: id } = await submitJob("/api/extra/model-info", { pthPath: pth });
      setInfoJob(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  return (
    <div>
      <PageHeader
        title="Models"
        description="Download, blend, and inspect voice model weights and checkpoints."
      >
        <div className="row">
          {(
            [
              ["download", "Download"],
              ["blend", "Blend"],
              ["inspect", "Inspect"],
            ] as Array<[Section, string]>
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={section === id ? "cta" : "ghost"}
              onClick={() => setSection(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </PageHeader>

      {section === "download" && <DownloadPanel />}
      {section === "blend" && <BlenderPanel />}
      {section === "inspect" && (
        <div>
          <div className="card">
            <div className="row">
              <input
                type="text"
                value={pth}
                onChange={(e) => setPth(e.target.value)}
                placeholder="logs/my-model/model.pth"
                style={{ flex: 1 }}
              />
              <button type="button" className="cta" onClick={inspect}>
                Inspect
              </button>
            </div>
            {error && <p style={{ color: "var(--err)" }}>{error}</p>}
          </div>
          <JobPanel jobId={infoJob} compact />
        </div>
      )}
    </div>
  );
}
