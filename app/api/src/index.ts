import "dotenv/config";
import cors from "cors";
import express from "express";
import { killJobTree } from "./cli";
import { errMsg } from "./errors";
import { getJob, setError } from "./jobs";
import { getOutputsDir, getPythonBin, getRepoRoot, runPythonModule } from "./python";
import batchRouter from "./routes/batch";
import blenderRouter from "./routes/blender";
import downloadRouter from "./routes/download";
import extraRouter from "./routes/extra";
import inferenceRouter from "./routes/inference";
import jobsRouter from "./routes/jobs";
import modelsRouter from "./routes/models";
import pluginsRouter from "./routes/plugins";
import presetsRouter from "./routes/presets";
import realtimeRouter, { attachRealtimeProxy } from "./routes/realtime";
import reportRouter from "./routes/report";
import settingsRouter from "./routes/settings";
import setupRouter from "./routes/setup";
import tensorboardRouter from "./routes/tensorboard";
import trainRouter from "./routes/train";
import ttsRouter from "./routes/tts";

const app = express();
const PORT = Number(process.env.API_PORT || process.env.PORT || 8000);

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// Static files produced by jobs (audio, plots, clips).
const outputsDir = getOutputsDir();
app.use("/outputs", express.static(outputsDir, { maxAge: "1h", fallthrough: true }));

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    service: "applio-api",
    repoRoot: getRepoRoot(),
    python: getPythonBin(),
    time: new Date().toISOString(),
  });
});

app.get("/api/diagnostics", async (_req, res) => {
  try {
    const r = await runPythonModule(["--version"]);
    const torch = await runPythonModule(["-c", "import torch; print(torch.__version__)"], {}).catch(
      (e: unknown) => ({
        stdout: "",
        stderr: String(e),
        code: 1,
      }),
    );
    res.json({
      pythonVersion: (r.stdout + r.stderr).trim(),
      torchVersion: (torch.stdout + torch.stderr).trim(),
      repoRoot: getRepoRoot(),
    });
  } catch (e) {
    res.status(500).json({ error: errMsg(e) });
  }
});

app.use("/api/models", modelsRouter);
app.use("/api/inference/batch", batchRouter); // before /api/inference (more specific first)
app.use("/api/inference", inferenceRouter);
app.use("/api/presets", presetsRouter);
app.use("/api/tts", ttsRouter);
app.use("/api/voice-blender", blenderRouter);
app.use("/api/download", downloadRouter);
app.use("/api/extra", extraRouter);
app.use("/api/train", trainRouter);
app.use("/api/settings", settingsRouter);
app.use("/api/tensorboard", tensorboardRouter);
app.use("/api/report", reportRouter);
app.use("/api/plugins", pluginsRouter);
app.use("/api/realtime", realtimeRouter);
app.use("/api/jobs", jobsRouter);
app.use("/api/setup", setupRouter);

app.post("/api/jobs/:id/stop", (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (job.status === "done" || job.status === "error") {
    return res.json({ ok: true, alreadyFinished: true });
  }
  const killed = killJobTree(job.id);
  if (killed) {
    setError(job, "Stopped by user");
    return res.json({ ok: true });
  }
  return res.status(404).json({ error: "Job has no running process (may have finished starting)." });
});

const server = app.listen(PORT, "127.0.0.1", () => {
  // eslint-disable-next-line no-console
  console.log(`[applio-api] listening on http://127.0.0.1:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`[applio-api] repoRoot=${getRepoRoot()} outputs=${outputsDir}`);
});

// Realtime audio frames ride raw WebSockets (Next rewrites don't proxy upgrades),
// so the WS proxy attaches directly to our HTTP server.
attachRealtimeProxy(server);

process.on("uncaughtException", (err) => {
  // eslint-disable-next-line no-console
  console.error("[applio-api] Uncaught exception:", err);
});
process.on("unhandledRejection", (reason) => {
  // eslint-disable-next-line no-console
  console.error("[applio-api] Unhandled rejection:", reason);
});
