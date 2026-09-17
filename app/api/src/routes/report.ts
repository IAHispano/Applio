import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { z } from "zod";
import { errMsg } from "../errors";
import { getOutputsDir, getRepoRoot, runPythonModule } from "../python";

const router = Router();
const ISSUE_URL = "https://github.com/IAHispano/Applio/issues/new";

router.get("/info", async (_req: Request, res: Response) => {
  let version = "unknown";
  try {
    version =
      JSON.parse(fs.readFileSync(path.join(getRepoRoot(), "assets", "config_template.json"), "utf-8"))
        .version || version;
  } catch {
    /* ignore */
  }
  let python = "";
  try {
    const r = await runPythonModule(["--version"]);
    python = (r.stdout + r.stderr).trim();
  } catch {
    /* ignore */
  }
  res.json({
    app: "Applio",
    version,
    platform: `${os.platform()} ${os.release()} (${os.arch()})`,
    node: process.version,
    python,
    cpus: os.cpus().length,
    totalMemGB: Math.round(os.totalmem() / 1024 ** 3),
    issueUrl: ISSUE_URL,
    time: new Date().toISOString(),
  });
});

router.post("/upload", (req: Request, res: Response) => {
  const parsed = z.object({ dataUrl: z.string().startsWith("data:video/") }).safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: "dataUrl must be a data:video/* URL" });
  try {
    const m = parsed.data.dataUrl.match(/^data:(video\/[a-z0-9.+-]+);base64,(.+)$/i);
    if (!m) return res.status(400).json({ error: "Malformed data URL" });
    const ext = m[1].includes("mp4") ? "mp4" : "webm";
    if (m[2].length > 200 * 1024 * 1024) return res.status(413).json({ error: "Clip too large (200MB max)" });
    const file = path.join(getOutputsDir(), `web_report_${Date.now()}.${ext}`);
    fs.writeFileSync(file, Buffer.from(m[2], "base64"));
    const rel = path.relative(getRepoRoot(), file).replace(/\\/g, "/");
    res.json({ ok: true, file: rel, issueUrl: ISSUE_URL });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

export default router;
