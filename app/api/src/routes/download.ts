import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { startCliJob } from "../cli";
import { errMsg } from "../errors";
import { appendLog, createJob, setDone, setError, setRunning } from "../jobs";
import { getRepoRoot, getUploadsDir } from "../python";

const router = Router();

const upload = multer({
  dest: getUploadsDir(),
  limits: { fileSize: 1024 * 1024 * 1024 },
});

// Port of rvc/lib/utils.py format_title: NFC, drop box-drawing chars, keep
// word chars/spaces/dots/dashes, spaces become underscores. \p{L}\p{N} (not
// \w, which is ASCII-only in JS) matches Python re.UNICODE \w incl. CJK.
function formatTitle(title: string): string {
  let s = title.normalize("NFC").replace(/[─-╿]+/g, "");
  s = s.replace(/[^\p{L}\p{N}_\s.-]/gu, "");
  s = s.replace(/\s+/g, "_");
  return s || "model";
}

router.post("/", (req: Request, res: Response) => {
  const parsed = z.object({ modelLink: z.string().url() }).safeParse(req.body);
  if (!parsed.success) {
    return res
      .status(400)
      .json({ error: "Invalid params (modelLink must be a URL)", details: parsed.error.flatten() });
  }
  const job = startCliJob("download", parsed.data, [
    "core.py",
    "download",
    "--model-link",
    parsed.data.modelLink,
  ]);
  return res.status(202).json({ jobId: job.id });
});

router.post("/drop", upload.single("file"), (req: Request, res: Response) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Upload a 'file' (.pth, .onnx or .index)." });
    const original = req.file.originalname;
    const lower = original.toLowerCase();
    if (!lower.endsWith(".pth") && !lower.endsWith(".index") && !lower.endsWith(".onnx")) {
      fs.rmSync(req.file.path, { force: true });
      return res.status(400).json({ error: "Not a valid model file (need .pth, .onnx or .index)." });
    }
    // Same sanitizing as the Gradio backend (rvc/lib/utils.py format_title).
    const fileName = formatTitle(path.basename(original));
    const lowerName = fileName.toLowerCase();
    let modelName = fileName;
    if (lowerName.includes(".pth")) modelName = fileName.slice(0, lowerName.indexOf(".pth"));
    else if (lowerName.includes(".onnx")) modelName = fileName.slice(0, lowerName.indexOf(".onnx"));
    else if (lowerName.includes(".index")) {
      for (const rep of ["nprobe_1_", "_v1", "_v2", "added_"]) modelName = modelName.replace(rep, "");
      modelName = modelName.slice(0, modelName.toLowerCase().indexOf(".index"));
    }
    if (!modelName) modelName = "model";
    const modelDir = path.join(getRepoRoot(), "logs", modelName);
    fs.mkdirSync(modelDir, { recursive: true });
    const dest = path.join(modelDir, fileName);
    if (fs.existsSync(dest)) fs.rmSync(dest);
    fs.renameSync(req.file.path, dest);
    return res.json({ ok: true, file: fileName, modelDir: `logs/${modelName}` });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    return res.status(500).json({ error: errMsg(err) });
  }
});

const PRETRAINS_URL = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json";

function pretrainedsCachePath(): string {
  const dir = path.join(getRepoRoot(), "rvc", "models", "pretraineds", "custom");
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "pretrains.json");
}

async function fetchPretrainedData(): Promise<Record<string, Record<string, { D: string; G: string }>>> {
  const cache = pretrainedsCachePath();
  try {
    return JSON.parse(fs.readFileSync(cache, "utf-8"));
  } catch {
    /* fetch below */
  }
  const r = await fetch(PRETRAINS_URL);
  if (!r.ok) throw new Error(`Could not fetch pretrains.json (${r.status})`);
  const data = (await r.json()) as Record<string, Record<string, { D: string; G: string }>>;
  fs.writeFileSync(cache, JSON.stringify(data, null, 2));
  return data;
}

router.get("/pretraineds", async (_req: Request, res: Response) => {
  try {
    const data = await fetchPretrainedData();
    res.json({
      models: Object.entries(data).map(([name, srs]) => ({ name, sampleRates: Object.keys(srs) })),
    });
  } catch (err) {
    res.json({ models: [{ name: "Titan", sampleRates: ["32k"] }], warning: errMsg(err) });
  }
});

router.post("/pretraineds", async (req: Request, res: Response) => {
  const parsed = z
    .object({
      model: z.string().optional(),
      sampleRate: z.string().optional(),
      urlG: z.string().optional().default(""),
      urlD: z.string().optional().default(""),
    })
    .safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const { model, sampleRate } = parsed.data;
  const { urlG, urlD } = parsed.data;
  try {
    let tasks: Array<{ url: string; dest: string }> = [];
    const saveDir = path.join(getRepoRoot(), "rvc", "models", "pretraineds", "custom");
    fs.mkdirSync(saveDir, { recursive: true });
    if (urlG || urlD) {
      tasks = [urlG, urlD].filter(Boolean).map((u) => ({
        url: u.replace("?download=true", ""),
        dest: path.join(saveDir, path.basename(u.split("?")[0])),
      }));
    } else {
      if (!model || !sampleRate)
        return res.status(400).json({ error: "Provide model+sampleRate or urlG/urlD." });
      const data = await fetchPretrainedData();
      const entry = data[model]?.[sampleRate];
      if (!entry) return res.status(400).json({ error: "Unknown pretrained model/sample rate." });
      tasks = [entry.D, entry.G]
        .filter((p) => p && p !== "null")
        .map((p) => ({ url: `https://huggingface.co/${p}`, dest: path.join(saveDir, path.basename(p)) }));
    }
    if (tasks.length === 0) return res.status(400).json({ error: "Nothing to download." });

    const job = createJob("download", { model, sampleRate, files: tasks.map((t) => t.url) });
    void (async () => {
      setRunning(job);
      try {
        for (const t of tasks) {
          appendLog(job, `Downloading ${t.url}`);
          const head = await fetch(t.url, { method: "HEAD" }).catch(() => null);
          const total = Number(head?.headers.get("content-length") || 0);
          const r = await fetch(t.url);
          if (!r.ok || !r.body) throw new Error(`Download failed (${r.status}): ${t.url}`);
          fs.mkdirSync(path.dirname(t.dest), { recursive: true });
          const file = fs.createWriteStream(t.dest);
          const reader = r.body.getReader();
          let done = 0;
          let lastPct = -1;
          for (;;) {
            const { done: end, value } = await reader.read();
            if (end) break;
            done += value.length;
            await new Promise<void>((resolve, reject) =>
              file.write(value, (e) => (e ? reject(e) : resolve())),
            );
            if (total > 0) {
              const pct = Math.floor((done / total) * 100);
              if (pct >= lastPct + 10) {
                lastPct = pct;
                appendLog(job, `${path.basename(t.dest)}: ${pct}%`);
              }
            }
          }
          await new Promise<void>((resolve) => file.close(() => resolve()));
          appendLog(job, `Saved ${t.dest}`);
        }
        setDone(job, { message: "Pretrained model downloaded successfully!" });
      } catch (err) {
        appendLog(job, `ERROR: ${errMsg(err)}`);
        setError(job, errMsg(err) || "Download failed");
      }
    })();
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) });
  }
});

export default router;
