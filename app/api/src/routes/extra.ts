import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { runPythonJson, startCliJob } from "../cli";
import { errMsg } from "../errors";
import { appendLog, createJob, setDone, setError, setRunning } from "../jobs";
import { getOutputsDir, getRepoRoot, getUploadsDir, resolveUserPath } from "../python";

const router = Router();

const AUDIO_EXTS = [
  ".wav",
  ".mp3",
  ".flac",
  ".ogg",
  ".opus",
  ".m4a",
  ".mp4",
  ".aac",
  ".alac",
  ".wma",
  ".aiff",
  ".webm",
  ".ac3",
];
const upload = multer({
  dest: getUploadsDir(),
  limits: { fileSize: 200 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!AUDIO_EXTS.includes(path.extname(file.originalname).toLowerCase())) {
      return cb(new Error("Unsupported audio type"));
    }
    cb(null, true);
  },
});

function inputFrom(req: Request): string {
  if (req.file) return req.file.path;
  const p = (req.body as Record<string, unknown>).inputPath;
  if (typeof p === "string" && p) return resolveUserPath(p);
  throw new Error("Provide an 'audio' upload or 'inputPath'.");
}

router.post("/analyze", upload.single("audio"), async (req: Request, res: Response) => {
  try {
    const inputAbs = inputFrom(req);
    const ts = Date.now();
    const plotAbs = path.join(getOutputsDir(), `audio_analysis_${ts}.png`);
    const job = createJob("other", { inputPath: inputAbs });
    void (async () => {
      setRunning(job);
      try {
        const code = [
          "import json",
          "from core import run_audio_analyzer_script",
          `info, plot = run_audio_analyzer_script(${JSON.stringify(inputAbs)}, ${JSON.stringify(plotAbs)})`,
          "print('APPLIO_JSON:' + json.dumps({'info': info, 'plot': plot}))",
        ].join("; ");
        const out = await runPythonJson<{ info: unknown; plot: string }>(code, (l) => appendLog(job, l));
        appendLog(job, `Plot saved at ${out.plot}`);
        setDone(job, { info: out.info }, path.relative(getRepoRoot(), out.plot).replace(/\\/g, "/"));
      } catch (err) {
        appendLog(job, `ERROR: ${errMsg(err)}`);
        setError(job, errMsg(err) || "Analysis failed");
      }
    })();
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    return res.status(400).json({ error: errMsg(err) });
  }
});

router.post("/model-info", (req: Request, res: Response) => {
  const parsed = z.object({ pthPath: z.string().min(1) }).safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  try {
    const abs = resolveUserPath(parsed.data.pthPath);
    if (!fs.existsSync(abs)) return res.status(400).json({ error: `File not found: ${parsed.data.pthPath}` });
    const job = startCliJob("other", parsed.data, ["core.py", "model-information", "--pth-path", abs]);
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    return res.status(400).json({ error: errMsg(err) });
  }
});

router.post("/f0", upload.single("audio"), (req: Request, res: Response) => {
  try {
    const inputAbs = inputFrom(req);
    const parsed = z
      .object({ method: z.enum(["crepe", "fcpe", "rmvpe"]).default("rmvpe") })
      .safeParse(req.body);
    if (!parsed.success)
      return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
    const ts = Date.now();
    const imgAbs = path.join(getOutputsDir(), `f0_plot_${ts}.png`);
    const txtAbs = path.join(getOutputsDir(), `f0_curve_${ts}.txt`);
    const job = startCliJob(
      "other",
      { inputPath: inputAbs, method: parsed.data.method },
      [
        "core.py",
        "f0-curve",
        "--input-path",
        inputAbs,
        "--method",
        parsed.data.method,
        "--output-image",
        imgAbs,
        "--output-txt",
        txtAbs,
      ],
      {
        parse: () => {
          if (!fs.existsSync(imgAbs)) throw new Error("F0 extraction finished but no plot was found.");
          return {
            result: { curveFile: path.relative(getRepoRoot(), txtAbs).replace(/\\/g, "/") },
            outputFile: path.relative(getRepoRoot(), imgAbs).replace(/\\/g, "/"),
          };
        },
      },
    );
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    return res.status(400).json({ error: errMsg(err) });
  }
});

export default router;
