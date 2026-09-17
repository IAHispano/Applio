import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { runPythonJson } from "../cli";
import { errMsg } from "../errors";
import { appendLog, createJob, setDone, setError, setRunning } from "../jobs";
import { getRepoRoot, getUploadsDir, resolveUserPath } from "../python";

const router = Router();

const upload = multer({
  dest: getUploadsDir(),
  limits: { fileSize: 1024 * 1024 * 1024 }, // models are large
  fileFilter: (_req, file, cb) => {
    const n = file.originalname.toLowerCase();
    if (!n.endsWith(".pth") && !n.endsWith(".onnx")) {
      return cb(new Error("Only .pth/.onnx model files are accepted"));
    }
    cb(null, true);
  },
});

const blenderSchema = z.object({
  modelName: z.string().min(1).max(120),
  pthPath1: z.string().min(1),
  pthPath2: z.string().min(1),
  ratio: z.coerce.number().min(0).max(1).default(0.5),
});

// python -c JSON: the blender's return type is inconsistent across outcomes.
router.post(
  "/",
  upload.fields([
    { name: "pth_file_1", maxCount: 1 },
    { name: "pth_file_2", maxCount: 1 },
  ]),
  async (req: Request, res: Response) => {
    const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
    try {
      const body = { ...(req.body as Record<string, unknown>) };
      if (files.pth_file_1?.[0]) body.pthPath1 = files.pth_file_1[0].path;
      if (files.pth_file_2?.[0]) body.pthPath2 = files.pth_file_2[0].path;
      const parsed = blenderSchema.safeParse(body);
      if (!parsed.success) {
        return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
      }
      const p = parsed.data;
      const p1 = resolveUserPath(p.pthPath1);
      const p2 = resolveUserPath(p.pthPath2);
      if (!fs.existsSync(p1)) return res.status(400).json({ error: `Model 1 not found: ${p.pthPath1}` });
      if (!fs.existsSync(p2)) return res.status(400).json({ error: `Model 2 not found: ${p.pthPath2}` });
      const safeName = path.basename(p.modelName).replace(/[^a-zA-Z0-9._-]/g, "_");

      const job = createJob("other", { ...p });
      void (async () => {
        setRunning(job);
        try {
          appendLog(job, `Blending into logs/${safeName}.pth (ratio ${p.ratio})`);
          const code = [
            "import json",
            "from core import run_model_blender_script",
            `r = run_model_blender_script(${JSON.stringify(safeName)}, ${JSON.stringify(p1)}, ${JSON.stringify(p2)}, ${p.ratio})`,
            "msg, f = (r if isinstance(r, tuple) else (str(r), None))",
            "print('APPLIO_JSON:' + json.dumps({'message': msg, 'file': f}))",
          ].join("; ");
          const out = await runPythonJson<{ message: string; file: string | null }>(code, (l) =>
            appendLog(job, l),
          );
          if (!out.file || !fs.existsSync(path.resolve(getRepoRoot(), out.file))) {
            throw new Error(out.message || "Blending failed");
          }
          appendLog(job, out.message);
          setDone(job, { message: out.message }, out.file);
        } catch (err) {
          appendLog(job, `ERROR: ${errMsg(err)}`);
          setError(job, errMsg(err) || "Blending failed");
        }
      })();
      return res.status(202).json({ jobId: job.id });
    } catch (err) {
      return res.status(500).json({ error: errMsg(err) || "Blender failed to start" });
    }
  },
);

export default router;
