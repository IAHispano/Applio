import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { killJobTree, runPythonJson, startCliJob } from "../cli";
import { errMsg } from "../errors";
import { getRepoRoot, getUploadsDir } from "../python";

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

function walkAudioDirs(root: string): string[] {
  if (!fs.existsSync(root)) return [];
  const out: string[] = [];
  const walk = (dir: string) => {
    let hasAudio = false;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (AUDIO_EXTS.includes(path.extname(e.name).toLowerCase())) hasAudio = true;
    }
    if (hasAudio) out.push(path.relative(getRepoRoot(), dir).replace(/\\/g, "/"));
  };
  walk(root);
  return out.sort();
}

function walkFiles(root: string, exts: string[], exclude: (f: string) => boolean = () => false): string[] {
  if (!fs.existsSync(root)) return [];
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (exts.includes(path.extname(e.name).toLowerCase()) && !exclude(e.name)) {
        out.push(path.relative(getRepoRoot(), full).replace(/\\/g, "/"));
      }
    }
  };
  walk(root);
  return out.sort();
}

router.get("/datasets", (_req: Request, res: Response) => {
  res.json({ datasets: walkAudioDirs(path.join(getRepoRoot(), "assets", "datasets")) });
});

router.get("/pretraineds", (_req: Request, res: Response) => {
  const root = path.join(getRepoRoot(), "rvc", "models", "pretraineds", "custom");
  const all = walkFiles(root, [".pth"]);
  res.json({
    g: all.filter((f) => path.basename(f).includes("G")),
    d: all.filter((f) => path.basename(f).includes("D")),
  });
});

router.get("/embedders", (_req: Request, res: Response) => {
  const root = path.join(getRepoRoot(), "rvc", "models", "embedders", "embedders_custom");
  if (!fs.existsSync(root)) return res.json({ embedders: [] });
  res.json({ embedders: fs.readdirSync(root).filter((e) => fs.statSync(path.join(root, e)).isDirectory()) });
});

router.get("/gpus", async (_req: Request, res: Response) => {
  try {
    const code = [
      "import json",
      "from rvc.configs.config import get_gpu_info, get_number_of_gpus",
      "print('APPLIO_JSON:' + json.dumps({'count': get_number_of_gpus(), 'info': get_gpu_info()}))",
    ].join("; ");
    res.json(await runPythonJson<{ count: number; info: string }>(code));
  } catch (err) {
    res.status(500).json({ error: errMsg(err) || "GPU query failed" });
  }
});

router.get("/exports", (_req: Request, res: Response) => {
  const logs = path.join(getRepoRoot(), "logs");
  res.json({
    models: walkFiles(logs, [".pth"]),
    indexes: walkFiles(logs, [".index"], (f) => f.includes("trained")),
  });
});

const upload = multer({ dest: getUploadsDir(), limits: { fileSize: 2 * 1024 * 1024 * 1024 } });
const clean = (n: string) => path.basename(n).replace(/[^a-zA-Z0-9._() -]/g, "_");

router.post("/upload-dataset", upload.array("files", 2000), (req: Request, res: Response) => {
  try {
    const name = clean(String((req.body as Record<string, unknown>).datasetName || ""));
    if (!name) return res.status(400).json({ error: "datasetName is required" });
    const dir = path.join(getRepoRoot(), "assets", "datasets", name);
    fs.mkdirSync(dir, { recursive: true });
    const saved: string[] = [];
    for (const f of (req.files || []) as Express.Multer.File[]) {
      const dest = path.join(dir, clean(f.originalname));
      fs.renameSync(f.path, dest);
      saved.push(`assets/datasets/${name}/${path.basename(dest)}`);
    }
    res.json({ ok: true, dataset: `assets/datasets/${name}`, files: saved.length });
  } catch (err) {
    for (const f of (req.files || []) as Express.Multer.File[]) fs.rmSync(f.path, { force: true });
    res.status(500).json({ error: errMsg(err) });
  }
});

router.post("/upload-pretrained", upload.single("file"), (req: Request, res: Response) => {
  try {
    if (!req.file?.originalname.includes(".pth")) {
      if (req.file) fs.rmSync(req.file.path, { force: true });
      return res.status(400).json({ error: "Upload a .pth file." });
    }
    const dir = path.join(getRepoRoot(), "rvc", "models", "pretraineds", "custom");
    fs.mkdirSync(dir, { recursive: true });
    const dest = path.join(dir, clean(req.file.originalname));
    fs.renameSync(req.file.path, dest);
    res.json({ ok: true, file: path.relative(getRepoRoot(), dest).replace(/\\/g, "/") });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    res.status(500).json({ error: errMsg(err) });
  }
});

router.post(
  "/upload-embedder",
  upload.fields([
    { name: "bin", maxCount: 1 },
    { name: "config", maxCount: 1 },
  ]),
  (req: Request, res: Response) => {
    const files = (req.files || {}) as Record<string, Express.Multer.File[]>;
    try {
      const folder = clean(String((req.body as Record<string, unknown>).folderName || "")).replace(
        /\.[^.]*$/,
        "",
      );
      if (!folder) return res.status(400).json({ error: "folderName is required" });
      if (!files.bin?.[0] || !files.config?.[0])
        return res.status(400).json({ error: "Upload both .bin and .json." });
      const dir = path.join(getRepoRoot(), "rvc", "models", "embedders", "embedders_custom", folder);
      fs.mkdirSync(dir, { recursive: true });
      fs.renameSync(files.bin[0].path, path.join(dir, clean(files.bin[0].originalname)));
      fs.renameSync(files.config[0].path, path.join(dir, clean(files.config[0].originalname)));
      res.json({ ok: true, folder });
    } catch (err) {
      for (const k of Object.keys(files)) for (const f of files[k]) fs.rmSync(f.path, { force: true });
      res.status(500).json({ error: errMsg(err) });
    }
  },
);

const maxCores = Math.min(os.cpus().length, 32);
const modelName = z.string().min(1).max(120);

router.post("/preprocess", (req: Request, res: Response) => {
  const parsed = z
    .object({
      modelName,
      datasetPath: z.string().min(1),
      sampleRate: z.enum(["32000", "40000", "48000"]).default("40000"),
      cpuCores: z.coerce.number().int().min(1).max(64).default(maxCores),
      cutPreprocess: z.enum(["Skip", "Simple", "Automatic"]).default("Automatic"),
      processEffects: z.coerce.boolean().default(false),
      noiseReduction: z.coerce.boolean().default(false),
      cleanStrength: z.coerce.number().min(0).max(1).default(0.5),
      chunkLen: z.coerce.number().min(0.5).max(5).default(3.0),
      overlapLen: z.coerce.number().min(0).max(0.4).default(0.3),
      normalizationMode: z.enum(["none", "pre", "post"]).default("post"),
    })
    .safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const p = parsed.data;
  const job = startCliJob("train", { step: "preprocess", ...p }, [
    "core.py",
    "preprocess",
    "--model-name",
    p.modelName,
    "--dataset-path",
    p.datasetPath,
    "--sample-rate",
    p.sampleRate,
    "--cpu-cores",
    String(p.cpuCores),
    "--cut-preprocess",
    p.cutPreprocess,
    ...(p.processEffects ? ["--process-effects"] : []),
    ...(p.noiseReduction ? ["--noise-reduction"] : []),
    "--noise-reduction-strength",
    String(p.cleanStrength),
    "--chunk-len",
    String(p.chunkLen),
    "--overlap-len",
    String(p.overlapLen),
    "--normalization-mode",
    p.normalizationMode,
  ]);
  return res.status(202).json({ jobId: job.id });
});

router.post("/extract", (req: Request, res: Response) => {
  const parsed = z
    .object({
      modelName,
      f0Method: z.enum(["crepe", "crepe-tiny", "rmvpe"]).default("rmvpe"),
      cpuCores: z.coerce.number().int().min(1).max(64).default(maxCores),
      gpu: z.string().default("0"),
      sampleRate: z.enum(["32000", "40000", "44100", "48000"]).default("40000"),
      embedderModel: z
        .enum([
          "contentvec",
          "spin",
          "spin-v2",
          "chinese-hubert-base",
          "japanese-hubert-base",
          "korean-hubert-base",
          "custom",
        ])
        .default("contentvec"),
      embedderModelCustom: z.string().optional(),
      includeMutes: z.coerce.number().int().min(0).max(10).default(2),
    })
    .safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const p = parsed.data;
  const args = [
    "core.py",
    "extract",
    "--model-name",
    p.modelName,
    "--f0-method",
    p.f0Method,
    "--cpu-cores",
    String(p.cpuCores),
    "--gpu",
    p.gpu,
    "--sample-rate",
    p.sampleRate,
    "--embedder-model",
    p.embedderModel,
    "--include-mutes",
    String(p.includeMutes),
  ];
  if (p.embedderModelCustom) args.push("--embedder-model-custom", p.embedderModelCustom);
  const job = startCliJob("train", { step: "extract", ...p }, args);
  return res.status(202).json({ jobId: job.id });
});

// shutdown_check is intentionally not exposed — the server must stay up.
router.post("/train", (req: Request, res: Response) => {
  const parsed = z
    .object({
      modelName,
      vocoder: z.enum(["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]).default("HiFi-GAN"),
      checkpointing: z.coerce.boolean().default(false),
      saveEveryEpoch: z.coerce.number().int().min(1).max(100).default(10),
      saveOnlyLatest: z.coerce.boolean().default(true),
      saveEveryWeights: z.coerce.boolean().default(true),
      totalEpoch: z.coerce.number().int().min(1).max(10000).default(200),
      sampleRate: z.string().default("40000"),
      batchSize: z.coerce.number().int().min(1).max(64).default(4),
      gpu: z.string().default("0"),
      pretrained: z.coerce.boolean().default(true),
      customPretrained: z.coerce.boolean().default(false),
      gPretrainedPath: z.string().optional(),
      dPretrainedPath: z.string().optional(),
      cleanup: z.coerce.boolean().default(false),
      cacheDataInGpu: z.coerce.boolean().default(false),
      indexAlgorithm: z.enum(["Auto", "Faiss", "KMeans"]).default("Auto"),
    })
    .safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const p = parsed.data;
  const allowedSr = p.vocoder === "RefineGAN" ? ["24000", "32000"] : ["32000", "40000", "48000"];
  if (!allowedSr.includes(p.sampleRate)) {
    return res
      .status(400)
      .json({ error: `sampleRate must be one of ${allowedSr.join(", ")} for ${p.vocoder}` });
  }
  if (p.customPretrained && (!p.gPretrainedPath || !p.dPretrainedPath)) {
    return res
      .status(400)
      .json({ error: "gPretrainedPath and dPretrainedPath are required with customPretrained." });
  }
  const gPre = p.gPretrainedPath || "";
  const dPre = p.dPretrainedPath || "";
  const job = startCliJob("train", { step: "train", ...p }, [
    "core.py",
    "train",
    "--model-name",
    p.modelName,
    "--vocoder",
    p.vocoder,
    ...(p.checkpointing ? ["--checkpointing"] : []),
    "--save-every-epoch",
    String(p.saveEveryEpoch),
    ...(p.saveOnlyLatest ? ["--save-only-latest"] : []),
    ...(p.saveEveryWeights ? ["--save-every-weights"] : []),
    "--total-epoch",
    String(p.totalEpoch),
    "--sample-rate",
    p.sampleRate,
    "--batch-size",
    String(p.batchSize),
    "--gpu",
    p.gpu,
    p.pretrained ? "--pretrained" : "--no-pretrained",
    ...(p.customPretrained
      ? ["--custom-pretrained", "--g-pretrained-path", gPre, "--d-pretrained-path", dPre]
      : []),
    ...(p.cleanup ? ["--cleanup"] : []),
    ...(p.cacheDataInGpu ? ["--cache-data-in-gpu"] : []),
    "--index-algorithm",
    p.indexAlgorithm,
  ]);
  return res.status(202).json({ jobId: job.id });
});

router.post("/index", (req: Request, res: Response) => {
  const parsed = z
    .object({
      modelName,
      indexAlgorithm: z.enum(["Auto", "Faiss", "KMeans"]).default("Auto"),
    })
    .safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const job = startCliJob("train", { step: "index", ...parsed.data }, [
    "core.py",
    "index",
    "--model-name",
    parsed.data.modelName,
    "--index-algorithm",
    parsed.data.indexAlgorithm,
  ]);
  return res.status(202).json({ jobId: job.id });
});

// {jobId} preferred; {modelName} falls back to the training pid file.
router.post("/stop", (req: Request, res: Response) => {
  const { jobId, modelName: m } = (req.body || {}) as { jobId?: string; modelName?: string };
  if (jobId && killJobTree(jobId)) return res.json({ ok: true, stopped: jobId });
  if (m) {
    try {
      const cfgPath = path.join(getRepoRoot(), "logs", path.basename(m), "config.json");
      const cfg = JSON.parse(fs.readFileSync(cfgPath, "utf-8")) as { process_pids?: unknown };
      const pids = Array.isArray(cfg.process_pids)
        ? cfg.process_pids.filter((p): p is number => typeof p === "number")
        : [];
      let killed = 0;
      for (const pid of pids) {
        try {
          if (process.platform === "win32") {
            spawn("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
          } else process.kill(pid, "SIGKILL");
          killed++;
        } catch {
          /* already dead */
        }
      }
      delete cfg.process_pids;
      fs.writeFileSync(cfgPath, JSON.stringify(cfg, null, 2));
      return res.json({ ok: true, killed });
    } catch (err) {
      return res.status(400).json({ error: errMsg(err) || "No pids found" });
    }
  }
  return res.status(404).json({ error: "No running job found (provide jobId or modelName)." });
});

export default router;
