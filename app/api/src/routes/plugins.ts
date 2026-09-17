import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { startCliJob } from "../cli";
import { errMsg } from "../errors";
import { getRepoRoot, getUploadsDir } from "../python";

const router = Router();
const upload = multer({
  dest: getUploadsDir(),
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.originalname.toLowerCase().endsWith(".zip"))
      return cb(new Error("Only .zip plugin packages are accepted"));
    cb(null, true);
  },
});

function installedDir(): string {
  const dir = path.join(getRepoRoot(), "plugins", "installed");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}
function configPath(): string {
  return path.join(getRepoRoot(), "assets", "config.json");
}
function enabledPlugins(): string[] {
  try {
    const cfg = JSON.parse(fs.readFileSync(configPath(), "utf-8"));
    return Array.isArray(cfg.plugins) ? cfg.plugins : [];
  } catch {
    return [];
  }
}
function saveEnabled(list: string[]) {
  const cfg = fs.existsSync(configPath()) ? JSON.parse(fs.readFileSync(configPath(), "utf-8")) : {};
  cfg.plugins = list;
  fs.writeFileSync(configPath(), JSON.stringify(cfg, null, 2));
}

router.get("/", (_req: Request, res: Response) => {
  const dir = installedDir();
  const folders = fs.readdirSync(dir).filter((e) => fs.statSync(path.join(dir, e)).isDirectory());
  const enabled = new Set(enabledPlugins());
  res.json({
    plugins: folders.map((f) => ({
      name: f,
      enabled: enabled.has(f),
      hasEntrypoint: fs.existsSync(path.join(dir, f, "plugin.py")),
    })),
  });
});

router.post("/install", upload.single("file"), (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ error: "Upload a 'file' (.zip)." });
  const zipPath = req.file.path;
  const folder = path.basename(req.file.originalname, ".zip").replace(/[^a-zA-Z0-9._-]/g, "_") || "plugin";
  const dest = path.join(installedDir(), folder);
  try {
    fs.mkdirSync(dest, { recursive: true });
    // Extract with stdlib zipfile so no `unzip` binary is needed.
    const code = [
      "import zipfile, sys",
      `zipfile.ZipFile(${JSON.stringify(zipPath)}).extractall(${JSON.stringify(dest)})`,
      "print('extracted')",
    ].join("; ");
    const job = startCliJob("other", { plugin: folder }, ["-c", code], {
      parse: () => {
        fs.rmSync(zipPath, { force: true });
        const req_file = path.join(dest, "requirements.txt");
        if (fs.existsSync(req_file)) {
          startCliJob("other", { plugin: folder, step: "pip-install" }, [
            "-m",
            "pip",
            "install",
            "-r",
            req_file,
          ]);
        }
        const enabled = new Set(enabledPlugins());
        enabled.add(folder);
        saveEnabled([...enabled]);
        return { result: { name: folder, requiresRestart: true } };
      },
    });
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    fs.rmSync(zipPath, { force: true });
    return res.status(500).json({ error: errMsg(err) });
  }
});

router.post("/toggle", (req: Request, res: Response) => {
  const parsed = z.object({ name: z.string().min(1), enabled: z.boolean() }).safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  const name = path.basename(parsed.data.name);
  if (!fs.existsSync(path.join(installedDir(), name)))
    return res.status(404).json({ error: "Plugin not installed" });
  const enabled = new Set(enabledPlugins());
  if (parsed.data.enabled) enabled.add(name);
  else enabled.delete(name);
  saveEnabled([...enabled]);
  res.json({ ok: true, requiresRestart: true });
});

export default router;
