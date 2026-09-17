import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { getRepoRoot } from "../python";

const router = Router();

function walk(dir: string, exts: string[], out: string[] = []): string[] {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, exts, out);
    else if (exts.some((e) => entry.name.endsWith(e))) out.push(full);
  }
  return out;
}

function toRepoRelative(abs: string): string {
  return path.relative(getRepoRoot(), abs).replace(/\\/g, "/");
}

router.get("/", (_req: Request, res: Response) => {
  const root = getRepoRoot();
  const logsDir = path.join(root, "logs");
  const audiosDir = path.join(root, "assets", "audios");

  const models = walk(logsDir, [".pth", ".onnx"])
    .filter((f) => !path.basename(f).startsWith("G_") && !path.basename(f).startsWith("D_"))
    .map(toRepoRelative)
    .sort();
  const indexes = walk(logsDir, [".index"])
    .filter((f) => !path.basename(f).includes("trained"))
    .map(toRepoRelative)
    .sort();
  const audios = walk(audiosDir, [".wav", ".mp3", ".flac", ".ogg", ".m4a"]).map(toRepoRelative).sort();

  res.json({ models, indexes, audios, root });
});

export default router;
