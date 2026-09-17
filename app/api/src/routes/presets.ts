import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { z } from "zod";
import { errMsg } from "../errors";
import { getRepoRoot } from "../python";

const router = Router();

function presetsDir(): string {
  return path.join(getRepoRoot(), "assets", "presets");
}
function formantDir(): string {
  return path.join(getRepoRoot(), "assets", "formant_shift");
}

function safeName(name: string): string {
  const base = path.basename(name).replace(/\.json$/i, "");
  if (!base || base === "." || base === "..") throw new Error("Invalid preset name");
  if (/[/\\]/.test(name)) throw new Error("Invalid preset name");
  return `${base}.json`;
}

const presetSchema = z.object({
  pitch: z.number().int().min(-24).max(24),
  index_rate: z.number().min(0).max(1),
  rms_mix_rate: z.number().min(0).max(1),
  protect: z.number().min(0).max(0.5),
});

router.get("/", (_req: Request, res: Response) => {
  const dir = presetsDir();
  if (!fs.existsSync(dir)) return res.json({ presets: [] });
  const presets = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      try {
        return {
          name: f.replace(/\.json$/i, ""),
          values: JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8")),
        };
      } catch {
        return { name: f.replace(/\.json$/i, ""), values: null };
      }
    });
  res.json({ presets });
});

router.post("/", (req: Request, res: Response) => {
  try {
    const { name, values } = req.body as { name: string; values: unknown };
    const file = safeName(String(name || ""));
    const parsed = presetSchema.parse(values);
    fs.mkdirSync(presetsDir(), { recursive: true });
    fs.writeFileSync(path.join(presetsDir(), file), JSON.stringify(parsed, null, 4));
    res.json({ ok: true, name: file.replace(/\.json$/i, "") });
  } catch (err) {
    res.status(400).json({ error: errMsg(err) || "Invalid preset" });
  }
});

router.delete("/:name", (req: Request, res: Response) => {
  try {
    const file = safeName(req.params.name);
    fs.rmSync(path.join(presetsDir(), file), { force: true });
    res.json({ ok: true });
  } catch (err) {
    res.status(400).json({ error: errMsg(err) });
  }
});

router.get("/formant", (_req: Request, res: Response) => {
  const dir = formantDir();
  if (!fs.existsSync(dir)) return res.json({ presets: [] });
  res.json({ presets: fs.readdirSync(dir).filter((f) => f.endsWith(".json")) });
});

export default router;
