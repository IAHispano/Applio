import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { z } from "zod";
import { errMsg } from "../errors";
import { getRepoRoot } from "../python";

const router = Router();

type JsonObject = Record<string, unknown>;

function configPath(): string {
  return path.join(getRepoRoot(), "assets", "config.json");
}
function templatePath(): string {
  return path.join(getRepoRoot(), "assets", "config_template.json");
}

function loadConfig(): JsonObject {
  const tpl = JSON.parse(fs.readFileSync(templatePath(), "utf-8")) as JsonObject;
  if (!fs.existsSync(configPath())) {
    fs.writeFileSync(configPath(), JSON.stringify(tpl, null, 2));
    return tpl;
  }
  const cfg = JSON.parse(fs.readFileSync(configPath(), "utf-8")) as JsonObject;
  return deepMerge(structuredClone(tpl), cfg);
}
function deepMerge(base: JsonObject, over: JsonObject): JsonObject {
  for (const k of Object.keys(over)) {
    const bv = base[k];
    const ov = over[k];
    if (ov && typeof ov === "object" && !Array.isArray(ov) && bv && typeof bv === "object") {
      deepMerge(bv as JsonObject, ov as JsonObject);
    } else base[k] = ov;
  }
  return base;
}
function saveConfig(cfg: JsonObject) {
  fs.writeFileSync(configPath(), JSON.stringify(cfg, null, 2));
}

const settingsSchema = z.object({
  model_index_filter: z.boolean().optional(),
  discord_presence: z.boolean().optional(),
  lang: z.object({ override: z.boolean(), selected_lang: z.string().min(1) }).optional(),
  model_author: z.string().nullable().optional(),
  precision: z.enum(["fp32", "fp16", "bf16"]).optional(),
  rmvpe_high_register: z
    .object({
      enabled: z.boolean(),
      mode: z.enum(["true_pitch", "fold"]),
      f0_ceil: z.number().min(1000).max(2000),
    })
    .optional(),
  realtime: z.record(z.unknown()).optional(),
});

router.get("/", (_req: Request, res: Response) => {
  try {
    res.json({ config: loadConfig() });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

router.put("/", (req: Request, res: Response) => {
  const parsed = settingsSchema.safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid settings", details: parsed.error.flatten() });
  try {
    const cfg = loadConfig();
    deepMerge(cfg, parsed.data);
    saveConfig(cfg);
    res.json({ ok: true, config: cfg });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

router.get("/languages", (_req: Request, res: Response) => {
  try {
    const dir = path.join(getRepoRoot(), "assets", "i18n", "languages");
    const codes = fs.existsSync(dir)
      ? fs
          .readdirSync(dir)
          .filter((f) => f.endsWith(".json"))
          .map((f) => f.replace(/\.json$/, ""))
      : ["en_US"];
    res.json({ languages: codes.sort(), selected: loadConfig().lang });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

router.get("/version-check", async (_req: Request, res: Response) => {
  try {
    const local = loadConfig().version || "unknown";
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 8000);
    const r = await fetch("https://api.github.com/repos/IAHispano/Applio/releases/latest", {
      signal: ctrl.signal,
    });
    clearTimeout(t);
    if (!r.ok) throw new Error(`GitHub API ${r.status}`);
    const latest = ((await r.json()) as { tag_name: string }).tag_name;
    const cmp = (a: string, b: string) => {
      const pa = a.replace(/^v/, "").split(".").map(Number);
      const pb = b.replace(/^v/, "").split(".").map(Number);
      for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        const d = (pa[i] || 0) - (pb[i] || 0);
        if (d !== 0) return d > 0 ? 1 : -1;
      }
      return 0;
    };
    const c = cmp(String(local), String(latest));
    res.json({ local, latest, status: c === 0 ? "up-to-date" : c < 0 ? "behind" : "ahead" });
  } catch (err) {
    res.status(502).json({ error: errMsg(err) || "Version check failed (offline?)" });
  }
});

export default router;
