import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { z } from "zod";
import { errMsg } from "../errors";
import { getPythonBin, getRepoRoot, getUploadsDir } from "../python";

const router = Router();

// Native names of the locales shipped in assets/i18n/languages, ported from
// the Gradio settings (tabs/settings/sections/lang.py LANGUAGE_DISPLAY_NAMES)
// so the dropdown shows something friendlier than the raw code.
const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
  af_AF: "Afrikaans",
  am_AM: "አማርኛ",
  ar_AR: "العربية",
  az_AZ: "Azərbaycan",
  ba_BA: "Башҡортса",
  be_BE: "Беларуская",
  bn_BN: "বাংলা",
  bs_BS: "Bosanski",
  ca_CA: "Català",
  ceb_CEB: "Cebuano",
  cs_CS: "Čeština",
  de_DE: "Deutsch",
  el_EL: "Ελληνικά",
  en_US: "English",
  es_ES: "Español",
  eu_EU: "Euskara",
  fa_FA: "فارسی",
  fj_FJ: "Na Vosa Vakaviti",
  fr_FR: "Français",
  ga_GA: "Gaeilge",
  gu_GU: "ગુજરાતી",
  he_HE: "עברית",
  hi_IN: "हिन्दी",
  hr_HR: "Hrvatski",
  ht_HT: "Kreyòl Ayisyen",
  hu_HU: "Magyar",
  id_ID: "Bahasa Indonesia",
  it_IT: "Italiano",
  ja_JA: "日本語",
  jv_JV: "Basa Jawa",
  ko_KO: "한국어",
  lt_LT: "Lietuvių",
  lv_LV: "Latviešu",
  mg_MG: "Malagasy",
  ml_IN: "മലയാളം",
  mr_MR: "मराठी",
  ms_MS: "Bahasa Melayu",
  mt_MT: "Malti",
  nl_NL: "Nederlands",
  otq_OTQ: "Hñähñu",
  pa_PA: "ਪੰਜਾਬੀ",
  pl_PL: "Polski",
  pt_BR: "Português (Brasil)",
  pt_PT: "Português (Portugal)",
  ro_RO: "Română",
  ru_RU: "Русский",
  sk_SK: "Slovenčina",
  sm_SM: "Gagana Sāmoa",
  sr_RS: "Српски",
  sw_SW: "Kiswahili",
  ta_IN: "தமிழ்",
  te_TE: "తెలుగు",
  th_TH: "ไทย",
  to_TO: "Lea faka-Tonga",
  tr_TR: "Türkçe",
  uk_UK: "Українська",
  ur_UR: "اردو",
  vi_VI: "Tiếng Việt",
  wu_WU: "吴语",
  zh_CN: "简体中文",
};

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
  theme: z.object({ file: z.string() }).optional(),
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
    const sorted = codes.sort();
    res.json({
      languages: sorted,
      named: sorted.map((code) => ({ code, name: LANGUAGE_DISPLAY_NAMES[code] || code })),
      selected: loadConfig().lang,
    });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

// Resolved UI language + its translation dictionary (Gradio I18nAuto parity:
// explicit override wins, otherwise the OS locale, otherwise English which is
// the source language so an empty dict falls back to the key itself).
router.get("/language", (_req: Request, res: Response) => {
  try {
    const cfg = loadConfig() as { lang?: { override?: boolean; selected_lang?: string } };
    let code = "en_US";
    if (cfg.lang?.override && cfg.lang.selected_lang) {
      code = cfg.lang.selected_lang;
    } else {
      try {
        code = new Intl.DateTimeFormat().resolvedOptions().locale.replace("-", "_");
      } catch {
        code = "en_US";
      }
    }
    const dir = path.join(getRepoRoot(), "assets", "i18n", "languages");
    let file = path.join(dir, `${code}.json`);
    if (!fs.existsSync(file)) {
      const prefix = code.split("_")[0];
      const alt = fs.existsSync(dir)
        ? fs.readdirSync(dir).find((f) => f.startsWith(`${prefix}_`))
        : undefined;
      if (alt) {
        code = alt.replace(/\.json$/, "");
        file = path.join(dir, alt);
      } else {
        code = "en_US";
        file = path.join(dir, "en_US.json");
      }
    }
    const dict = fs.existsSync(file)
      ? (JSON.parse(fs.readFileSync(file, "utf-8")) as Record<string, string>)
      : {};
    res.json({ code, dict });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

// Discord rich presence. The Gradio app started RPCManager in-process at boot;
// here the API owns it instead: a detached python interpreter holds the
// pypresence connection open for as long as it lives (set_activity persists
// server-side until the connection closes).
let presenceProc: ChildProcess | null = null;
function presencePidFile(): string {
  const dir = getUploadsDir();
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "discord_presence.pid");
}
function presenceAlive(): boolean {
  if (presenceProc && presenceProc.exitCode === null) return true;
  try {
    const pid = Number(fs.readFileSync(presencePidFile(), "utf-8"));
    if (pid > 0) {
      process.kill(pid, 0);
      return true;
    }
  } catch {
    /* no pid file or stale */
  }
  return false;
}
export function startPresence(): boolean {
  if (presenceAlive()) return true;
  try {
    const code = [
      "from assets.discord_presence import RPCManager",
      "RPCManager.start_presence()",
      "import threading; threading.Event().wait()",
    ].join("; ");
    presenceProc = spawn(getPythonBin(), ["-c", code], {
      cwd: getRepoRoot(),
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    presenceProc.unref();
    fs.writeFileSync(presencePidFile(), String(presenceProc.pid || ""));
    return true;
  } catch {
    presenceProc = null;
    return false;
  }
}
export function stopPresence(): void {
  try {
    const pid = Number(fs.readFileSync(presencePidFile(), "utf-8"));
    if (pid > 0) {
      try {
        process.kill(pid);
      } catch {
        /* already gone */
      }
    }
  } catch {
    /* noop */
  }
  try {
    presenceProc?.kill();
  } catch {
    /* noop */
  }
  presenceProc = null;
  fs.rmSync(presencePidFile(), { force: true });
}
export function autoStartPresence(): void {
  try {
    const cfg = loadConfig() as { discord_presence?: boolean };
    if (cfg.discord_presence) startPresence();
  } catch {
    /* presence is best-effort */
  }
}
router.get("/presence", (_req: Request, res: Response) => {
  res.json({ running: presenceAlive() });
});
router.post("/presence", (req: Request, res: Response) => {
  const parsed = z.object({ enabled: z.boolean() }).safeParse(req.body);
  if (!parsed.success)
    return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
  try {
    const cfg = loadConfig();
    deepMerge(cfg, { discord_presence: parsed.data.enabled });
    saveConfig(cfg);
    if (parsed.data.enabled) {
      if (!startPresence())
        return res
          .status(502)
          .json({ error: "Could not start Discord presence (is Discord running? pypresence installed?)" });
    } else {
      stopPresence();
    }
    res.json({ ok: true, running: presenceAlive() });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

// Restart the API process (Gradio restart_applio parity). Under `dev` the
// file watcher respawns it; in production a supervisor (systemd/docker/pm2)
// must do the same — otherwise this is a plain shutdown.
router.post("/restart", (_req: Request, res: Response) => {
  res.json({
    ok: true,
    message:
      "API is restarting. If it does not come back, restart it manually (dev watcher, systemd, docker or pm2).",
  });
  setTimeout(() => process.exit(0), 500).unref?.();
});

// Theme system (Gradio themes parity, ours is file-based: assets/themes/*.json).
// The built-in default lives in globals.css :root; selecting "" restores it.
function themesDir(): string {
  const dir = path.join(getRepoRoot(), "assets", "themes");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

interface ThemeListEntry {
  id: string;
  name: string;
  description: string;
  example: boolean;
}

router.get("/themes", (_req: Request, res: Response) => {
  try {
    const dir = themesDir();
    const entries: ThemeListEntry[] = fs
      .readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .map((f) => {
        try {
          const raw = JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8")) as {
            name?: string;
            description?: string;
          };
          return {
            id: f,
            name: typeof raw.name === "string" && raw.name ? raw.name : f.replace(/\.json$/, ""),
            description: typeof raw.description === "string" ? raw.description : "",
            example: f.endsWith(".example.json"),
          };
        } catch {
          return { id: f, name: f, description: "Invalid JSON — fix or remove this file.", example: false };
        }
      });
    res.json({
      themes: entries,
      selected: (loadConfig().theme as { file?: string } | undefined)?.file || "",
    });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

const BUILTIN_THEME_FALLBACK = { name: "Default", version: "1.0.0", colors: {}, fonts: {}, radius: {} };

router.get("/theme", (req: Request, res: Response) => {
  try {
    const file = String(req.query.file || (loadConfig().theme as { file?: string } | undefined)?.file || "");
    // Legacy Gradio values (e.g. "Applio.py") and "" both mean the built-in default.
    if (!file?.endsWith(".json")) return res.json({ id: "", theme: BUILTIN_THEME_FALLBACK });
    // Confine to the themes dir (no traversal).
    const abs = path.resolve(themesDir(), path.basename(file));
    if (!abs.startsWith(themesDir() + path.sep) || !abs.endsWith(".json")) {
      return res.status(403).json({ error: "Only assets/themes/*.json files." });
    }
    if (!fs.existsSync(abs)) return res.status(404).json({ error: `Theme not found: ${file}` });
    res.json({ id: path.basename(file), theme: JSON.parse(fs.readFileSync(abs, "utf-8")) });
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
