import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { z } from "zod";
import { runPythonJson } from "../cli";
import { errMsg } from "../errors";
import { getRepoRoot, resolveUserPath } from "../python";

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
  const audios = walk(audiosDir, [
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
  ])
    .map(toRepoRelative)
    .sort();

  res.json({ models, indexes, audios, root });
});

export interface ModelDetail {
  id: string;
  name: string;
  pthPath: string;
  pthSize: number;
  indexPath: string | null;
  indexSize: number | null;
  modifiedAt: string;
  folder: string;
}

router.get("/library", (_req: Request, res: Response) => {
  try {
    const root = getRepoRoot();
    const logsDir = path.join(root, "logs");
    if (!fs.existsSync(logsDir)) {
      return res.json({ models: [] });
    }

    const allPths = walk(logsDir, [".pth", ".onnx"]).filter(
      (f) => !path.basename(f).startsWith("G_") && !path.basename(f).startsWith("D_"),
    );
    const allIndexes = walk(logsDir, [".index"]).filter((f) => !path.basename(f).includes("trained"));

    const result: ModelDetail[] = [];

    for (const pth of allPths) {
      const pthRel = toRepoRelative(pth);
      const stat = fs.statSync(pth);
      const stem = path.basename(pth).replace(/\.(pth|onnx)$/i, "");
      const folder = path.relative(logsDir, path.dirname(pth)).replace(/\\/g, "/");

      // Match index in same folder or with similar name
      const matchedIdx = allIndexes.find((idx) => {
        const idxDir = path.dirname(idx);
        if (idxDir === path.dirname(pth)) return true;
        const idxStem = path.basename(idx).replace(/\.index$/i, "");
        return (
          idxStem.toLowerCase().includes(stem.toLowerCase()) ||
          stem.toLowerCase().includes(idxStem.toLowerCase())
        );
      });

      let idxSize: number | null = null;
      let idxRel: string | null = null;
      if (matchedIdx && fs.existsSync(matchedIdx)) {
        idxSize = fs.statSync(matchedIdx).size;
        idxRel = toRepoRelative(matchedIdx);
      }

      result.push({
        id: pthRel,
        name: stem,
        pthPath: pthRel,
        pthSize: stat.size,
        indexPath: idxRel,
        indexSize: idxSize,
        modifiedAt: stat.mtime.toISOString(),
        folder: folder || "root",
      });
    }

    result.sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));
    return res.json({ models: result });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) });
  }
});

router.post("/inspect", async (req: Request, res: Response) => {
  try {
    const parsed = z.object({ pthPath: z.string().min(1) }).safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: "pthPath is required" });
    }
    const abs = resolveUserPath(parsed.data.pthPath);
    if (!fs.existsSync(abs)) {
      return res.status(404).json({ error: `File not found: ${parsed.data.pthPath}` });
    }

    const pyCode = [
      "import json, torch",
      `data = torch.load(${JSON.stringify(abs)}, map_location='cpu', weights_only=True)`,
      "meta = {",
      "  'model_name': str(data.get('model_name', 'None')),",
      "  'author': str(data.get('author', 'None')),",
      "  'epochs': str(data.get('epoch', 'None')),",
      "  'step': str(data.get('step', 'None')),",
      "  'sr': str(data.get('sr', 'None')),",
      "  'f0': str(data.get('f0', 'None')),",
      "  'vocoder': str(data.get('vocoder', 'None')),",
      "  'embedder_model': str(data.get('embedder_model', 'None')),",
      "  'creation_date': str(data.get('creation_date', 'None')),",
      "  'model_hash': str(data.get('model_hash', 'None')),",
      "}",
      "print('APPLIO_JSON:' + json.dumps(meta))",
    ].join("; ");

    const meta = await runPythonJson<Record<string, string>>(pyCode);
    return res.json({ ok: true, metadata: meta });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) || "Inspection failed" });
  }
});

router.delete("/:name", (req: Request, res: Response) => {
  try {
    const name = decodeURIComponent(req.params.name);
    const root = getRepoRoot();
    const logsDir = path.join(root, "logs");

    let targetPath = path.resolve(logsDir, name);
    if (!targetPath.startsWith(logsDir)) {
      return res.status(400).json({ error: "Invalid model path: escapes logs directory" });
    }

    if (!fs.existsSync(targetPath)) {
      // Check if name is a file within logs
      const candidateFile = path.resolve(root, name);
      if (candidateFile.startsWith(logsDir) && fs.existsSync(candidateFile)) {
        targetPath = candidateFile;
      } else {
        return res.status(404).json({ error: `Model not found: ${name}` });
      }
    }

    const stat = fs.statSync(targetPath);
    if (stat.isDirectory()) {
      fs.rmSync(targetPath, { recursive: true, force: true });
    } else {
      fs.rmSync(targetPath, { force: true });
      // If parent dir is now empty, clean it
      const parent = path.dirname(targetPath);
      if (parent !== logsDir && fs.existsSync(parent) && fs.readdirSync(parent).length === 0) {
        fs.rmdirSync(parent);
      }
    }

    return res.json({ ok: true, message: `Deleted ${name}` });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) });
  }
});

// Speaker IDs for multi-speaker models (Gradio get_speakers_id parity:
// torch.load(pth)["speakers_id"] -> range(n), else [0]).
router.get("/speakers", async (req: Request, res: Response) => {
  try {
    const pthPath = String(req.query.pthPath || "");
    if (!pthPath) return res.status(400).json({ error: "Provide 'pthPath'." });
    const abs = resolveUserPath(pthPath);
    if (!fs.existsSync(abs)) return res.status(404).json({ error: `Model not found: ${pthPath}` });
    const code = [
      "import json, torch",
      `ckpt = torch.load(${JSON.stringify(abs)}, map_location='cpu')`,
      "n = ckpt.get('speakers_id', 0) if isinstance(ckpt, dict) else 0",
      "print('APPLIO_JSON:' + json.dumps({'speakers': list(range(n)) if n else [0]}))",
    ].join("; ");
    const out = await runPythonJson<{ speakers: number[] }>(code);
    res.json(out);
  } catch (err) {
    res.status(500).json({ error: errMsg(err) || "Could not read speakers" });
  }
});

export default router;
