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

    // Robust checkpoint reader: community .pth files vary widely (legacy
    // pickles, bare state_dicts, missing metadata keys). Try the safe loader
    // first, then unrestricted load for trusted local files, then a tolerant
    // load that substitutes inert placeholders for classes that no longer
    // exist in this environment (so metadata stays readable).
    const pyCode = [
      "import io as _io",
      "import json",
      "import pickle as _pickle",
      "import torch",
      `path = ${JSON.stringify(abs)}`,
      "class _TolerantUnpickler(_pickle.Unpickler):",
      "    def find_class(self, module, name):",
      "        try:",
      "            return super().find_class(module, name)",
      "        except Exception:",
      "            return type(name, (), {})",
      "class _TolerantPickle:",
      "    Unpickler = _TolerantUnpickler",
      "    load = staticmethod(lambda f, **kw: _TolerantUnpickler(f, **kw).load())",
      "    loads = staticmethod(lambda s, **kw: _TolerantUnpickler(_io.BytesIO(s), **kw).load())",
      "data = None",
      "load_error = None",
      "try:",
      "    try:",
      "        data = torch.load(path, map_location='cpu', weights_only=True)",
      "    except TypeError:",
      "        data = torch.load(path, map_location='cpu')",
      "    except Exception:",
      "        data = torch.load(path, map_location='cpu', weights_only=False)",
      "except Exception:",
      "    try:",
      "        data = torch.load(path, map_location='cpu', weights_only=False, pickle_module=_TolerantPickle)",
      "    except TypeError:",
      "        try:",
      "            data = torch.load(path, map_location='cpu')",
      "        except Exception as e:",
      "            load_error = str(e)[:1000]",
      "            data = None",
      "    except Exception as e:",
      "        load_error = str(e)[:1000]",
      "        data = None",
      "if data is None:",
      "    raise SystemExit('LOAD_FAILED:' + (load_error or 'unknown error'))",
      "d = data if isinstance(data, dict) else {}",
      "def _g(k, default='None'):",
      "    try:",
      "        v = d.get(k, default)",
      "    except Exception:",
      "        return default",
      "    if v is None:",
      "        return default",
      "    try:",
      "        s = str(v)",
      "        return s if s else default",
      "    except Exception:",
      "        return default",
      "meta = {",
      "  'model_name': _g('model_name'),",
      "  'author': _g('author'),",
      "  'epochs': _g('epoch'),",
      "  'step': _g('step'),",
      "  'sr': _g('sr'),",
      "  'f0': _g('f0'),",
      "  'version': _g('version'),",
      "  'vocoder': _g('vocoder'),",
      "  'embedder_model': _g('embedder_model'),",
      "  'creation_date': _g('creation_date'),",
      "  'model_hash': _g('model_hash'),",
      "  'dataset_length': _g('dataset_length'),",
      "  'speakers_id': _g('speakers_id', '0'),",
      "}",
      "print('APPLIO_JSON:' + json.dumps(meta))",
    ].join("\n");

    const meta = await runPythonJson<Record<string, string>>(pyCode);
    return res.json({ ok: true, metadata: meta });
  } catch (err) {
    const msg = errMsg(err) || "Inspection failed";
    // Surface a helpful hint when torch itself cannot unpickle the file.
    if (msg.includes("LOAD_FAILED:")) {
      const detail = msg.split("LOAD_FAILED:")[1]?.trim() || "could not be parsed";
      return res.status(422).json({
        error: `Could not read checkpoint (file may be corrupted or not an RVC checkpoint): ${detail.slice(0, 500)}`,
      });
    }
    return res.status(500).json({ error: msg });
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
      "import io as _io",
      "import json",
      "import pickle as _pickle",
      "import torch",
      `path = ${JSON.stringify(abs)}`,
      "class _TolerantUnpickler(_pickle.Unpickler):",
      "    def find_class(self, module, name):",
      "        try:",
      "            return super().find_class(module, name)",
      "        except Exception:",
      "            return type(name, (), {})",
      "class _TolerantPickle:",
      "    Unpickler = _TolerantUnpickler",
      "    load = staticmethod(lambda f, **kw: _TolerantUnpickler(f, **kw).load())",
      "    loads = staticmethod(lambda s, **kw: _TolerantUnpickler(_io.BytesIO(s), **kw).load())",
      "try:",
      "    try:",
      "        ckpt = torch.load(path, map_location='cpu', weights_only=True)",
      "    except TypeError:",
      "        ckpt = torch.load(path, map_location='cpu')",
      "    except Exception:",
      "        ckpt = torch.load(path, map_location='cpu', weights_only=False)",
      "except Exception:",
      "    try:",
      "        ckpt = torch.load(path, map_location='cpu', weights_only=False, pickle_module=_TolerantPickle)",
      "    except Exception as e:",
      "        raise SystemExit('LOAD_FAILED:' + str(e)[:500])",
      "try:",
      "    n = ckpt.get('speakers_id', 0) if isinstance(ckpt, dict) else 0",
      "    n = int(n)",
      "except Exception:",
      "    n = 0",
      "print('APPLIO_JSON:' + json.dumps({'speakers': list(range(n)) if n else [0]}))",
    ].join("\n");
    const out = await runPythonJson<{ speakers: number[] }>(code);
    res.json(out);
  } catch (err) {
    const msg = errMsg(err) || "Could not read speakers";
    if (msg.includes("LOAD_FAILED:")) {
      return res
        .status(422)
        .json({ error: `Could not read checkpoint: ${msg.split("LOAD_FAILED:")[1]?.trim()?.slice(0, 300)}` });
    }
    res.status(500).json({ error: msg });
  }
});

export default router;
