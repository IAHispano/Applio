import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { z } from "zod";
import { startCliJob } from "../cli";
import { errMsg } from "../errors";
import { getOutputsDir, getRepoRoot, getUploadsDir, resolveUserPath } from "../python";
import { EMBEDDER_MODELS, EXPORT_FORMATS } from "../schemas";

const router = Router();

const upload = multer({
  dest: getUploadsDir(),
  limits: { fileSize: 5 * 1024 * 1024 }, // txt files are small
  fileFilter: (_req, file, cb) => {
    if (!file.originalname.toLowerCase().endsWith(".txt")) {
      return cb(new Error("Only .txt files are accepted"));
    }
    cb(null, true);
  },
});

interface TtsVoiceRaw {
  ShortName: string;
  FriendlyName?: string;
  Gender?: string;
  Locale?: string;
}

interface TtsVoice {
  shortName: string;
  friendlyName: string;
  gender: string;
  locale: string;
}

router.get("/voices", (_req: Request, res: Response) => {
  try {
    const raw = JSON.parse(
      fs.readFileSync(path.join(getRepoRoot(), "rvc", "lib", "tools", "tts_voices.json"), "utf-8"),
    ) as TtsVoiceRaw[];
    const voices: TtsVoice[] = raw.map((v) => ({
      shortName: v.ShortName,
      friendlyName: v.FriendlyName || v.ShortName,
      gender: v.Gender || "",
      locale: v.Locale || "",
    }));
    res.json({ voices });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) || "Could not load voices" });
  }
});

const ttsSchema = z.object({
  ttsText: z.string().default(""),
  ttsVoice: z.string().min(1),
  ttsRate: z.coerce.number().int().min(-100).max(100).default(0),
  pthPath: z.string().min(1),
  indexPath: z.string().default(""),
  pitch: z.coerce.number().int().min(-24).max(24).default(0),
  indexRate: z.coerce.number().min(0).max(1).default(0.75),
  volumeEnvelope: z.coerce.number().min(0).max(1).default(1),
  protect: z.coerce.number().min(0).max(0.5).default(0.5),
  f0Method: z.enum(["crepe", "crepe-tiny", "rmvpe", "fcpe"]).default("rmvpe"),
  splitAudio: z.coerce.boolean().default(false),
  f0Autotune: z.coerce.boolean().default(false),
  f0AutotuneStrength: z.coerce.number().min(0).max(1).default(1),
  proposedPitch: z.coerce.boolean().default(false),
  proposedPitchThreshold: z.coerce.number().min(50).max(1200).default(155),
  cleanAudio: z.coerce.boolean().default(false),
  cleanStrength: z.coerce.number().min(0).max(1).default(0.5),
  exportFormat: z.enum(EXPORT_FORMATS).default("WAV"),
  embedderModel: z.enum(EMBEDDER_MODELS).default("contentvec"),
  embedderModelCustom: z.string().optional().default(""),
  sid: z.coerce.number().int().min(0).default(0),
});

router.post("/", upload.single("txt_file"), (req: Request, res: Response) => {
  try {
    const body = { ...(req.body as Record<string, unknown>) };
    let ttsFile = "";
    if (req.file) {
      const text = fs.readFileSync(req.file.path, "utf-8"); // validates UTF-8 like process_input
      const dest = path.join(getUploadsDir(), `tts_input_${Date.now()}.txt`);
      fs.writeFileSync(dest, text, "utf-8");
      fs.rmSync(req.file.path, { force: true });
      ttsFile = dest;
    }
    const parsed = ttsSchema.safeParse(body);
    if (!parsed.success) {
      return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
    }
    const p = parsed.data;
    if (!p.ttsText && !ttsFile) {
      return res.status(400).json({ error: "Provide 'ttsText' or upload a 'txt_file'." });
    }
    const pthAbs = resolveUserPath(p.pthPath);
    if (!fs.existsSync(pthAbs)) return res.status(400).json({ error: `Model not found: ${p.pthPath}` });

    const ts = Date.now();
    const outTts = path.join(getOutputsDir(), `tts_output_${ts}.wav`);
    const outRvc = path.join(getOutputsDir(), `tts_rvc_output_${ts}.wav`);
    const args = [
      "core.py",
      "tts",
      "--tts-file",
      ttsFile,
      "--tts-text",
      p.ttsText,
      "--tts-voice",
      p.ttsVoice,
      "--tts-rate",
      String(p.ttsRate),
      "--output-tts-path",
      outTts,
      "--output-rvc-path",
      outRvc,
      "--pth-path",
      p.pthPath,
      "--index-path",
      p.indexPath || "",
      "--pitch",
      String(p.pitch),
      "--index-rate",
      String(p.indexRate),
      "--volume-envelope",
      String(p.volumeEnvelope),
      "--protect",
      String(p.protect),
      "--f0-method",
      p.f0Method,
      "--export-format",
      p.exportFormat,
      "--embedder-model",
      p.embedderModel,
      "--sid",
      String(p.sid),
      ...(p.splitAudio ? ["--split-audio"] : []),
      ...(p.f0Autotune ? ["--f0-autotune"] : []),
      "--f0-autotune-strength",
      String(p.f0AutotuneStrength),
      ...(p.proposedPitch ? ["--proposed-pitch"] : []),
      "--proposed-pitch-threshold",
      String(p.proposedPitchThreshold),
      ...(p.cleanAudio ? ["--clean-audio"] : []),
      "--clean-strength",
      String(p.cleanStrength),
    ];
    if (p.embedderModel === "custom" && p.embedderModelCustom) {
      args.push("--embedder-model-custom", p.embedderModelCustom);
    }
    const ext = p.exportFormat.toLowerCase();
    const job = startCliJob("tts", { ...p, ttsFile }, args, {
      parse: () => {
        const finalAbs = outRvc.replace(/\.wav$/i, `.${ext}`);
        const served = fs.existsSync(finalAbs) ? finalAbs : outRvc;
        if (!fs.existsSync(served)) throw new Error("TTS finished but no output file was found.");
        return {
          result: { ttsIntermediate: `assets/audios/${path.basename(outTts)}` },
          outputFile: path.relative(getRepoRoot(), served).replace(/\\/g, "/"),
        };
      },
    });
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    return res.status(500).json({ error: errMsg(err) || "TTS failed to start" });
  }
});

export default router;
