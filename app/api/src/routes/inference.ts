import fs from "node:fs";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import multer from "multer";
import { trackPid } from "../cli";
import { errMsg } from "../errors";
import { appendLog, createJob, getJob, setDone, setError, setRunning } from "../jobs";
import { getOutputsDir, getRepoRoot, getUploadsDir, resolveUserPath, runPythonModule } from "../python";
import { type InferenceParams, inferenceParamsSchema } from "../schemas";

const router = Router();

const AUDIO_EXTS = new Set(
  ".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.alac,.wma,.aiff,.webm,.ac3".split(","),
);

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, getUploadsDir()),
  filename: (_req, file, cb) => {
    const safe = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${Date.now()}_${safe}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!AUDIO_EXTS.has(ext)) return cb(new Error(`Unsupported audio type: ${ext}`));
    cb(null, true);
  },
});

function toCliArgs(p: InferenceParams, inputPath: string, outputPath: string): string[] {
  const flag = (name: string, value: boolean) => (value ? [`--${name}`] : []);
  const args: string[] = [
    "core.py",
    "infer",
    "--input-path",
    inputPath,
    "--output-path",
    outputPath,
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
    ...flag("split-audio", p.splitAudio),
    ...flag("f0-autotune", p.f0Autotune),
    "--f0-autotune-strength",
    String(p.f0AutotuneStrength),
    ...flag("proposed-pitch", p.proposedPitch),
    "--proposed-pitch-threshold",
    String(p.proposedPitchThreshold),
    ...flag("clean-audio", p.cleanAudio),
    "--clean-strength",
    String(p.cleanStrength),
    ...flag("formant-shifting", p.formantShifting),
    "--formant-qfrency",
    String(p.formantQfrency),
    "--formant-timbre",
    String(p.formantTimbre),
    ...flag("post-process", p.postProcess),
    ...flag("reverb", p.reverb),
    "--reverb-room-size",
    String(p.reverbRoomSize),
    "--reverb-damping",
    String(p.reverbDamping),
    "--reverb-wet-gain",
    String(p.reverbWetGain),
    "--reverb-dry-gain",
    String(p.reverbDryGain),
    "--reverb-width",
    String(p.reverbWidth),
    "--reverb-freeze-mode",
    String(p.reverbFreezeMode),
    ...flag("pitch-shift", p.pitchShift),
    "--pitch-shift-semitones",
    String(p.pitchShiftSemitones),
    ...flag("limiter", p.limiter),
    "--limiter-threshold",
    String(p.limiterThreshold),
    "--limiter-release-time",
    String(p.limiterReleaseTime),
    ...flag("gain", p.gain),
    "--gain-db",
    String(p.gainDb),
    ...flag("distortion", p.distortion),
    "--distortion-gain",
    String(p.distortionGain),
    ...flag("chorus", p.chorus),
    "--chorus-rate",
    String(p.chorusRate),
    "--chorus-depth",
    String(p.chorusDepth),
    "--chorus-center-delay",
    String(p.chorusCenterDelay),
    "--chorus-feedback",
    String(p.chorusFeedback),
    "--chorus-mix",
    String(p.chorusMix),
    ...flag("bitcrush", p.bitcrush),
    "--bitcrush-bit-depth",
    String(p.bitcrushBitDepth),
    ...flag("clipping", p.clipping),
    "--clipping-threshold",
    String(p.clippingThreshold),
    ...flag("compressor", p.compressor),
    "--compressor-threshold",
    String(p.compressorThreshold),
    "--compressor-ratio",
    String(p.compressorRatio),
    "--compressor-attack",
    String(p.compressorAttack),
    "--compressor-release",
    String(p.compressorRelease),
    ...flag("delay", p.delay),
    "--delay-seconds",
    String(p.delaySeconds),
    "--delay-feedback",
    String(p.delayFeedback),
    "--delay-mix",
    String(p.delayMix),
  ];
  if (p.embedderModel === "custom" && p.embedderModelCustom) {
    args.push("--embedder-model-custom", p.embedderModelCustom);
  }
  return args;
}

router.post("/", upload.single("audio"), async (req: Request, res: Response) => {
  try {
    const body = { ...(req.body as Record<string, unknown>) };
    const parsed = inferenceParamsSchema.safeParse(body);
    if (!parsed.success) {
      if (req.file) fs.rmSync(req.file.path, { force: true });
      return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
    }
    const params = parsed.data;

    let inputAbs: string;
    if (req.file) {
      inputAbs = req.file.path;
    } else if (typeof body.inputPath === "string" && body.inputPath.length > 0) {
      inputAbs = resolveUserPath(body.inputPath);
    } else {
      return res.status(400).json({ error: "Provide an 'audio' file upload or 'inputPath'." });
    }

    const pthAbs = resolveUserPath(params.pthPath);
    if (!fs.existsSync(pthAbs)) {
      if (req.file) fs.rmSync(req.file.path, { force: true });
      return res.status(400).json({ error: `Model not found: ${params.pthPath}` });
    }
    if (params.indexPath) {
      const idxAbs = resolveUserPath(params.indexPath);
      if (!fs.existsSync(idxAbs)) {
        if (req.file) fs.rmSync(req.file.path, { force: true });
        return res.status(400).json({ error: `Index not found: ${params.indexPath}` });
      }
    }

    const job = createJob("inference", { ...params, inputPath: inputAbs });
    void runInferenceJob(job.id, params, inputAbs);
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    if (req.file) fs.rmSync(req.file.path, { force: true });
    return res.status(500).json({ error: errMsg(err) || "Inference failed to start" });
  }
});

async function runInferenceJob(jobId: string, params: InferenceParams, inputAbs: string) {
  const job = getJob(jobId);
  if (!job) return;
  setRunning(job);
  try {
    const ts = Date.now();
    const ext = String(params.exportFormat || "WAV").toLowerCase();
    const outAbs = path.join(getOutputsDir(), `web_output_${ts}.${ext === "m4a" ? "m4a" : ext}`);
    // core.py expects a .wav output path then renames by export format; give .wav stem
    const outWav = outAbs.replace(/\.[a-z0-9]+$/i, ".wav");
    const args = toCliArgs(params, inputAbs, outWav);
    const result = await runPythonModule(args, {
      onData: (chunk) => {
        const trimmed = chunk.trim().slice(0, 1000);
        if (trimmed) appendLog(job, trimmed);
      },
      onSpawn: (pid) => trackPid(job.id, pid),
    });
    trackPid(job.id, undefined);
    if (result.code !== 0) {
      throw new Error(result.stderr.slice(-3000) || `Inference failed with code ${result.code}`);
    }
    const finalAbs = outWav.replace(/\.wav$/i, `.${ext}`);
    const served = fs.existsSync(finalAbs) ? finalAbs : outWav;
    if (!fs.existsSync(served)) throw new Error("Inference finished but no output file was found.");
    const rel = path.relative(getRepoRoot(), served).replace(/\\/g, "/");
    appendLog(job, `Done -> ${rel}`);
    setDone(job, { stdout: result.stdout.slice(-2000) }, rel);
  } catch (err) {
    trackPid(job.id, undefined);
    appendLog(job, `ERROR: ${errMsg(err)}`);
    setError(job, errMsg(err) || "Inference failed");
  }
}

export default router;
