import fs from "node:fs";
import { type Request, type Response, Router } from "express";
import { startCliJob } from "../cli";
import { errMsg } from "../errors";
import { resolveUserPath } from "../python";
import { type BatchInferenceParams, batchInferenceSchema } from "../schemas";

const router = Router();

router.post("/", (req: Request, res: Response) => {
  try {
    const parsed = batchInferenceSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: "Invalid params", details: parsed.error.flatten() });
    }
    const p: BatchInferenceParams = parsed.data;
    const inputFolder = resolveUserPath(p.inputFolder);
    const outputFolder = resolveUserPath(p.outputFolder);
    if (!fs.existsSync(inputFolder))
      return res.status(400).json({ error: `Input folder not found: ${p.inputFolder}` });
    fs.mkdirSync(outputFolder, { recursive: true });
    const pthAbs = resolveUserPath(p.pthPath);
    if (!fs.existsSync(pthAbs)) return res.status(400).json({ error: `Model not found: ${p.pthPath}` });

    const flag = (name: string, v: boolean) => (v ? [`--${name}`] : []);
    const args = [
      "core.py",
      "batch-infer",
      "--input-folder",
      inputFolder,
      "--output-folder",
      outputFolder,
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
    const job = startCliJob("batch-inference", p, args);
    return res.status(202).json({ jobId: job.id });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) || "Batch inference failed to start" });
  }
});

export default router;
