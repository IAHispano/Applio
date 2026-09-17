import { z } from "zod";

// Mirrors core.py click options (_infer_opts + _post_process_opts).

export const F0_METHODS = [
  "crepe",
  "crepe-tiny",
  "rmvpe",
  "fcpe",
  "hybrid[crepe+rmvpe]",
  "hybrid[crepe+fcpe]",
  "hybrid[rmvpe+fcpe]",
  "hybrid[crepe+rmvpe+fcpe]",
] as const;

export const EXPORT_FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"] as const;

export const EMBEDDER_MODELS = [
  "contentvec",
  "spin",
  "spin-v2",
  "chinese-hubert-base",
  "japanese-hubert-base",
  "korean-hubert-base",
  "custom",
] as const;

export const inferenceParamsSchema = z.object({
  // Model selection (paths relative to repo root, e.g. logs/my-model/model.pth)
  pthPath: z.string().min(1, "pthPath is required"),
  indexPath: z.string().default(""),

  // Core inference options (core.py _infer_opts)
  pitch: z.coerce.number().int().min(-24).max(24).default(0),
  indexRate: z.coerce.number().min(0).max(1).default(0.75),
  volumeEnvelope: z.coerce.number().min(0).max(1).default(1),
  protect: z.coerce.number().min(0).max(0.5).default(0.33),
  f0Method: z.enum(F0_METHODS).default("rmvpe"),
  splitAudio: z.coerce.boolean().default(false),
  f0Autotune: z.coerce.boolean().default(false),
  f0AutotuneStrength: z.coerce.number().min(0).max(1).default(1),
  proposedPitch: z.coerce.boolean().default(false),
  proposedPitchThreshold: z.coerce.number().min(50).max(1199).default(155),
  cleanAudio: z.coerce.boolean().default(false),
  cleanStrength: z.coerce.number().min(0).max(1).default(0.7),
  exportFormat: z.enum(EXPORT_FORMATS).default("WAV"),
  embedderModel: z.enum(EMBEDDER_MODELS).default("contentvec"),
  embedderModelCustom: z.string().optional().default(""),
  sid: z.coerce.number().int().min(0).default(0),

  // Post-process (core.py _post_process_opts) — all optional, default off/neutral
  formantShifting: z.coerce.boolean().default(false),
  formantQfrency: z.coerce.number().default(1.0),
  formantTimbre: z.coerce.number().default(1.0),
  postProcess: z.coerce.boolean().default(false),
  reverb: z.coerce.boolean().default(false),
  reverbRoomSize: z.coerce.number().default(0.5),
  reverbDamping: z.coerce.number().default(0.5),
  reverbWetGain: z.coerce.number().default(0.5),
  reverbDryGain: z.coerce.number().default(0.5),
  reverbWidth: z.coerce.number().default(0.5),
  reverbFreezeMode: z.coerce.number().default(0.5),
  pitchShift: z.coerce.boolean().default(false),
  pitchShiftSemitones: z.coerce.number().default(0),
  limiter: z.coerce.boolean().default(false),
  limiterThreshold: z.coerce.number().default(-6),
  limiterReleaseTime: z.coerce.number().default(0.01),
  gain: z.coerce.boolean().default(false),
  gainDb: z.coerce.number().default(0),
  distortion: z.coerce.boolean().default(false),
  distortionGain: z.coerce.number().default(25),
  chorus: z.coerce.boolean().default(false),
  chorusRate: z.coerce.number().default(1.0),
  chorusDepth: z.coerce.number().default(0.25),
  chorusCenterDelay: z.coerce.number().default(7),
  chorusFeedback: z.coerce.number().default(0),
  chorusMix: z.coerce.number().default(0.5),
  bitcrush: z.coerce.boolean().default(false),
  bitcrushBitDepth: z.coerce.number().int().default(8),
  clipping: z.coerce.boolean().default(false),
  clippingThreshold: z.coerce.number().default(-6),
  compressor: z.coerce.boolean().default(false),
  compressorThreshold: z.coerce.number().default(0),
  compressorRatio: z.coerce.number().default(1),
  compressorAttack: z.coerce.number().default(1.0),
  compressorRelease: z.coerce.number().default(100),
  delay: z.coerce.boolean().default(false),
  delaySeconds: z.coerce.number().default(0.5),
  delayFeedback: z.coerce.number().default(0),
  delayMix: z.coerce.number().default(0.5),
});

export type InferenceParams = z.infer<typeof inferenceParamsSchema>;

export const batchInferenceSchema = inferenceParamsSchema.omit({}).extend({
  inputFolder: z.string().min(1),
  outputFolder: z.string().min(1),
});

export type BatchInferenceParams = z.infer<typeof batchInferenceSchema>;
