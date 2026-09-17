"use client";

import { Sliders, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { errMsg, fetchJob, fetchModels, type Job, pollJob, submitInference } from "../lib/api";
import AudioPlayer from "./AudioPlayer";

const F0_METHODS = [
  "rmvpe",
  "fcpe",
  "crepe",
  "crepe-tiny",
  "hybrid[crepe+rmvpe]",
  "hybrid[crepe+fcpe]",
  "hybrid[rmvpe+fcpe]",
  "hybrid[crepe+rmvpe+fcpe]",
];

const EMBEDDERS = [
  "contentvec",
  "spin",
  "spin-v2",
  "chinese-hubert-base",
  "japanese-hubert-base",
  "korean-hubert-base",
  "custom",
];
const FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"];

export default function InferenceForm() {
  const [models, setModels] = useState<string[]>([]);
  const [indexes, setIndexes] = useState<string[]>([]);
  const [audios, setAudios] = useState<string[]>([]);
  const [loadError, setLoadError] = useState("");

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [pthPath, setPthPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [inputPath, setInputPath] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [protect, setProtect] = useState(0.33);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [embedderModel, setEmbedderModel] = useState("contentvec");
  const [exportFormat, setExportFormat] = useState("WAV");
  const [splitAudio, setSplitAudio] = useState(false);
  const [f0Autotune, setF0Autotune] = useState(false);
  const [f0AutotuneStrength, setF0AutotuneStrength] = useState(1);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.7);
  const [terms, setTerms] = useState(false);

  // Formant Shifting
  const [formantShifting, setFormantShifting] = useState(false);
  const [formantQfrency, setFormantQfrency] = useState(1.0);
  const [formantTimbre, setFormantTimbre] = useState(1.0);

  // Studio Post-Process FX Rack
  const [postProcess, setPostProcess] = useState(false);
  const [reverb, setReverb] = useState(false);
  const [reverbRoomSize, setReverbRoomSize] = useState(0.5);
  const [reverbWetGain, setReverbWetGain] = useState(0.5);
  const [reverbDryGain, setReverbDryGain] = useState(0.5);

  const [delay, setDelay] = useState(false);
  const [delaySeconds, setDelaySeconds] = useState(0.3);
  const [delayMix, setDelayMix] = useState(0.4);

  const [compressor, setCompressor] = useState(false);
  const [compressorThreshold, setCompressorThreshold] = useState(-12);
  const [compressorRatio, setCompressorRatio] = useState(4);

  const [limiter, setLimiter] = useState(false);
  const [limiterThreshold, setLimiterThreshold] = useState(-2);

  const [chorus, setChorus] = useState(false);
  const [chorusRate, setChorusRate] = useState(1.0);
  const [chorusDepth, setChorusDepth] = useState(0.25);

  const [distortion, setDistortion] = useState(false);
  const [distortionGain, setDistortionGain] = useState(15);

  const [gain, setGain] = useState(false);
  const [gainDb, setGainDb] = useState(0);

  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  // Check URL query parameters for model pre-selection (e.g. from /models)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const m = params.get("model");
      if (m) {
        setPthPath(m);
        const idx = params.get("index");
        if (idx) setIndexPath(idx);
      }
    }
  }, []);

  // Preset apply/save bridge (PresetsPanel dispatches / requests these events)
  useEffect(() => {
    const onApply = (e: Event) => {
      const v = (e as CustomEvent).detail as {
        pitch: number;
        index_rate: number;
        rms_mix_rate: number;
        protect: number;
      };
      if (typeof v.pitch === "number") setPitch(Math.max(-24, Math.min(24, v.pitch)));
      if (typeof v.index_rate === "number") setIndexRate(v.index_rate);
      if (typeof v.rms_mix_rate === "number") setVolumeEnvelope(v.rms_mix_rate);
      if (typeof v.protect === "number") setProtect(v.protect);
    };
    const onRequest = () => {
      window.dispatchEvent(
        new CustomEvent("applio:read-preset", {
          detail: { pitch, index_rate: indexRate, rms_mix_rate: volumeEnvelope, protect },
        }),
      );
    };
    window.addEventListener("applio:apply-preset", onApply);
    window.addEventListener("applio:request-preset", onRequest);
    return () => {
      window.removeEventListener("applio:apply-preset", onApply);
      window.removeEventListener("applio:request-preset", onRequest);
    };
  }, [indexRate, pitch, protect, volumeEnvelope]);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m.models);
        setIndexes(m.indexes);
        setAudios(m.audios);
        if (m.models.length > 0 && !pthPath) {
          setPthPath(m.models[0]);
          const stem = (m.models[0].split("/").pop() ?? "").replace(/\.(pth|onnx)$/, "");
          const match = m.indexes.find((i) => i.toLowerCase().includes(stem.toLowerCase().slice(0, 8)));
          if (match) setIndexPath(match);
        }
        if (m.audios.length > 0) setInputPath(m.audios[0]);
      })
      .catch((e) => setLoadError(errMsg(e)));
  }, [pthPath]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: polling is keyed by job id on purpose
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const stop = pollJob(job.id, setJob);
    return stop;
  }, [job?.id]);

  // Create temporary URL for uploaded original audio for A/B comparison
  const originalAudioUrl = useMemo(() => {
    if (audioFile) return URL.createObjectURL(audioFile);
    if (inputPath) return `/${inputPath}`;
    return null;
  }, [audioFile, inputPath]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError("");
    if (!terms) {
      setSubmitError("You must agree to the Terms of Use to proceed.");
      return;
    }
    if (!pthPath) {
      setSubmitError("Select a voice model (.pth).");
      return;
    }
    if (!audioFile && !inputPath) {
      setSubmitError("Upload an audio file or pick one from assets/audios.");
      return;
    }
    const fd = new FormData();
    if (audioFile) fd.append("audio", audioFile);
    if (inputPath) fd.append("inputPath", inputPath);
    fd.append("pthPath", pthPath);
    fd.append("indexPath", indexPath);
    fd.append("pitch", String(pitch));
    fd.append("indexRate", String(indexRate));
    fd.append("volumeEnvelope", String(volumeEnvelope));
    fd.append("protect", String(protect));
    fd.append("f0Method", f0Method);
    fd.append("embedderModel", embedderModel);
    fd.append("exportFormat", exportFormat);
    fd.append("splitAudio", String(splitAudio));
    fd.append("f0Autotune", String(f0Autotune));
    if (f0Autotune) fd.append("f0AutotuneStrength", String(f0AutotuneStrength));
    fd.append("cleanAudio", String(cleanAudio));
    if (cleanAudio) fd.append("cleanStrength", String(cleanStrength));

    // Formant shifting
    if (formantShifting) {
      fd.append("formantShifting", "true");
      fd.append("formantQfrency", String(formantQfrency));
      fd.append("formantTimbre", String(formantTimbre));
    }

    // Studio Post-Processing
    if (postProcess) {
      fd.append("postProcess", "true");
      if (reverb) {
        fd.append("reverb", "true");
        fd.append("reverbRoomSize", String(reverbRoomSize));
        fd.append("reverbWetGain", String(reverbWetGain));
        fd.append("reverbDryGain", String(reverbDryGain));
      }
      if (delay) {
        fd.append("delay", "true");
        fd.append("delaySeconds", String(delaySeconds));
        fd.append("delayMix", String(delayMix));
      }
      if (compressor) {
        fd.append("compressor", "true");
        fd.append("compressorThreshold", String(compressorThreshold));
        fd.append("compressorRatio", String(compressorRatio));
      }
      if (limiter) {
        fd.append("limiter", "true");
        fd.append("limiterThreshold", String(limiterThreshold));
      }
      if (chorus) {
        fd.append("chorus", "true");
        fd.append("chorusRate", String(chorusRate));
        fd.append("chorusDepth", String(chorusDepth));
      }
      if (gain) {
        fd.append("gain", "true");
        fd.append("gainDb", String(gainDb));
      }
      if (distortion) {
        fd.append("distortion", "true");
        fd.append("distortionGain", String(distortionGain));
      }
    }

    setSubmitting(true);
    try {
      const { jobId } = await submitInference(fd);
      const { job: fresh } = await fetchJob(jobId);
      setJob(fresh);
    } catch (err) {
      setSubmitError(errMsg(err) || "Submit failed");
    } finally {
      setSubmitting(false);
    }
  }

  const directAudioUrl = job?.outputFile ? `/outputs/${job.outputFile.split("/").pop()}` : null;

  return (
    <form onSubmit={onSubmit}>
      {loadError && (
        <div className="card">
          <strong>API offline.</strong>{" "}
          <span className="muted">
            Start it with <code>npm run dev</code>. {loadError}
          </span>
        </div>
      )}

      <div className="flex gap-4 items-start flex-col lg:flex-row">
        {/* Left Column: Model & Input Audio */}
        <div className="flex flex-col w-full lg:w-[320px] lg:min-w-[320px]">
          <div className="card">
            <h2>Model Selection</h2>
            <div className="space-y-3">
              <div>
                <label>Voice Model (.pth)</label>
                <input
                  type="text"
                  list="models"
                  value={pthPath}
                  onChange={(e) => setPthPath(e.target.value)}
                  placeholder="logs/my-model/model.pth"
                />
                <datalist id="models">
                  {models.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>
              <div>
                <label>Index File (.index, optional)</label>
                <input
                  type="text"
                  list="indexes"
                  value={indexPath}
                  onChange={(e) => setIndexPath(e.target.value)}
                  placeholder="logs/my-model/added.index"
                />
                <datalist id="indexes">
                  {indexes.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>
            </div>
            <p className="muted text-xs mt-2">
              Models are automatically discovered in <code>logs/</code>.
            </p>
          </div>

          <div className="card">
            <h2>Audio Input</h2>
            <label>Upload audio (wav/mp3/flac/ogg/m4a, max 200MB)</label>
            <input
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.aiff,.webm"
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
            />
            <label>…or pick a file already in assets/audios</label>
            <input
              type="text"
              list="audios"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="assets/audios/input.wav"
            />
            <datalist id="audios">
              {audios.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
        </div>

        {/* Right Column: Settings, FX Rack, and Conversion */}
        <div className="flex-1 flex flex-col min-w-0 w-full">
          <div className="card">
            <h2>Conversion Settings</h2>
            <div className="grid2">
              <div>
                <label>Pitch: {pitch} semitones (-24…24)</label>
                <input
                  type="range"
                  min={-24}
                  max={24}
                  step={1}
                  value={pitch}
                  onChange={(e) => setPitch(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Search Feature Ratio: {indexRate}</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={indexRate}
                  onChange={(e) => setIndexRate(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Volume Envelope: {volumeEnvelope}</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={volumeEnvelope}
                  onChange={(e) => setVolumeEnvelope(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Protect Voiceless Consonants: {protect}</label>
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={protect}
                  onChange={(e) => setProtect(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Pitch Extraction Algorithm</label>
                <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
                  {F0_METHODS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Embedder Model</label>
                <select value={embedderModel} onChange={(e) => setEmbedderModel(e.target.value)}>
                  {EMBEDDERS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Export Format</label>
                <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
                  {FORMATS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Advanced Tuning Details */}
            <details className="mt-4 border-t border-white/10 pt-3">
              <summary className="cursor-pointer text-sm font-semibold text-neutral-300 hover:text-white">
                Advanced Tuning (Split / Autotune / Noise Clean)
              </summary>
              <div className="space-y-3 mt-3">
                <div className="row flex-wrap gap-4">
                  <label className="flex items-center gap-2 cursor-pointer m-0">
                    <input
                      type="checkbox"
                      checked={splitAudio}
                      onChange={(e) => setSplitAudio(e.target.checked)}
                    />
                    <span>Split Audio (Process in chunks)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer m-0">
                    <input
                      type="checkbox"
                      checked={f0Autotune}
                      onChange={(e) => setF0Autotune(e.target.checked)}
                    />
                    <span>Pitch Autotune</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer m-0">
                    <input
                      type="checkbox"
                      checked={cleanAudio}
                      onChange={(e) => setCleanAudio(e.target.checked)}
                    />
                    <span>Clean Audio Artifacts</span>
                  </label>
                </div>
                {f0Autotune && (
                  <div>
                    <label>Autotune Strength: {f0AutotuneStrength}</label>
                    <input
                      type="range"
                      min={0.1}
                      max={1}
                      step={0.05}
                      value={f0AutotuneStrength}
                      onChange={(e) => setF0AutotuneStrength(Number(e.target.value))}
                    />
                  </div>
                )}
                {cleanAudio && (
                  <div>
                    <label>Clean Strength: {cleanStrength}</label>
                    <input
                      type="range"
                      min={0.1}
                      max={1}
                      step={0.05}
                      value={cleanStrength}
                      onChange={(e) => setCleanStrength(Number(e.target.value))}
                    />
                  </div>
                )}
              </div>
            </details>

            {/* Formant Shifting Section */}
            <details className="mt-3 border-t border-white/10 pt-3">
              <summary className="cursor-pointer text-sm font-semibold text-neutral-300 hover:text-white flex items-center gap-2">
                <Sparkles size={16} />
                <span>Formant Shifting (Vocal Tract & Timbre)</span>
              </summary>
              <div className="space-y-3 mt-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formantShifting}
                    onChange={(e) => setFormantShifting(e.target.checked)}
                  />
                  <span>Enable Formant Shifting</span>
                </label>
                {formantShifting && (
                  <div className="grid2">
                    <div>
                      <label>Formant Q-Frequency: {formantQfrency}</label>
                      <input
                        type="range"
                        min={0.5}
                        max={2.0}
                        step={0.05}
                        value={formantQfrency}
                        onChange={(e) => setFormantQfrency(Number(e.target.value))}
                      />
                    </div>
                    <div>
                      <label>Formant Timbre: {formantTimbre}</label>
                      <input
                        type="range"
                        min={0.5}
                        max={2.0}
                        step={0.05}
                        value={formantTimbre}
                        onChange={(e) => setFormantTimbre(Number(e.target.value))}
                      />
                    </div>
                  </div>
                )}
              </div>
            </details>

            {/* Studio FX Rack Section */}
            <details className="mt-3 border-t border-white/10 pt-3">
              <summary className="cursor-pointer text-sm font-semibold text-neutral-300 hover:text-white flex items-center gap-2">
                <Sliders size={16} />
                <span>Studio Audio FX Rack (Reverb, Delay, Compressor, Chorus)</span>
              </summary>
              <div className="space-y-4 mt-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={postProcess}
                    onChange={(e) => setPostProcess(e.target.checked)}
                  />
                  <span>Enable Studio Post-Process FX Chain</span>
                </label>

                {postProcess && (
                  <div className="space-y-4 border border-white/5 rounded-xl p-4 bg-black/30">
                    {/* Reverb */}
                    <div>
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                        <input
                          type="checkbox"
                          checked={reverb}
                          onChange={(e) => setReverb(e.target.checked)}
                        />
                        <span>Studio Reverb</span>
                      </label>
                      {reverb && (
                        <div className="grid2 mt-2">
                          <div>
                            <label>Room Size: {reverbRoomSize}</label>
                            <input
                              type="range"
                              min={0.1}
                              max={1}
                              step={0.05}
                              value={reverbRoomSize}
                              onChange={(e) => setReverbRoomSize(Number(e.target.value))}
                            />
                          </div>
                          <div>
                            <label>Wet Gain: {reverbWetGain}</label>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.05}
                              value={reverbWetGain}
                              onChange={(e) => setReverbWetGain(Number(e.target.value))}
                            />
                          </div>
                          <div>
                            <label>Dry Gain: {reverbDryGain}</label>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.05}
                              value={reverbDryGain}
                              onChange={(e) => setReverbDryGain(Number(e.target.value))}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Delay */}
                    <div>
                      <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                        <input type="checkbox" checked={delay} onChange={(e) => setDelay(e.target.checked)} />
                        <span>Stereo Delay</span>
                      </label>
                      {delay && (
                        <div className="grid2 mt-2">
                          <div>
                            <label>Delay Time: {delaySeconds}s</label>
                            <input
                              type="range"
                              min={0.05}
                              max={1.0}
                              step={0.05}
                              value={delaySeconds}
                              onChange={(e) => setDelaySeconds(Number(e.target.value))}
                            />
                          </div>
                          <div>
                            <label>Delay Mix: {delayMix}</label>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.05}
                              value={delayMix}
                              onChange={(e) => setDelayMix(Number(e.target.value))}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Compressor & Limiter */}
                    <div className="grid2">
                      <div>
                        <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                          <input
                            type="checkbox"
                            checked={compressor}
                            onChange={(e) => setCompressor(e.target.checked)}
                          />
                          <span>Compressor</span>
                        </label>
                        {compressor && (
                          <div className="space-y-2 mt-2">
                            <label>Threshold: {compressorThreshold} dB</label>
                            <input
                              type="range"
                              min={-40}
                              max={0}
                              step={1}
                              value={compressorThreshold}
                              onChange={(e) => setCompressorThreshold(Number(e.target.value))}
                            />
                            <label>Ratio: {compressorRatio}:1</label>
                            <input
                              type="range"
                              min={1}
                              max={20}
                              step={0.5}
                              value={compressorRatio}
                              onChange={(e) => setCompressorRatio(Number(e.target.value))}
                            />
                          </div>
                        )}
                      </div>

                      <div>
                        <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                          <input
                            type="checkbox"
                            checked={limiter}
                            onChange={(e) => setLimiter(e.target.checked)}
                          />
                          <span>Peak Limiter</span>
                        </label>
                        {limiter && (
                          <div className="space-y-2 mt-2">
                            <label>Ceiling: {limiterThreshold} dB</label>
                            <input
                              type="range"
                              min={-12}
                              max={0}
                              step={0.5}
                              value={limiterThreshold}
                              onChange={(e) => setLimiterThreshold(Number(e.target.value))}
                            />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Chorus, Distortion & Gain */}
                    <div className="grid2">
                      <div>
                        <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                          <input
                            type="checkbox"
                            checked={chorus}
                            onChange={(e) => setChorus(e.target.checked)}
                          />
                          <span>Chorus / Detune</span>
                        </label>
                        {chorus && (
                          <div className="space-y-2 mt-2">
                            <label>Rate: {chorusRate} Hz</label>
                            <input
                              type="range"
                              min={0.1}
                              max={5}
                              step={0.1}
                              value={chorusRate}
                              onChange={(e) => setChorusRate(Number(e.target.value))}
                            />
                            <label>Depth: {chorusDepth}</label>
                            <input
                              type="range"
                              min={0.05}
                              max={1}
                              step={0.05}
                              value={chorusDepth}
                              onChange={(e) => setChorusDepth(Number(e.target.value))}
                            />
                          </div>
                        )}
                      </div>

                      <div>
                        <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                          <input
                            type="checkbox"
                            checked={distortion}
                            onChange={(e) => setDistortion(e.target.checked)}
                          />
                          <span>Harmonic Distortion</span>
                        </label>
                        {distortion && (
                          <div className="space-y-2 mt-2">
                            <label>Drive Gain: {distortionGain} dB</label>
                            <input
                              type="range"
                              min={0}
                              max={40}
                              step={1}
                              value={distortionGain}
                              onChange={(e) => setDistortionGain(Number(e.target.value))}
                            />
                          </div>
                        )}
                      </div>

                      <div>
                        <label className="flex items-center gap-2 cursor-pointer font-medium text-white">
                          <input type="checkbox" checked={gain} onChange={(e) => setGain(e.target.checked)} />
                          <span>Output Gain Boost</span>
                        </label>
                        {gain && (
                          <div className="space-y-2 mt-2">
                            <label>Gain: {gainDb} dB</label>
                            <input
                              type="range"
                              min={-12}
                              max={12}
                              step={0.5}
                              value={gainDb}
                              onChange={(e) => setGainDb(Number(e.target.value))}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </details>
          </div>

          {/* Conversion Action & Audio Player Result */}
          <div className="card">
            <h2>Conversion</h2>
            <label className="terms flex items-center gap-2 cursor-pointer mb-3">
              <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
              <span>I agree to the Terms of Use (conversion is enabled once checked).</span>
            </label>

            <div className="row" style={{ marginTop: 12 }}>
              <button type="submit" className="cta" disabled={submitting}>
                {submitting ? "Submitting…" : "Convert Voice"}
              </button>
              {job && <span className={`badge ${job.status}`}>{job.status}</span>}
              {job && <span className="muted text-xs">job {job.id}</span>}
            </div>

            {submitError && <p style={{ color: "var(--err)" }}>{submitError}</p>}

            {/* Custom Studio Audio Player with A/B compare */}
            {job && directAudioUrl && job.status === "done" && (
              <div className="mt-4">
                <AudioPlayer
                  src={directAudioUrl}
                  originalSrc={originalAudioUrl}
                  title={`Converted output · ${pthPath.split("/").pop()}`}
                  filename={job.outputFile?.split("/").pop()}
                />
              </div>
            )}

            {job && job.status === "error" && <p style={{ color: "var(--err)" }}>{job.error}</p>}

            {job && job.logs.length > 0 && (
              <div className="mt-4">
                <p className="muted text-xs mb-1">Engine logs (streaming from Python engine)</p>
                <div className="log">{job.logs.slice(-60).join("\n")}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </form>
  );
}
