"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Radio, Disc, Play, Square, ListMusic, ChevronDown } from "lucide-react";
import PageHeader from "../../components/layout/PageHeader";
import SliderField from "../../components/ui/SliderField";
import { apiGet, apiSend, errMsg, fetchModels } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { useSpeakers } from "../../lib/useSpeakers";

const API_HTTP = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
function apiWs(path: string): string {
  const u = API_HTTP.replace(/^http/, "ws");
  return `${u}${path}`;
}

// AudioWorklet processors for the realtime streaming client
const INPUT_WORKLET = `
class InputProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.ring = new Float32Array(48000); this.pos = 0; this.block = 0;
    this.port.onmessage = (e) => { this.block = e.data.block_frame || 0; }; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) { for (let i = 0; i < ch.length; i++) { this.ring[this.pos] = ch[i]; this.pos = (this.pos + 1) % this.ring.length; } }
    if (this.block > 0 && this.pos % this.block === 0 && ch) {
      const out = new Float32Array(this.block);
      for (let i = 0; i < this.block; i++) out[i] = this.ring[(this.pos - this.block + i + this.ring.length) % this.ring.length];
      this.port.postMessage({ chunk: out }, [out.buffer]);
    }
    return true;
  }
}
registerProcessor('input-processor', InputProcessor);`;

const PLAYBACK_WORKLET = `
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.ring = new Float32Array(98304); this.rp = 0; this.wp = 0;
    this.port.onmessage = (e) => { const c = new Float32Array(e.data.chunk);
      for (let i = 0; i < c.length; i++) { this.ring[this.wp] = c[i]; this.wp = (this.wp + 1) % this.ring.length; } }; }
  process(inputs, outputs) {
    const outL = outputs[0] && outputs[0][0];
    const outR = outputs[0] && outputs[0][1];
    if (!outL) return true;
    const len = outL.length;
    for (let i = 0; i < len; i++) {
      let s = 0;
      if (this.rp !== this.wp) { s = this.ring[this.rp]; this.rp = (this.rp + 1) % this.ring.length; }
      outL[i] = s; if (outR) outR[i] = s;
    }
    return true;
  }
}
registerProcessor('playback-processor', PlaybackProcessor);`;

interface RtStatus {
  running: boolean;
  startedAt: string | null;
  logs: string[];
}

export default function RealtimePage() {
  const { t } = useI18n();
  const [engine, setEngine] = useState<RtStatus | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [indexes, setIndexes] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [index, setIndex] = useState("");
  const [inputs, setInputs] = useState<Array<{ id: string; label: string }>>([]);
  const [outputs, setOutputs] = useState<Array<{ id: string; label: string }>>([]);
  const [inDev, setInDev] = useState("");
  const [outDev, setOutDev] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0);
  const [protect, setProtect] = useState(0.5);
  const [volumeEnvelope, setVolumeEnvelope] = useState(1);
  const [sid, setSid] = useState(0);
  const [f0Method, setF0Method] = useState("fcpe");
  const [embedder, setEmbedder] = useState("contentvec");
  const [embedderCustom, setEmbedderCustom] = useState("");
  const [autotune, setAutotune] = useState(false);
  const [autotuneStrength, setAutotuneStrength] = useState(1);
  const [proposedPitch, setProposedPitch] = useState(false);
  const [proposedPitchThreshold, setProposedPitchThreshold] = useState(155);
  const [cleanAudio, setCleanAudio] = useState(false);
  const [cleanStrength, setCleanStrength] = useState(0.5);
  const [chunkMs, setChunkMs] = useState(250);
  const [crossfade, setCrossfade] = useState(0.05);
  const [extraSize, setExtraSize] = useState(2.5);
  const [silent, setSilent] = useState(-60);
  const [vad, setVad] = useState(true);
  const [inGain, setInGain] = useState(100);
  const [outGain, setOutGain] = useState(100);
  const [streaming, setStreaming] = useState(false);
  const [latency, setLatency] = useState(0);
  const [volume, setVolume] = useState(-90);
  const [msg, setMsg] = useState("");
  const [recOn, setRecOn] = useState(false);
  const [recPath, setRecPath] = useState("assets/audios/record_audio.wav");
  const [recFormat, setRecFormat] = useState("WAV");

  const speakers = useSpeakers(model);

  useEffect(() => {
    if (!speakers.includes(sid)) setSid(0);
  }, [speakers, sid]);

  const sessRef = useRef<{
    ws: WebSocket;
    ctx: AudioContext;
    stream: MediaStream;
    nodes: AudioNode[];
    els: HTMLAudioElement[];
  } | null>(null);

  const refreshEngine = useCallback(async () => {
    try {
      // Status polls must bypass the apiGet cache or engine state freezes.
      setEngine(await apiGet<RtStatus>("/api/realtime/status", { ttlMs: 0 }));
    } catch (e) {
      setMsg(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refreshEngine();
    fetchModels()
      .then((m) => {
        setModels(m.models);
        setIndexes(m.indexes);
        if (m.models[0]) setModel(m.models[0]);
      })
      .catch(() => {});
    const t = setInterval(refreshEngine, 5000);
    return () => clearInterval(t);
  }, [refreshEngine]);

  async function startEngine() {
    setMsg(t("Starting real-time audio service…"));
    try {
      await apiSend("/api/realtime/start", "POST");
      setMsg(t("Real-time audio service running"));
      refreshEngine();
    } catch (e) {
      setMsg(errMsg(e));
    }
  }

  async function stopEngine() {
    await apiSend("/api/realtime/stop", "POST").catch(() => {});
    refreshEngine();
  }

  async function enumDevices() {
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      const devs = await navigator.mediaDevices.enumerateDevices();
      setInputs(
        devs
          .filter((d) => d.kind === "audioinput")
          .map((d, i) => ({ id: d.deviceId, label: d.label || `Input ${i + 1}` })),
      );
      setOutputs(
        devs
          .filter((d) => d.kind === "audiooutput")
          .map((d, i) => ({ id: d.deviceId, label: d.label || `Output ${i + 1}` })),
      );
    } catch {
      setMsg(t("Microphone permission denied — device list unavailable."));
    }
  }

  async function startStream() {
    setMsg("");
    if (!engine?.running) {
      setMsg(t("Start the engine first."));
      return;
    }
    if (!model) {
      setMsg(t("Select a voice model."));
      return;
    }
    try {
      const block = Math.round((chunkMs * 48000) / 1000);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          ...(inDev ? { deviceId: { exact: inDev } } : {}),
          channelCount: { exact: 1 },
          sampleRate: { exact: 48000 },
        },
      });
      const ctx = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      await ctx.audioWorklet.addModule(
        URL.createObjectURL(new Blob([INPUT_WORKLET], { type: "application/javascript" })),
      );
      await ctx.audioWorklet.addModule(
        URL.createObjectURL(new Blob([PLAYBACK_WORKLET], { type: "application/javascript" })),
      );
      const src = ctx.createMediaStreamSource(stream);
      const inNode = new AudioWorkletNode(ctx, "input-processor");
      inNode.port.postMessage({ block_frame: block });
      src.connect(inNode);
      const playNode = new AudioWorkletNode(ctx, "playback-processor", { outputChannelCount: [2] });
      const gain = ctx.createGain();
      gain.gain.value = outGain / 100;
      playNode.connect(gain);
      const dest = ctx.createMediaStreamDestination();
      gain.connect(dest);
      const el = new Audio();
      el.srcObject = dest.stream;
      const anyEl = el as HTMLAudioElement & { setSinkId?: (id: string) => Promise<void> };
      if (outDev && anyEl.setSinkId) {
        try {
          await anyEl.setSinkId(outDev);
        } catch {
          /* Chrome-only; fall back to default output */
        }
      }
      await el.play();

      const ws = new WebSocket(apiWs("/api/realtime/ws-audio"));
      ws.binaryType = "arraybuffer";
      sessRef.current = { ws, ctx, stream, nodes: [src, inNode, playNode, gain], els: [el] };
      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "init",
            block_frame: block,
            cross_fade_overlap_size: crossfade,
            extra_convert_size: extraSize,
            model_path: model,
            index_path: index || "",
            f0_method: f0Method,
            embedder_model: embedder,
            embedder_model_custom: embedder === "custom" ? embedderCustom : "",
            silent_threshold: silent,
            vad_enabled: vad,
            sid,
            input_audio_gain: inGain,
            f0_up_key: pitch,
            index_rate: indexRate,
            protect,
            volume_envelope: volumeEnvelope,
            autotune,
            autotune_strength: autotuneStrength,
            proposed_pitch: proposedPitch,
            proposed_pitch_threshold: proposedPitchThreshold,
            clean_audio: cleanAudio,
            clean_strength: cleanStrength,
            post_process: false,
            kwargs: {},
          }),
        );
        setStreaming(true);
        setMsg(t("Streaming ✓ speak into your microphone."));
        apiSend("/api/realtime/config", "PUT", { model_file: model, index_file: index }).catch(() => {});
      };
      inNode.port.onmessage = (e) => {
        const chunk: Float32Array = e.data.chunk;
        if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          try {
            const m = JSON.parse(ev.data);
            if (m.type === "latency") setLatency(m.value);
            if (typeof m.volume === "number") setVolume(m.volume);
          } catch {
            /* ignore */
          }
        } else {
          playNode.port.postMessage({ chunk: ev.data }, [ev.data]);
        }
      };
      ws.onclose = () => {
        if (sessRef.current) stopStream(true);
      };
      ws.onerror = () => setMsg(t("WebSocket error — is the engine running?"));
    } catch (e) {
      setMsg(errMsg(e));
      stopStream(true);
    }
  }

  function stopStream(silentStop = false) {
    const s = sessRef.current;
    sessRef.current = null;
    try {
      s?.ws.close();
    } catch {
      /* noop */
    }
    try {
      s?.stream.getTracks().forEach((t) => {
        t.stop();
      });
    } catch {
      /* noop */
    }
    try {
      s?.nodes.forEach((n) => {
        n.disconnect();
      });
    } catch {
      /* noop */
    }
    try {
      s?.els.forEach((el) => {
        el.pause();
      });
    } catch {
      /* noop */
    }
    try {
      s?.ctx.close();
    } catch {
      /* noop */
    }
    setStreaming(false);
    if (!silentStop) setMsg(t("Stopped."));
  }

  async function changeConfig(key: string, value: number | string | boolean, ifKwargs = false) {
    // Output gain is applied to the local GainNode instead.
    if (key === "output_audio_gain") return;
    try {
      const ws = new WebSocket(apiWs("/api/realtime/change-config"));
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("change-config unreachable"));
        setTimeout(() => reject(new Error("change-config timeout")), 5000);
      });
      ws.send(JSON.stringify({ type: "init", key, value, if_kwargs: ifKwargs }));
      setTimeout(() => ws.close(), 500);
    } catch (e) {
      setMsg(errMsg(e));
    }
  }

  async function toggleRecord() {
    setMsg("");
    if (!engine?.running) {
      setMsg(t("Start the engine first."));
      return;
    }
    try {
      const r = await apiSend<{ type: string; value: string; button: string; path: string | null }>(
        "/api/realtime/record",
        "POST",
        {
          record_button: recOn ? "Stop" : "Start",
          record_audio_path: recPath || undefined,
          export_format: recFormat,
        },
      );
      setRecOn(r.button === "Stop");
      setMsg(r.value + (r.path ? ` → ${r.path}` : ""));
    } catch (e) {
      setMsg(errMsg(e));
    }
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Realtime")}
        description={t(
          "Stream low-latency live microphone audio through voice conversion models in real time.",
        )}
      >
        <span className={`badge ${engine?.running ? "done" : "queued"}`}>
          {engine?.running ? t("active") : t("stopped")}
        </span>
        {!engine?.running ? (
          <button type="button" className="cta h-8 px-3 text-xs flex items-center gap-1.5 rounded-lg" onClick={startEngine}>
            <Play size={14} className="shrink-0" />
            <span>{t("Start Service")}</span>
          </button>
        ) : (
          <button type="button" className="ghost h-8 px-3 text-xs flex items-center gap-1.5 rounded-lg" onClick={stopEngine}>
            <Square size={14} className="text-white shrink-0" />
            <span>{t("Stop Service")}</span>
          </button>
        )}
        <button type="button" className="ghost h-8 px-3 text-xs flex items-center gap-1.5 rounded-lg" onClick={enumDevices}>
          <ListMusic size={14} className="text-white shrink-0" />
          <span>{t("List Audio Devices")}</span>
        </button>
      </PageHeader>
      <div>
        {msg && (
          <p className="text-xs text-neutral-400 m-0" role="status" aria-live="polite">
            {msg}
          </p>
        )}
        {engine && engine.logs.length > 0 && (
          <details className="mt-2 text-xs text-neutral-400 group">
            <summary className="cursor-pointer hover:text-white transition-colors py-1 flex items-center gap-1 select-none">
              <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
              <span>{t("Activity Details")}</span>
            </summary>
            <pre className="log mt-1 max-h-40 overflow-y-auto text-[11px] p-2 rounded-lg bg-black/40 border border-white/5 font-sans" role="log" aria-live="polite">
              {engine.logs.slice(-10).join("\n")}
            </pre>
          </details>
        )}
      </div>

      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Model & Audio Devices")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Configure real-time low-latency inference inputs, target voice model, and DSP pitch parameters.")}
          </p>
        </div>
        <div className="grid2">
          <div>
            <label htmlFor="rt-model">{t("Voice Model")}</label>
            <input
              id="rt-model"
              type="text"
              list="rtmodels"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            />
            <datalist id="rtmodels">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label htmlFor="rt-index">{t("Index (optional)")}</label>
            <input
              id="rt-index"
              type="text"
              list="rtidx"
              value={index}
              onChange={(e) => setIndex(e.target.value)}
            />
            <datalist id="rtidx">
              {indexes.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label htmlFor="rt-in-dev">{t("Input Device")}</label>
            <select id="rt-in-dev" value={inDev} onChange={(e) => setInDev(e.target.value)}>
              <option value="">{t("Default")}</option>
              {inputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="rt-out-dev">{t("Output Device")}</label>
            <select id="rt-out-dev" value={outDev} onChange={(e) => setOutDev(e.target.value)}>
              <option value="">{t("Default")}</option>
              {outputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <SliderField
              id="rt-pitch"
              label={t("Pitch")}
              value={pitch}
              min={-24}
              max={24}
              step={1}
              unit="st"
              onChange={(v) => {
                setPitch(v);
                if (streaming) changeConfig("f0_up_key", v);
              }}
            />
          </div>
          <div>
            <SliderField
              id="rt-index-rate"
              label={t("Index rate")}
              value={indexRate}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => {
                setIndexRate(v);
                if (streaming) changeConfig("index_rate", v);
              }}
            />
          </div>
          <div>
            <SliderField
              id="rt-protect"
              label={t("Protect Voiceless Consonants")}
              value={protect}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(v) => {
                setProtect(v);
                if (streaming) changeConfig("protect", v);
              }}
            />
          </div>
          <div>
            <SliderField
              id="rt-volume-envelope"
              label={t("Volume Envelope")}
              value={volumeEnvelope}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => {
                setVolumeEnvelope(v);
                if (streaming) changeConfig("volume_envelope", v);
              }}
            />
          </div>
          <div>
            <label htmlFor="rt-speaker-id">{t("Speaker ID")}</label>
            <select
              id="rt-speaker-id"
              value={sid}
              onChange={(e) => {
                setSid(Number(e.target.value));
                if (streaming) changeConfig("sid", Number(e.target.value));
              }}
            >
              {speakers.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="rt-f0-method">{t("Pitch extraction")}</label>
            <select id="rt-f0-method" value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {["rmvpe", "fcpe", "crepe", "crepe-tiny"].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="rt-embedder">{t("Embedder")}</label>
            <select id="rt-embedder" value={embedder} onChange={(e) => setEmbedder(e.target.value)}>
              {[
                "contentvec",
                "spin",
                "spin-v2",
                "chinese-hubert-base",
                "japanese-hubert-base",
                "korean-hubert-base",
                "custom",
              ].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          {embedder === "custom" && (
            <div>
              <label htmlFor="rt-custom-embedder">{t("Custom embedder path (reconnect to apply)")}</label>
              <input
                id="rt-custom-embedder"
                type="text"
                value={embedderCustom}
                onChange={(e) => setEmbedderCustom(e.target.value)}
                placeholder="rvc/models/embedders/embedders_custom/my-embedder"
              />
            </div>
          )}
        </div>
        <details>
          <summary>{t("Voice cleanup (autotune / proposed pitch / clean)")}</summary>
          <div className="row">
            <label htmlFor="rt-autotune" className="flex items-center gap-2 cursor-pointer">
              <input
                id="rt-autotune"
                type="checkbox"
                checked={autotune}
                onChange={(e) => {
                  setAutotune(e.target.checked);
                  if (streaming) changeConfig("autotune", e.target.checked);
                }}
              />{" "}
              {t("Autotune")}
            </label>
            <label htmlFor="rt-proposed-pitch" className="flex items-center gap-2 cursor-pointer">
              <input
                id="rt-proposed-pitch"
                type="checkbox"
                checked={proposedPitch}
                onChange={(e) => {
                  setProposedPitch(e.target.checked);
                  if (streaming) changeConfig("proposed_pitch", e.target.checked);
                }}
              />{" "}
              {t("Proposed Pitch")}
            </label>
            <label htmlFor="rt-clean-audio" className="flex items-center gap-2 cursor-pointer">
              <input
                id="rt-clean-audio"
                type="checkbox"
                checked={cleanAudio}
                onChange={(e) => {
                  setCleanAudio(e.target.checked);
                  if (streaming) changeConfig("clean_audio", e.target.checked);
                }}
              />{" "}
              {t("Clean Audio")}
            </label>
          </div>
          <div className="grid2" style={{ marginTop: 8 }}>
            <div>
              <SliderField
                id="rt-autotune-strength"
                label={t("Autotune Strength")}
                value={autotuneStrength}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setAutotuneStrength(v);
                  if (streaming) changeConfig("autotune_strength", v);
                }}
              />
            </div>
            <div>
              <SliderField
                id="rt-proposed-threshold"
                label={t("Proposed Pitch Threshold")}
                value={proposedPitchThreshold}
                min={50}
                max={1200}
                step={1}
                unit="Hz"
                onChange={(v) => {
                  setProposedPitchThreshold(v);
                  if (streaming) changeConfig("proposed_pitch_threshold", v);
                }}
              />
            </div>
            <div>
              <SliderField
                id="rt-clean-strength"
                label={t("Clean Strength")}
                value={cleanStrength}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => {
                  setCleanStrength(v);
                  if (streaming) changeConfig("clean_strength", v);
                }}
              />
            </div>
          </div>
        </details>
        <details>
          <summary>{t("Latency / VAD / gains")}</summary>
          <div className="grid2">
            <div>
              <SliderField
                id="rt-chunk-ms"
                label={`${t("Chunk")} ${t("(reconnect to apply)")}`}
                value={chunkMs}
                min={50}
                max={1000}
                step={10}
                unit="ms"
                onChange={setChunkMs}
              />
            </div>
            <div>
              <SliderField
                id="rt-crossfade"
                label={t("Crossfade")}
                value={crossfade}
                min={0.05}
                max={0.2}
                step={0.01}
                unit="s"
                onChange={(v) => {
                  setCrossfade(v);
                  if (streaming) changeConfig("cross_fade_overlap_size", v);
                }}
              />
            </div>
            <div>
              <SliderField
                id="rt-extra-size"
                label={t("Extra convert")}
                value={extraSize}
                min={0.1}
                max={5}
                step={0.1}
                unit="s"
                onChange={(v) => {
                  setExtraSize(v);
                  if (streaming) changeConfig("extra_convert_size", v);
                }}
              />
            </div>
            <div>
              <SliderField
                id="rt-silent-threshold"
                label={t("Silence threshold")}
                value={silent}
                min={-90}
                max={-60}
                step={1}
                unit="dB"
                onChange={(v) => {
                  setSilent(v);
                  if (streaming) changeConfig("silent_threshold", v);
                }}
              />
            </div>
            <div>
              <SliderField
                id="rt-in-gain"
                label={t("Input gain")}
                value={inGain}
                min={0}
                max={200}
                step={1}
                unit="%"
                onChange={setInGain}
              />
            </div>
            <div>
              <SliderField
                id="rt-out-gain"
                label={`${t("Output gain")} ${t("(local)")}`}
                value={outGain}
                min={0}
                max={200}
                step={1}
                unit="%"
                onChange={setOutGain}
              />
            </div>
          </div>
          <label
            htmlFor="rt-vad-enabled"
            className="checkbox-label flex items-center gap-2 cursor-pointer mt-3"
          >
            <input
              id="rt-vad-enabled"
              type="checkbox"
              checked={vad}
              onChange={(e) => {
                setVad(e.target.checked);
                if (streaming) changeConfig("vad_enabled", e.target.checked);
              }}
            />
            <span>{t("Enable VAD")}</span>
          </label>
        </details>
        <div className="flex items-center justify-between gap-4 pt-2 border-t border-white/5">
          <div className="flex items-center gap-3">
            {!streaming ? (
              <button
                type="button"
                className="cta h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl"
                onClick={startStream}
              >
                <Play size={16} className="shrink-0" />
                <span>{t("Start Streaming")}</span>
              </button>
            ) : (
              <button
                type="button"
                className="ghost h-10 px-5 flex items-center gap-2 text-sm font-medium rounded-xl text-neutral-200 hover:text-white"
                onClick={() => stopStream()}
              >
                <Square size={16} className="text-white shrink-0" />
                <span>{t("Stop Streaming")}</span>
              </button>
            )}
          </div>
          {streaming && (
            <span className="text-xs text-neutral-400 tabular-nums" role="status" aria-live="polite">
              latency {latency.toFixed(0)}ms · volume {volume.toFixed(0)}dB
            </span>
          )}
        </div>
      </div>

      <div className="card space-y-4">
        <div className="border-b border-white/10 pb-3.5 space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Disc size={18} className="text-white" />
              <h2 className="text-base font-bold text-white m-0">{t("Record Output")}</h2>
            </div>
          </div>
          <p className="text-xs text-neutral-400 m-0 leading-relaxed">
            {t("Records the converted stream server-side via the engine.")}
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
          <div>
            <label htmlFor="rt-rec-path">{t("Recording path (server)")}</label>
            <input
              id="rt-rec-path"
              type="text"
              value={recPath}
              onChange={(e) => setRecPath(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="rt-rec-format">{t("Export Format")}</label>
            <select id="rt-rec-format" value={recFormat} onChange={(e) => setRecFormat(e.target.value)}>
              {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="pt-3.5 border-t border-white/5 flex justify-end">
          <button
            type="button"
            className={
              recOn
                ? "ghost h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl text-neutral-200 hover:text-white"
                : "cta h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl"
            }
            onClick={toggleRecord}
          >
            <Disc size={16} className={recOn ? "animate-pulse text-white shrink-0" : "shrink-0"} />
            <span>{recOn ? t("Stop Recording") : t("Start Recording")}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
