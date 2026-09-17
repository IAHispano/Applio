"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import PageHeader from "../../components/layout/PageHeader";
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
    const out = outputs[0];
    for (let n = 0; n < out[0].length; n++) {
      let v = 0;
      if (this.rp !== this.wp) { v = this.ring[this.rp]; this.rp = (this.rp + 1) % this.ring.length; }
      for (let c = 0; c < out.length; c++) out[c][n] = v;
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
  const [terms, setTerms] = useState(false);
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
    setMsg(t("Starting realtime engine (uvicorn + rvc/realtime/client.py)…"));
    try {
      await apiSend("/api/realtime/start", "POST");
      setMsg(t("Engine running ✓"));
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
    if (!terms) {
      setMsg(t("You must agree to the Terms of Use to proceed."));
      return;
    }
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
    <div>
      <PageHeader
        title={t("Realtime")}
        description={t(
          "Stream low-latency live microphone audio through voice conversion models in real time.",
        )}
      >
        <span className={`badge ${engine?.running ? "done" : "queued"}`}>
          {engine?.running ? t("engine running") : t("engine stopped")}
        </span>
        {!engine?.running ? (
          <button type="button" className="cta" onClick={startEngine}>
            {t("Start Engine")}
          </button>
        ) : (
          <button type="button" className="ghost" onClick={stopEngine}>
            {t("Stop Engine")}
          </button>
        )}
        <button type="button" className="ghost" onClick={enumDevices}>
          {t("List Audio Devices")}
        </button>
      </PageHeader>
      <div className="mb-4">
        {msg && <p className="muted">{msg}</p>}
        {engine && engine.logs.length > 0 && (
          <div className="log" style={{ marginTop: 8 }}>
            {engine.logs.slice(-10).join("\n")}
          </div>
        )}
      </div>

      <div className="card">
        <h2>{t("Model + Audio")}</h2>
        <div className="grid2">
          <div>
            <label>{t("Voice Model")}</label>
            <input type="text" list="rtmodels" value={model} onChange={(e) => setModel(e.target.value)} />
            <datalist id="rtmodels">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label>{t("Index (optional)")}</label>
            <input type="text" list="rtidx" value={index} onChange={(e) => setIndex(e.target.value)} />
            <datalist id="rtidx">
              {indexes.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <div>
            <label>{t("Input Device")}</label>
            <select value={inDev} onChange={(e) => setInDev(e.target.value)}>
              <option value="">{t("Default")}</option>
              {inputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>{t("Output Device")}</label>
            <select value={outDev} onChange={(e) => setOutDev(e.target.value)}>
              <option value="">{t("Default")}</option>
              {outputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>
              {t("Pitch")}: {pitch}
            </label>
            <input
              type="range"
              min={-24}
              max={24}
              step={1}
              value={pitch}
              onChange={(e) => {
                setPitch(Number(e.target.value));
                if (streaming) changeConfig("f0_up_key", Number(e.target.value));
              }}
            />
          </div>
          <div>
            <label>
              {t("Index rate")}: {indexRate}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={indexRate}
              onChange={(e) => {
                setIndexRate(Number(e.target.value));
                if (streaming) changeConfig("index_rate", Number(e.target.value));
              }}
            />
          </div>
          <div>
            <label>
              {t("Protect Voiceless Consonants")}: {protect} {t("(default 0.5)")}
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={protect}
              onChange={(e) => {
                setProtect(Number(e.target.value));
                if (streaming) changeConfig("protect", Number(e.target.value));
              }}
            />
          </div>
          <div>
            <label>
              {t("Volume Envelope")}: {volumeEnvelope} {t("(default 1)")}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volumeEnvelope}
              onChange={(e) => {
                setVolumeEnvelope(Number(e.target.value));
                if (streaming) changeConfig("volume_envelope", Number(e.target.value));
              }}
            />
          </div>
          <div>
            <label>{t("Speaker ID")}</label>
            <select
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
            <label>{t("Pitch extraction")}</label>
            <select value={f0Method} onChange={(e) => setF0Method(e.target.value)}>
              {["rmvpe", "fcpe", "crepe", "crepe-tiny"].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>{t("Embedder")}</label>
            <select value={embedder} onChange={(e) => setEmbedder(e.target.value)}>
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
              <label>{t("Custom embedder path (reconnect to apply)")}</label>
              <input
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
            <label>
              <input
                type="checkbox"
                checked={autotune}
                onChange={(e) => {
                  setAutotune(e.target.checked);
                  if (streaming) changeConfig("autotune", e.target.checked);
                }}
              />{" "}
              {t("Autotune")}
            </label>
            <label>
              <input
                type="checkbox"
                checked={proposedPitch}
                onChange={(e) => {
                  setProposedPitch(e.target.checked);
                  if (streaming) changeConfig("proposed_pitch", e.target.checked);
                }}
              />{" "}
              {t("Proposed Pitch")}
            </label>
            <label>
              <input
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
              <label>
                {t("Autotune Strength")}: {autotuneStrength}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={autotuneStrength}
                onChange={(e) => {
                  setAutotuneStrength(Number(e.target.value));
                  if (streaming) changeConfig("autotune_strength", Number(e.target.value));
                }}
              />
            </div>
            <div>
              <label>
                {t("Proposed Pitch Threshold")}: {proposedPitchThreshold}
              </label>
              <input
                type="range"
                min={50}
                max={1200}
                step={1}
                value={proposedPitchThreshold}
                onChange={(e) => {
                  setProposedPitchThreshold(Number(e.target.value));
                  if (streaming) changeConfig("proposed_pitch_threshold", Number(e.target.value));
                }}
              />
            </div>
            <div>
              <label>
                {t("Clean Strength")}: {cleanStrength}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={cleanStrength}
                onChange={(e) => {
                  setCleanStrength(Number(e.target.value));
                  if (streaming) changeConfig("clean_strength", Number(e.target.value));
                }}
              />
            </div>
          </div>
        </details>
        <details>
          <summary>{t("Latency / VAD / gains")}</summary>
          <div className="grid2">
            <div>
              <label>
                {t("Chunk")}: {chunkMs}ms {t("(reconnect to apply)")}
              </label>
              <input
                type="range"
                min={50}
                max={1000}
                step={10}
                value={chunkMs}
                onChange={(e) => setChunkMs(Number(e.target.value))}
              />
            </div>
            <div>
              <label>
                {t("Crossfade")}: {crossfade}s
              </label>
              <input
                type="range"
                min={0.05}
                max={0.2}
                step={0.01}
                value={crossfade}
                onChange={(e) => {
                  setCrossfade(Number(e.target.value));
                  if (streaming) changeConfig("cross_fade_overlap_size", Number(e.target.value));
                }}
              />
            </div>
            <div>
              <label>
                {t("Extra convert")}: {extraSize}s
              </label>
              <input
                type="range"
                min={0.1}
                max={5}
                step={0.1}
                value={extraSize}
                onChange={(e) => {
                  setExtraSize(Number(e.target.value));
                  if (streaming) changeConfig("extra_convert_size", Number(e.target.value));
                }}
              />
            </div>
            <div>
              <label>
                {t("Silence threshold")}: {silent}dB
              </label>
              <input
                type="range"
                min={-90}
                max={-60}
                step={1}
                value={silent}
                onChange={(e) => {
                  setSilent(Number(e.target.value));
                  if (streaming) changeConfig("silent_threshold", Number(e.target.value));
                }}
              />
            </div>
            <div>
              <label>
                {t("Input gain")}: {inGain}%
              </label>
              <input
                type="range"
                min={0}
                max={200}
                step={1}
                value={inGain}
                onChange={(e) => setInGain(Number(e.target.value))}
              />
            </div>
            <div>
              <label>
                {t("Output gain")}: {outGain}% {t("(local)")}
              </label>
              <input
                type="range"
                min={0}
                max={200}
                step={1}
                value={outGain}
                onChange={(e) => setOutGain(Number(e.target.value))}
              />
            </div>
          </div>
          <label className="checkbox-label">
            <input
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
        <label className="terms">
          <input type="checkbox" checked={terms} onChange={(e) => setTerms(e.target.checked)} />
          <span>{t("I agree to the terms of use")}</span>
        </label>
        <div className="row" style={{ marginTop: 12 }}>
          {!streaming ? (
            <button type="button" className="cta" onClick={startStream}>
              {t("Start Streaming")}
            </button>
          ) : (
            <button type="button" className="ghost" onClick={() => stopStream()}>
              {t("Stop Streaming")}
            </button>
          )}
          {streaming && (
            <span className="muted">
              latency {latency.toFixed(0)}ms · volume {volume.toFixed(0)}dB
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <h2>{t("Record Output")}</h2>
        <p className="muted">{t("Records the converted stream server-side via the engine.")}</p>
        <div className="grid2">
          <div>
            <label>{t("Recording path (server)")}</label>
            <input type="text" value={recPath} onChange={(e) => setRecPath(e.target.value)} />
          </div>
          <div>
            <label>{t("Export Format")}</label>
            <select value={recFormat} onChange={(e) => setRecFormat(e.target.value)}>
              {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button type="button" className={recOn ? "ghost" : "cta"} onClick={toggleRecord}>
            {recOn ? t("Stop") : t("Start")}
          </button>
        </div>
      </div>
    </div>
  );
}
