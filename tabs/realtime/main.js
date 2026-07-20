window._activeStream = null;
window._audioCtx = null;
window._workletNode = null;
window._playbackNode = null;
window._ws = null;
window.OutputAudioRoute = null;
window.MonitorAudioRoute = null;
window.lastSend = 0;
window.responseMs = 0;

// Function to display status
function setStatus(msg, use_alert = true) {
  const realtimeStatus = document.querySelector(
    "#realtime-status-info h2.output-class",
  ); // find status text box
  if (use_alert) alert(msg); // Use alert instead of gr.Info

  if (realtimeStatus) {
    realtimeStatus.innerText = msg;
    realtimeStatus.style.whiteSpace = "nowrap";
    realtimeStatus.style.textAlign = "center";
  }
}

async function addModuleFromString(ctx, codeStr) {
  const blob = new Blob([codeStr], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);

  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
}

function createOutputRoute(audioCtx, playbackNode, sinkId, gainValue = 1.0) {
  const dest = audioCtx.createMediaStreamDestination(); // Create a MediaStreamDestination Node
  const gainNode = audioCtx.createGain(); // Create a GainNode (Volume Control Node)
  gainNode.gain.value = gainValue; // Sets the initial gain (volume).

  // Connect the Audio Nodes
  playbackNode.connect(gainNode);
  gainNode.connect(dest);

  // Create and Configure the audio Element
  const el = document.createElement("audio");
  el.autoplay = true;
  el.srcObject = dest.stream;
  el.style.display = "none";
  document.body.appendChild(el);

  if (el.setSinkId) el.setSinkId(sinkId).catch((err) => console.error(err));
  return { dest, gainNode, el }; // Returns the objects (destination node, gain node, and audio element)
}

const inputWorkletSource = `
class InputProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferCapacity = 48000; // Stores up to 1 second of audio at 48kHz
        this.buffer = new Float32Array(this.bufferCapacity);
        this.writePointer = 0;
        this.readPointer = 0;
        this.availableSamples = 0;
        this.block_frame = 960; // Standard layout processing slice size
        // Handle dynamic frame configuration updates from the main main thread
        this.port.onmessage = (e) => {
            if (e.data && e.data.block_frame) this.block_frame = e.data.block_frame;
        };
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true; // Keep worklet alive if no stream is active

        const frame = input[0];
        const frameSize = frame.length;

        // Push new incoming samples into the internal circular ring buffer
        if (this.availableSamples + frameSize <= this.bufferCapacity) {
            for (let i = 0; i < frameSize; i++) {
                this.buffer[this.writePointer] = frame[i];
                this.writePointer = (this.writePointer + 1) % this.bufferCapacity;
            }

            this.availableSamples += frameSize;
        }

        // Slice accumulated audio into uniform blocks and dispatch them to the main thread
        while (this.availableSamples >= this.block_frame) {
            const chunk = new Float32Array(this.block_frame);

            for (let i = 0; i < this.block_frame; i++) {
                chunk[i] = this.buffer[this.readPointer];
                this.readPointer = (this.readPointer + 1) % this.bufferCapacity;
            }

            this.availableSamples -= this.block_frame;
            this.port.postMessage({ chunk }); 
        }
        return true;
    }
}
registerProcessor('input-processor', InputProcessor);
`;

const playbackWorkletSource = `
class PlaybackProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        const bufferSize = options.processorOptions && options.processorOptions.bufferSize ? options.processorOptions.bufferSize : 98304;
        this.buffer = new Float32Array(bufferSize);
        this.bufferCapacity = bufferSize;
        this.writePointer = 0;
        this.readPointer = 0;
        this.availableSamples = 0;

        // Listen for returned server-processed audio chunks and load them into the playback ring buffer
        this.port.onmessage = (e) => {
            if (e.data && e.data.chunk) {
                const chunk = new Float32Array(e.data.chunk);
                const chunkSize = chunk.length;

                // Guard against ring buffer overflows (drop chunk if filled)
                if (this.availableSamples + chunkSize > this.bufferCapacity) return;

                // Handle standard inline write vs circular wrapping wrap-around logic
                if (this.writePointer + chunkSize <= this.bufferCapacity) {
                    this.buffer.set(chunk, this.writePointer);
                } else {
                    const firstPart = this.bufferCapacity - this.writePointer;
                    this.buffer.set(chunk.subarray(0, firstPart), this.writePointer);
                    this.buffer.set(chunk.subarray(firstPart), 0);
                }

                this.writePointer = (this.writePointer + chunkSize) % this.bufferCapacity;
                this.availableSamples += chunkSize;
            }
        };
    }

    process(inputs, outputs) {
        const output = outputs[0];
        if (!output || !output[0]) return true;

        const frame = output[0];
        const frameSize = frame.length;

        // Populate the hardware output frame if there are enough accumulated samples
        if (this.availableSamples >= frameSize) {
            if (this.readPointer + frameSize <= this.bufferCapacity) {
                frame.set(this.buffer.subarray(this.readPointer, this.readPointer + frameSize));
            } else {
                const firstPart = this.bufferCapacity - this.readPointer;
                frame.set(this.buffer.subarray(this.readPointer, this.bufferCapacity), 0);
                frame.set(this.buffer.subarray(0, frameSize - firstPart), firstPart);
            }

            this.readPointer = (this.readPointer + frameSize) % this.bufferCapacity;
            this.availableSamples -= frameSize;
        } else {
            // Underflow protection: fill buffer with silence (zeros) to prevent harsh digital crackling
            frame.fill(0);
        }

        // Duplicate audio channel configurations for stereo topologies if supported
        if (output.length > 1) output[1].set(output[0]);
        return true;
    }
}
registerProcessor('playback-processor', PlaybackProcessor);
`;

window.getAudioDevices = async function () {
  if (!navigator.mediaDevices) {
    // If somehow the browser does not support.
    setStatus("Browser does not support accessing audio devices via API");
    return { inputs: {}, outputs: {} };
  }

  try {
    // Request audio permissions to the browser.
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error(err);
    setStatus("It looks like your device doesn't have an audio input.");

    return { inputs: {}, outputs: {} };
  }

  // Read the audio devices available on the browser and filter out the devices.
  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = {};
  const outputs = {};

  for (const device of devices) {
    if (device.kind === "audioinput") {
      inputs[device.label + ` (${device.deviceId.slice(0, 10)})`] = device.deviceId;
    } else if (device.kind === "audiooutput") {
      outputs[device.label + ` (${device.deviceId.slice(0, 10)})`] = device.deviceId;
    }
  }

  // Returns the audio devices
  if (!Object.keys(inputs).length && !Object.keys(outputs).length)
    return { inputs: {}, outputs: {} };
  return { inputs: inputs, outputs: outputs };
};

window.StreamAudioRealtime = async function (
  terms_checkbox,
  input_audio_device,
  input_audio_gain,
  output_audio_device,
  output_audio_gain,
  monitor_output_device,
  monitor_audio_gain,
  use_monitor_device,
  vad_enabled,
  chunk_size,
  cross_fade_overlap_size,
  extra_convert_size,
  silent_threshold,
  pitch,
  index_rate,
  volume_envelope,
  protect,
  f0_method,
  model_file,
  index_file,
  sid,
  autotune,
  autotune_strength,
  proposed_pitch,
  proposed_pitch_threshold,
  embedder_model,
  embedder_model_custom,
  exclusive_mode,
  clean_audio,
  clean_strength,
  post_process,
  reverb,
  pitch_shift,
  limiter,
  gain,
  distortion,
  chorus,
  bitcrush,
  clipping,
  compressor,
  delay,
  reverb_room_size,
  reverb_damping,
  reverb_wet_gain,
  reverb_dry_gain,
  reverb_width,
  reverb_freeze_mode,
  pitch_shift_semitones,
  limiter_threshold,
  limiter_release_time,
  gain_db,
  distortion_gain,
  chorus_rate,
  chorus_depth,
  chorus_center_delay,
  chorus_feedback,
  chorus_mix,
  bitcrush_bit_depth,
  clipping_threshold,
  compressor_threshold,
  compressor_ratio,
  compressor_attack,
  compressor_release,
  delay_seconds,
  delay_feedback,
  delay_mix,
) {
  const SampleRate = 48000;
  const ReadChunkSize = Math.round((chunk_size * SampleRate) / 1000 / 128);
  const block_frame = parseInt(ReadChunkSize) * 128;
  const ButtonState = { start_button: true, stop_button: false };

  if (!terms_checkbox) {
    setStatus("You must agree to the Terms of Use to proceed.");
    return ButtonState;
  }

  const devices = await window.getAudioDevices();
  input_audio_device = devices["inputs"][input_audio_device];
  output_audio_device = devices["outputs"][output_audio_device];

  if (use_monitor_device && devices["outputs"][monitor_output_device]) {
    monitor_output_device = devices["outputs"][monitor_output_device];
  }

  try {
    if (!input_audio_device || !output_audio_device) {
      setStatus("Please select valid input/output devices!");
      return ButtonState;
    }

    if (use_monitor_device && !monitor_output_device) {
      setStatus("Please select a valid monitor device!");
      return ButtonState;
    }

    if (!model_file) {
      setStatus("Model path not provided. Aborting conversion.");
      return ButtonState;
    }

    setStatus("Starting Realtime...", (use_alert = false));

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: { exact: input_audio_device }, // audio: input_audio_device ? { deviceId: { exact: input_audio_device } } : true
        channelCount: { exact: 1 },
        sampleRate: { exact: SampleRate },
        latency: { ideal: 0 },
        // disable all browser processing (You can make it optional)
        echoCancellation: !exclusive_mode,
        noiseSuppression: !exclusive_mode,
        autoGainControl: !exclusive_mode,
      },
    });

    let latencyHint = "playback";
    if (exclusive_mode) latencyHint = "interactive";

    window._activeStream = stream;
    window._audioCtx = new AudioContext({
      sampleRate: SampleRate,
      latencyHint: latencyHint,
    });

    // Load processing modules.
    await addModuleFromString(window._audioCtx, inputWorkletSource);
    await addModuleFromString(window._audioCtx, playbackWorkletSource);
    // await window._audioCtx.audioWorklet.addModule('/input_processor.js');
    // await window._audioCtx.audioWorklet.addModule('/playback_processor.js');

    // Initialize audio web parts
    const src = window._audioCtx.createMediaStreamSource(stream);
    const inputNode = new AudioWorkletNode(window._audioCtx, "input-processor");
    const playbackNode = new AudioWorkletNode(
      window._audioCtx,
      "playback-processor",
      {
        processorOptions: {
          bufferSize: block_frame * 2, // Double or more is recommended to avoid loss of sound.
        },
      },
    );

    inputNode.port.postMessage({ block_frame: block_frame });
    src.connect(inputNode);

    // Create audio and monitor output
    window.OutputAudioRoute = createOutputRoute(
      window._audioCtx,
      playbackNode,
      output_audio_device,
      output_audio_gain / 100,
    );
    if (use_monitor_device && monitor_output_device)
      window.MonitorAudioRoute = createOutputRoute(
        window._audioCtx,
        playbackNode,
        monitor_output_device,
        monitor_audio_gain / 100,
      );
    // Configure path and websocket
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl =
      protocol +
      "//" +
      location.hostname +
      `:${location.port}` +
      "/api/ws-audio";
    const ws = new WebSocket(wsUrl);

    // Set new values ​​of buttons to avoid users initiating multiple realtime threads
    ButtonState.start_button = false;
    ButtonState.stop_button = true;

    ws.binaryType = "arraybuffer";
    window._ws = ws;

    ws.onopen = () => {
      console.log("[WS] Connected!");
      // send all parameters to websocket realtime
      ws.send(
        JSON.stringify({
          type: "init",
          chunk_size: ReadChunkSize,
          cross_fade_overlap_size: cross_fade_overlap_size,
          extra_convert_size: extra_convert_size,
          model_path: model_file,
          index_path: index_file || "",
          f0_method: f0_method,
          embedder_model: embedder_model,
          embedder_model_custom: embedder_model_custom || "",
          silent_threshold: silent_threshold,
          vad_enabled: vad_enabled,
          sid: sid,
          input_audio_gain: input_audio_gain,
          f0_up_key: pitch,
          index_rate: index_rate,
          protect: protect,
          volume_envelope: volume_envelope,
          autotune: autotune,
          autotune_strength: autotune_strength,
          proposed_pitch: proposed_pitch,
          proposed_pitch_threshold: proposed_pitch_threshold,
          clean_audio: clean_audio,
          clean_strength: clean_strength,
          post_process: post_process,
          kwargs: {
            reverb: reverb,
            pitch_shift: pitch_shift,
            limiter: limiter,
            gain: gain,
            distortion: distortion,
            chorus: chorus,
            bitcrush: bitcrush,
            clipping: clipping,
            compressor: compressor,
            delay: delay,
            reverb_room_size: reverb_room_size,
            reverb_damping: reverb_damping,
            reverb_wet_level: reverb_wet_gain,
            reverb_dry_level: reverb_dry_gain,
            reverb_width: reverb_width,
            reverb_freeze_mode: reverb_freeze_mode,
            pitch_shift_semitones: pitch_shift_semitones,
            limiter_threshold: limiter_threshold,
            limiter_release: limiter_release_time,
            gain_db: gain_db,
            distortion_gain: distortion_gain,
            chorus_rate: chorus_rate,
            chorus_depth: chorus_depth,
            chorus_delay: chorus_center_delay,
            chorus_feedback: chorus_feedback,
            chorus_mix: chorus_mix,
            bitcrush_bit_depth: bitcrush_bit_depth,
            clipping_threshold: clipping_threshold,
            compressor_threshold: compressor_threshold,
            compressor_ratio: compressor_ratio,
            compressor_attack: compressor_attack,
            compressor_release: compressor_release,
            delay_seconds: delay_seconds,
            delay_feedback: delay_feedback,
            delay_mix: delay_mix,
          },
        }),
      );
    };

    inputNode.port.onmessage = (e) => {
      // send audio from node to websocket for realtime
      const chunk = e.data && e.data.chunk;

      if (!chunk) return;
      if (ws.readyState === WebSocket.OPEN) {
        window.lastSend = performance.now();
        ws.send(chunk); // Send raw audio
      }
    };

    ws.onmessage = (ev) => {
      // Read the string values ​​sent back from the websocket
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        // Show latency information in the status bar of the interface
        if (msg.type === "latency")
          setStatus(
            `Latency: ${msg.value.toFixed(2)} ms | Volume: ${msg.volume.toFixed(2)} dB | Response: ${window.responseMs.toFixed(2)} ms`,
            (use_alert = false),
          );
        return;
      }
      // Send audio to the playback node to the audio device
      const ab = ev.data;
      playbackNode.port.postMessage({ chunk: ab }, [ab]);

      window.responseMs = performance.now() - window.lastSend;
    };

    ws.onclose = () => console.log("[WS] Closed!");
    window._workletNode = inputNode;
    window._playbackNode = playbackNode;

    if (window._audioCtx.state === "suspended") await window._audioCtx.resume();

    console.log("Realtime is ready!");
    return ButtonState;
  } catch (err) {
    console.error("An error has occurred:", err);
    alert("An error has occurred:" + err.message);
    // stop realtime when error
    return StopAudioStream();
  }
};

window.ChangeConfig = async function (value, key, if_kwargs = false) {
  if (key === "output_audio_gain") {
    window.OutputAudioRoute.gainNode.gain.value = value / 100;
  } else if (key == "monitor_audio_gain") {
    if (window.MonitorAudioRoute)
      window.MonitorAudioRoute.gainNode.gain.value = value / 100;
  } else {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl =
      protocol +
      "//" +
      location.hostname +
      `:${location.port}` +
      "/api/change-config";
    const ws = new WebSocket(wsUrl);

    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "init",
          key: key,
          value: value,
          if_kwargs: if_kwargs,
        }),
      );

      ws.close();
    };
  }
};

window.StopAudioStream = async function () {
  try {
    if (window._ws) {
      window._ws.close();
      window._ws = null;
    }

    if (window._activeStream) {
      window._activeStream.getTracks().forEach((t) => t.stop());
      window._activeStream = null;
    }

    if (window._workletNode) {
      window._workletNode.disconnect();
      window._workletNode = null;
    }

    if (window._playbackNode) {
      window._playbackNode.disconnect();
      window._playbackNode = null;
    }

    if (window._audioCtx) {
      await window._audioCtx.close();
      window._audioCtx = null;
    }

    if (window.OutputAudioRoute) window.OutputAudioRoute = null;
    if (window.MonitorAudioRoute) window.MonitorAudioRoute = null;

    document.querySelectorAll("audio").forEach((a) => {
      a.pause();
      a.srcObject = null;
      a.remove();
    });
    setStatus("Stopped", (use_alert = false));

    return { start_button: true, stop_button: false };
  } catch (e) {
    setStatus(`An error has occurred: ${e}`);

    return { start_button: false, stop_button: true };
  }
};

window.SoundfileRecordAudio = async function (
  RecordButton,
  RecordAudioPath,
  ExportFormat,
) {
  const protocol = location.protocol === "https:" ? "https:" : "http:";
  const url =
    protocol + "//" + location.hostname + `:${location.port}` + "/api/record";

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      record_button: RecordButton,
      record_audio_path: RecordAudioPath,
      export_format: ExportFormat,
    }),
  });

  const msg = await res.json();

  if (msg.type === "info" || msg.type === "warnings") {
    alert(msg.value);

    return {
      button: msg.button,
      path: msg.path,
    };
  }
};
