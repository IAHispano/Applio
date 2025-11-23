() => {
    window._activeStream = null;
    window._audioCtx = null;
    window._workletNode = null;
    window._playbackNode = null;
    window._ws = null;

    // Function to display status
    function setStatus(msg, use_alert = true) {
        const realtimeStatus = document.querySelector("#realtime-status-info h2.output-class"); // find status text box
        if (use_alert) alert(msg); // Use alert instead of gr.Info

        if (realtimeStatus) {
            realtimeStatus.innerText = msg;
            realtimeStatus.style.whiteSpace = "nowrap";
            realtimeStatus.style.textAlign = "center";
        }
    }

    async function addModuleFromString(ctx, codeStr) {
        const blob = new Blob([codeStr], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);

        await ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
    };

    function createOutputRoute(audioCtx, playbackNode, sinkId, gainValue = 1.0) {
        const dest = audioCtx.createMediaStreamDestination(); // Create a MediaStreamDestination Node
        const gainNode = audioCtx.createGain(); // Create a GainNode (Volume Control Node)
        gainNode.gain.value = gainValue; // Sets the initial gain (volume).

        // Connect the Audio Nodes
        playbackNode.connect(gainNode);
        gainNode.connect(dest);

        // Create and Configure the audio Element
        const el = document.createElement('audio');
        el.autoplay = true;
        el.srcObject = dest.stream;
        el.style.display = 'none';
        document.body.appendChild(el);

        if (el.setSinkId) el.setSinkId(sinkId).catch(err => console.error(err));
        return { dest, gainNode, el }; // Returns the objects (destination node, gain node, and audio element)
    }

    const inputWorkletSource = `
        class InputProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                // Initialize a buffer to hold incoming audio data. Starts empty.
                this.buffer = new Float32Array(0);
                // The desired size for each chunk of audio data to be sent out. Default is 128 samples.
                this.block_frame = 128;
                // Set up an event listener.
                this.port.onmessage = (e) => {
                    // Allows the main thread to dynamically change the chunk size.
                    if (e.data && e.data.block_frame) this.block_frame = e.data.block_frame;
                };
            }

            // The main method called by the AudioWorklet system every processing block (usually 128 samples).
            process(inputs) {
                // Get the data from the first input port (inputs[0]) and the first audio channel (input[0]).
                const input = inputs[0];
                // Check if there is valid input data. If not, return 'true' to continue running.
                if (!input || !input[0]) return true;
                const frame = input[0]; // 'frame' is a Float32Array containing the 128 new audio samples.

                // Create a new array with a size equal to the length of the old buffer plus the new frame.
                const newBuf = new Float32Array(this.buffer.length + frame.length);
                newBuf.set(this.buffer, 0); // Copy the old buffer data to the start of the new array.
                newBuf.set(frame, this.buffer.length); // Append the new frame data to the end of the new array.
                this.buffer = newBuf;

                // Loop until the buffer is smaller than the required 'block_frame' size.
                while (this.buffer.length >= this.block_frame) {
                    // Slice the required chunk size from the beginning of the buffer.
                    const chunk = this.buffer.slice(0, this.block_frame);
                    // Send the audio chunk
                    this.port.postMessage({chunk}, [chunk.buffer]);
                    // Remove the sent chunk from the buffer
                    this.buffer = this.buffer.slice(this.block_frame);
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
                    // Get the buffer size from options (or use a default value of 98304).
                    const bufferSize = options.processorOptions && options.processorOptions.bufferSize ? options.processorOptions.bufferSize: 98304;
                    // Circular Buffer initialization:
                    this.buffer = new Float32Array(bufferSize); // The array holding the audio data.
                    this.bufferCapacity = bufferSize; // The maximum capacity of the buffer.
                    this.writePointer = 0; // The next position to write new data.
                    this.readPointer = 0; // The next position to read data.
                    this.availableSamples = 0; // The count of audio samples currently in the buffer.
                    // Set up a listener for incoming audio chunks from the main.
                    this.port.onmessage = (e) => {
                        if (e.data && e.data.chunk) {
                            const chunk = new Float32Array(e.data.chunk); // The received audio data chunk.
                            const chunkSize = chunk.length;

                            // Check if adding the new chunk would overflow the buffer. If so, discard it.
                            if (this.availableSamples + chunkSize > this.bufferCapacity) return;

                            // Write the data chunk into the circular buffer.
                            for (let i = 0; i < chunkSize; i++) {
                                this.buffer[this.writePointer] = chunk[i];
                                // Advance the write pointer, wrapping around when reaching the end.
                                this.writePointer = (this.writePointer + 1) % this.bufferCapacity;
                            }

                            // Update the count of available samples.
                            this.availableSamples += chunkSize;
                        }
                    };
                }

                // The main method called when the system needs audio data for playback.
                process(inputs, outputs) {
                    // Get the first channel of the first output port.
                    const output = outputs[0];
                    if (!output || !output[0]) return true;

                    const frame = output[0]; // 'frame' is the Float32Array that must be filled with output audio data.
                    const frameSize = frame.length;

                    // Check if there are enough samples in the buffer to fill the output frame.
                    if (this.availableSamples >= frameSize) {
                        // Read data from the circular buffer and fill the output frame.
                        for (let i = 0; i < frameSize; i++) {
                            frame[i] = this.buffer[this.readPointer];
                            // Advance the read pointer, wrapping around.
                            this.readPointer = (this.readPointer + 1) % this.bufferCapacity;
                        }
                        // Update the count of available samples.
                        this.availableSamples -= frameSize;
                    } else {
                        frame.fill(0);
                    }

                    // If a second channel exists (for stereo playback), copy the data from the first channel (mono-to-stereo).
                    if (output.length > 1) output[1].set(output[0]);
                    return true;
                }
            }
            registerProcessor('playback-processor', PlaybackProcessor);
            `;

    window.getAudioDevices = async function() {
        if (!navigator.mediaDevices) { // If somehow the browser does not support.
            setStatus("Browser does not support accessing audio devices via API");
            return {"inputs": {}, "outputs": {}};
        }

        try {
            // Request audio permissions to the browser.
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error(err);
            setStatus("It looks like your device doesn't have an audio input.")

            return {"inputs": {}, "outputs": {}};
        }

        // Read the audio devices available on the browser and filter out the devices.
        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = {};
        const outputs = {};
        
        for (const device of devices) {
            if (device.kind === "audioinput") {
                inputs[device.label] = device.deviceId
            } else if (device.kind === "audiooutput") {
                outputs[device.label] = device.deviceId
            }
        }

        // Returns the audio devices
        if (!Object.keys(inputs).length && !Object.keys(outputs).length) return {"inputs": {}, "outputs": {}};
        return {"inputs": inputs, "outputs": outputs};
    };
        
    window.StreamAudioRealtime = async function(
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
    ) {
        const SampleRate = 48000;
        const ReadChunkSize = Math.round(chunk_size * SampleRate / 1000 / 128);
        const block_frame = parseInt(ReadChunkSize) * 128;
        const ButtonState = { start_button: true, stop_button: false };

        if (!terms_checkbox) {
            setStatus("You must agree to the Terms of Use to proceed.")
            return ButtonState;
        }
        
        const devices = await window.getAudioDevices()
        input_audio_device = devices["inputs"][input_audio_device]
        output_audio_device = devices["outputs"][output_audio_device]
        
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
                setStatus("Model path not provided. Aborting conversion.")
                return ButtonState;
            }

            setStatus("Starting Realtime...", use_alert=false)

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: { exact: input_audio_device }, // audio: input_audio_device ? { deviceId: { exact: input_audio_device } } : true
                    channelCount: 1,
                    sampleRate: SampleRate,
                    // disable all browser processing (You can make it optional)
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            window._activeStream = stream;
            window._audioCtx = new AudioContext({ sampleRate: SampleRate, latencyHint: "interactive" });

            // Load processing modules.
            await addModuleFromString(window._audioCtx, inputWorkletSource);
            await addModuleFromString(window._audioCtx, playbackWorkletSource);
            // await window._audioCtx.audioWorklet.addModule('/input_processor.js');
            // await window._audioCtx.audioWorklet.addModule('/playback_processor.js');

            // Initialize audio web parts
            const src = window._audioCtx.createMediaStreamSource(stream);
            const inputNode = new AudioWorkletNode(window._audioCtx, 'input-processor');
            const playbackNode = new AudioWorkletNode(window._audioCtx, 'playback-processor', {
                processorOptions: {
                    bufferSize: block_frame * 2 // Double or more is recommended to avoid loss of sound.
                }
            });

            inputNode.port.postMessage({ block_frame: block_frame });
            src.connect(inputNode);

            // Create audio and monitor output
            createOutputRoute(window._audioCtx, playbackNode, output_audio_device, output_audio_gain / 100);
            if (use_monitor_device && monitor_output_device) createOutputRoute(window._audioCtx, playbackNode, monitor_output_device, monitor_audio_gain / 100);
            // Configure path and websocket
            const protocol = (location.protocol === "https:") ? "wss:" : "ws:";
            const wsUrl = protocol + '//' + location.hostname + `:${location.port}` + '/api/ws-audio';
            const ws = new WebSocket(wsUrl);

            // Set new values ​​of buttons to avoid users initiating multiple realtime threads
            ButtonState.start_button = false;
            ButtonState.stop_button = true;

            ws.binaryType = "arraybuffer";
            window._ws = ws;

            ws.onopen = () => {
                console.log("[WS] Connected!")
                // send all parameters to websocket realtime
                ws.send(
                    JSON.stringify({
                        type: 'init',
                        chunk_size: ReadChunkSize,
                        cross_fade_overlap_size: cross_fade_overlap_size,
                        extra_convert_size: extra_convert_size,
                        model_file: model_file,
                        index_file: index_file || '',
                        f0_method: f0_method,
                        embedder_model: embedder_model,
                        embedder_model_custom: embedder_model_custom || '',
                        silent_threshold: silent_threshold,
                        vad_enabled: vad_enabled,
                        sid: sid,
                        input_audio_gain: input_audio_gain,
                        pitch: pitch,
                        index_rate: index_rate,
                        protect: protect,
                        volume_envelope: volume_envelope,
                        autotune: autotune,
                        autotune_strength: autotune_strength,
                        proposed_pitch: proposed_pitch,
                        proposed_pitch_threshold: proposed_pitch_threshold
                    })
                );
            };

            inputNode.port.onmessage = (e) => { // send audio from node to websocket for realtime
                const chunk = e.data && e.data.chunk;

                if (!chunk) return;
                if (ws.readyState === WebSocket.OPEN) ws.send(chunk); // Send raw audio
            };

            ws.onmessage = (ev) => { 
                // Read the string values ​​sent back from the websocket
                if (typeof ev.data === 'string') {
                    const msg = JSON.parse(ev.data);
                    // Show latency information in the status bar of the interface
                    if (msg.type === 'latency') setStatus(`Latency: ${msg.value.toFixed(1)} ms`, use_alert=false)
                    return;
                }
                // Send audio to the playback node to the audio device
                const ab = ev.data;
                playbackNode.port.postMessage({ chunk: ab }, [ab]);
            };

            ws.onclose = () => console.log("[WS] Closed!");
            window._workletNode = inputNode;
            window._playbackNode = playbackNode;

            if (window._audioCtx.state === 'suspended') await window._audioCtx.resume();

            console.log("Realtime is ready!");
            return ButtonState;
        } catch (err) {
            console.error("An error has occurred:", err);
            alert("An error has occurred:" + err.message);
            // stop realtime when error
            return StopAudioStream();
        }
    };

    window.StopAudioStream = async function() {
        try {
            if (window._ws) {
                window._ws.close();
                window._ws = null;
            }

            if (window._activeStream) {
                window._activeStream.getTracks().forEach(t => t.stop());
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

            document.querySelectorAll('audio').forEach(a => a.remove());
            setStatus("Stopped", use_alert=false);

            return {"start_button": true, "stop_button": false};
        } catch (e) {
            setStatus(`An error has occurred: ${e}`);

            return {"start_button": false, "stop_button": true}
        }
    };
}