import os
import sys
import json
from fastapi import FastAPI, WebSocketDisconnect, WebSocket, Request
import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from .core import VoiceChanger, AUDIO_SAMPLE_RATE

app = FastAPI()
vc_instance = None
params = {}


@app.websocket("/change-config")
async def change_config(ws: WebSocket):
    global vc_instance, params

    if vc_instance is None:
        return

    await ws.accept()

    text = await ws.receive_text()
    jsons = json.loads(text)

    if jsons["if_kwargs"]:
        params["kwargs"][jsons["key"]] = jsons["value"]
    else:
        params[jsons["key"]] = jsons["value"]

    crossfade_frame = int(params.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE)
    extra_frame = int(params.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE)

    if (
        vc_instance.crossfade_frame != crossfade_frame or
        vc_instance.extra_frame != extra_frame
    ):
        del (
            vc_instance.vc_model.audio_buffer,
            vc_instance.vc_model.convert_buffer,
            vc_instance.vc_model.pitch_buffer,
            vc_instance.vc_model.pitchf_buffer,
        )
        del (
            vc_instance.fade_in_window,
            vc_instance.fade_out_window,
            vc_instance.sola_buffer
        )

        vc_instance.vc_model.realloc(
            vc_instance.block_frame,
            vc_instance.extra_frame,
            vc_instance.crossfade_frame,
            vc_instance.sola_search_frame,
        )
        vc_instance.generate_strength()

    vc_instance.vc_model.input_sensitivity = 10 ** (params.get("silent_threshold", -90) / 20)

    vad_enabled = params.get("vad_enabled", True)
    if vad_enabled is False:
        vc_instance.vc_model.vad = None
    elif vad_enabled and vc_instance.vc_model.vad is None:
        from rvc.realtime.utils.vad import VADProcessor

        vc_instance.vc_model.vad = VADProcessor(
            sensitivity_mode=3,
            sample_rate=vc_instance.vc_model.sample_rate,
            frame_duration_ms=30,  
        )

    # The VAD parameters have been assigned by default.
    # if vc_instance.vc_model.vad is not None:
    #     vc_instance.vc_model.vad.vad.set_mode(vad_sensitivity)
    #     vc_instance.vc_model.vad.frame_length = int(vc_instance.vc_model.sample_rate * (vad_frame_ms / 1000.0))

    clean_audio = params.get("clean_audio", False)
    clean_strength = params.get("clean_strength", 0.5)

    if clean_audio is False:
        vc_instance.vc_model.reduced_noise = None
    elif clean_audio and vc_instance.vc_model.reduced_noise is None:
        from noisereduce.torchgate import TorchGate

        vc_instance.vc_model.reduced_noise = (
            TorchGate(
                vc_instance.vc_model.pipeline.tgt_sr,
                prop_decrease=clean_strength,
            ).to(vc_instance.vc_model.device)
        )

    if vc_instance.vc_model.reduced_noise is not None:
        vc_instance.vc_model.reduced_noise.prop_decrease = clean_strength

    post_process = params.get("post_process", False)
    kwargs = params.get("kwargs", {})

    if post_process is False:
        vc_instance.vc_model.board = None
        vc_instance.vc_model.kwargs = None
    elif post_process and vc_instance.vc_model.kwargs != kwargs:
        # Post-process requires creating a new pendalboard.
        new_board = vc_instance.vc_model.setup_pedalboard(**kwargs)
        vc_instance.vc_model.board = new_board
        vc_instance.vc_model.kwargs = kwargs.copy()

    model_pth = params.get("model_path", vc_instance.vc_model.model_path)
    if model_pth and vc_instance.vc_model.model_path != model_pth:
        vc_instance.vc_model.model_path = model_pth
        vc_instance.vc_model.pipeline.vc.load_model(model_pth)
        vc_instance.vc_model.pipeline.vc.setup_network()
        # Set a new version, otherwise it will crash.
        vc_instance.vc_model.pipeline.version = vc_instance.vc_model.pipeline.vc.version

    sid = params.get("sid", vc_instance.vc_model.pipeline.sid)
    if vc_instance.vc_model.pipeline.sid != sid:
        import torch
        # This is for multi-SID models.
        vc_instance.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc_instance.vc_model.pipeline.device, dtype=torch.int64
        )

    index_path = params.get("index_path", None)
    if index_path and vc_instance.vc_model.index_path != index_path:
        from rvc.realtime.pipeline import load_faiss_index

        index, big_npy = load_faiss_index(
            index_path.strip()
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip()
            .replace("trained", "added")
        )

        vc_instance.vc_model.pipeline.index = index
        vc_instance.vc_model.pipeline.big_npy = big_npy
        vc_instance.vc_model.index_path = index_path
    else:
        vc_instance.vc_model.pipeline.index = None
        vc_instance.vc_model.pipeline.big_npy = None
        vc_instance.vc_model.index_path = None

    f0_method = params.get("f0_method", vc_instance.vc_model.pipeline.f0_method)
    if vc_instance.vc_model.pipeline.f0_method != f0_method:
        f0_model = vc_instance.vc_model.pipeline.setup_f0(f0_method)
        vc_instance.vc_model.pipeline.f0_model = f0_model
        vc_instance.vc_model.pipeline.f0_method = f0_method

    embedder_model = params.get("embedder_model", vc_instance.vc_model.embedder_model)
    embedder_model_custom = params.get("embedder_model_custom", vc_instance.vc_model.embedder_model_custom)

    if (
        vc_instance.vc_model.embedder_model != embedder_model or
        vc_instance.vc_model.embedder_model_custom != embedder_model_custom
    ):
        from rvc.lib.utils import load_embedding

        hubert_model = load_embedding(embedder_model, embedder_model_custom)
        hubert_model = hubert_model.to(vc_instance.device).float()
        hubert_model.eval()

        vc_instance.vc_model.pipeline.hubert_model = hubert_model
        vc_instance.vc_model.embedder_model = embedder_model
        vc_instance.vc_model.embedder_model_custom = embedder_model_custom

@app.post("/record")
async def record(request: Request):
    global vc_instance

    data = await request.json()
    record_button = data.get("record_button", "Stop")
    record_audio_path = data.get("record_audio_path", None)
    export_format = data.get("export_format", "WAV")

    if vc_instance is None:
        return {
            "type": "warnings",
            "value": "Realtime pipeline not found!",
            "button": "Start",
            "path": None
        }

    if record_button == "Start":
        if not record_audio_path:
            record_audio_path = os.path.join(now_dir, "assets", "audios", "record_audio.wav")

        vc_instance.record_audio = True
        vc_instance.record_audio_path = record_audio_path
        vc_instance.export_format = export_format
        vc_instance.setup_soundfile_record()

        return {
            "type": "info",
            "value": "Start recording...",
            "button": "Stop",
            "path": None
        }
    else:
        vc_instance.record_audio = False
        vc_instance.record_audio_path = None
        vc_instance.soundfile = None

        return {
            "type": "info",
            "value": "Stop recording!",
            "button": "Start",
            "path": record_audio_path
        }

@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    global vc_instance, params
    await ws.accept()

    print("[WS] Connected!")

    try:
        text = await ws.receive_text()
        params = json.loads(text)

        read_chunk_size = int(params["chunk_size"])
        block_frame = read_chunk_size * 128

        print("Starting Realtime...")

        if vc_instance is None:
            vc_instance = VoiceChanger(
                read_chunk_size=read_chunk_size,
                cross_fade_overlap_size=params["cross_fade_overlap_size"],
                extra_convert_size=params["extra_convert_size"],
                model_path=params["model_path"],
                index_path=str(params["index_path"]),
                f0_method=params["f0_method"],
                embedder_model=params["embedder_model"],
                embedder_model_custom=params["embedder_model_custom"],
                silent_threshold=params["silent_threshold"],
                vad_enabled=params["vad_enabled"],
                vad_sensitivity=3,
                vad_frame_ms=30,
                sid=params["sid"],
                clean_audio=params["clean_audio"],
                clean_strength=params["clean_strength"],
                post_process=params["post_process"],
                **params["kwargs"]
            )

        print("Realtime is ready!")

        while True:
            audio = await ws.receive_bytes()
            arr = np.frombuffer(audio, dtype=np.float32)

            if arr.size != block_frame:
                arr = (
                    np.pad(arr, (0, block_frame - arr.size)).astype(np.float32)
                    if arr.size < block_frame
                    else arr[:block_frame].astype(np.float32)
                )

            if vc_instance is None:
                # Avoid errors when disconnecting.
                return

            audio_output, _, perf = vc_instance.on_request(
                arr * (params["input_audio_gain"] / 100.0),
                f0_up_key=params["f0_up_key"],
                index_rate=params["index_rate"],
                protect=params["protect"],
                volume_envelope=params["volume_envelope"],
                f0_autotune=params["autotune"],
                f0_autotune_strength=params["autotune_strength"],
                proposed_pitch=params["proposed_pitch"],
                proposed_pitch_threshold=params["proposed_pitch_threshold"],
            )

            await ws.send_text(json.dumps({"type": "latency", "value": perf[1]}))
            await ws.send_bytes(audio_output.tobytes())
    except WebSocketDisconnect:
        print("[WS] Disconnected!")
    finally:
        if vc_instance is not None:
            del vc_instance
            vc_instance = None

        torch.cuda.empty_cache()

        try:
            await ws.close()
        except:
            pass
