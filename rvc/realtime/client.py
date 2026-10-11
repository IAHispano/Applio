import os
import sys
import json
from fastapi import FastAPI, WebSocketDisconnect, WebSocket, Request
import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from .core import VoiceChanger, AUDIO_SAMPLE_RATE
from .pipeline import CudaGraphCaptureError
from .utils.chunks import automatic_block_frame
from .configuration import apply_config_update
from rvc.lib.user_config import resolve_logs_path

app = FastAPI()
vc_instance = None
params = {}
active_audio_ws = None


async def apply_message(ws, message):
    global params
    request_id = None
    try:
        if isinstance(message, str):
            message = json.loads(message)
        if not isinstance(message, dict):
            raise ValueError("Settings must be an object")
        request_id = message.get("id")
        if vc_instance is None:
            raise ValueError("Realtime pipeline not found")
        params = apply_config_update(vc_instance, params, message)
        await ws.send_text(json.dumps({"type": "config_applied", "id": request_id,
                                       "block_frame": vc_instance.block_frame}))
    except Exception as error:
        await ws.send_text(json.dumps({"type": "config_error", "id": request_id, "message": str(error)}))


@app.websocket("/change-config")
async def change_config(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await apply_message(ws, await ws.receive_text())
    except WebSocketDisconnect:
        pass


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
            "path": None,
        }

    if record_button == "Start":
        if not record_audio_path:
            record_audio_path = os.path.join(
                now_dir, "assets", "audios", "record_audio.wav"
            )

        vc_instance.record_audio = True
        vc_instance.record_audio_path = record_audio_path
        vc_instance.export_format = export_format
        vc_instance.setup_soundfile_record()

        return {
            "type": "info",
            "value": "Start recording...",
            "button": "Stop",
            "path": None,
        }
    else:
        vc_instance.record_audio = False
        vc_instance.record_audio_path = None
        vc_instance.soundfile = None

        return {
            "type": "info",
            "value": "Stop recording!",
            "button": "Start",
            "path": record_audio_path,
        }


@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    global vc_instance, params, active_audio_ws
    await ws.accept()
    if active_audio_ws is not None:
        await ws.send_text(
            json.dumps(
                {"type": "error", "message": "A realtime stream is already active."}
            )
        )
        await ws.close(code=1013)
        return
    active_audio_ws = ws

    print("[WS] Connected!")

    try:
        text = await ws.receive_text()
        params = json.loads(text)
        params["model_path"] = resolve_logs_path(params["model_path"])
        params["index_path"] = resolve_logs_path(params.get("index_path")) or ""

        block_frame = params["block_frame"]

        print("Starting Realtime...")

        if vc_instance is None:
            vc_instance = VoiceChanger(
                block_frame=block_frame,
                cross_fade_overlap_size=params["cross_fade_overlap_size"],
                extra_convert_size=params["extra_convert_size"],
                model_path=params["model_path"],
                index_path=params["index_path"],
                f0_method=params["f0_method"],
                embedder_model=params["embedder_model"],
                embedder_model_custom=params["embedder_model_custom"],
                silent_threshold=params["silent_threshold"],
                vad_enabled=params["vad_enabled"],
                vad_sensitivity=params.get("vad_sensitivity", 1),
                vad_frame_ms=30,
                sid=params["sid"],
                clean_audio=params["clean_audio"],
                clean_strength=params["clean_strength"],
                post_process=params["post_process"],
                **params["kwargs"],
            )

        vc_instance.vc_model.pipeline.vc.decoder_fp16 = (
            params.get("decoder_fp16", False)
            and vc_instance.vc_model.pipeline.vc.vocoder in ("HiFi-GAN", "MRF HiFi-GAN")
        )
        # Initialize model kernels before the browser starts capturing. Audio
        # sent during model loading would otherwise remain queued and stale.
        warmup_options = dict(
            f0_up_key=params.get("f0_up_key", 0),
            index_rate=params.get("index_rate", 0),
            protect=params.get("protect", 0.5),
            volume_envelope=params.get("volume_envelope", 1),
            f0_autotune=params.get("autotune", False),
            f0_autotune_strength=params.get("autotune_strength", 1),
            proposed_pitch=params.get("proposed_pitch", False),
            proposed_pitch_threshold=params.get("proposed_pitch_threshold", 155),
        )
        warmup_ms = vc_instance.warmup(**warmup_options)
        if params.get("automatic_block_size", False):
            # Retain headroom for device/OS jitter, and enlarge blocks on CPU,
            # AMD, or expensive model configurations before capturing audio.
            for _ in range(3):
                needed = automatic_block_frame(warmup_ms, AUDIO_SAMPLE_RATE)
                if needed <= block_frame:
                    break
                block_frame = needed
                vc_instance.block_frame = block_frame
                vc_instance.vc_model.realloc(
                    block_frame,
                    vc_instance.extra_frame,
                    vc_instance.crossfade_frame,
                    vc_instance.sola_search_frame,
                )
                vc_instance.generate_strength()
                warmup_ms = vc_instance.warmup(**warmup_options)
        vc_instance.last_processing_ms = warmup_ms
        params["block_frame"] = block_frame
        print("Realtime is ready!")
        await ws.send_text(json.dumps({"type": "ready", "block_frame": block_frame}))

        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                await apply_message(ws, message["text"])
                block_frame = vc_instance.block_frame
                continue
            audio = message.get("bytes")
            if audio is None:
                continue
            if vc_instance is None:
                return
            # The separate configuration endpoint can also resize the pipeline.
            block_frame = vc_instance.block_frame
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

            audio_output, vol, perf = vc_instance.on_request(
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

            vc_instance.last_processing_ms = perf[1]
            await ws.send_text(
                json.dumps({"type": "latency", "value": perf[1], "volume": vol})
            )
            await ws.send_bytes(audio_output.tobytes())
    except CudaGraphCaptureError:
        # PyTorch cannot reliably recover an invalid capture's allocator/RNG
        # state in-process. The API restarts this service with graphs disabled.
        try:
            await ws.send_text(json.dumps({"type": "retry_eager"}))
            await ws.close(code=1012)
        finally:
            os._exit(86)
    except WebSocketDisconnect:
        print("[WS] Disconnected!")
    finally:
        active_audio_ws = None
        if vc_instance is not None:
            del vc_instance
            vc_instance = None

        torch.cuda.empty_cache()

        try:
            await ws.close()
        except:
            pass
