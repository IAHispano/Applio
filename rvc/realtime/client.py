import os
import sys
import json
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
import numpy as np
import torch

sys.path.append(os.getcwd())

from .core import VoiceChanger

app = FastAPI()
vc_instance = None

@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    global vc_instance
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
                model_path=params["model_file"], 
                index_path=str(params["index_file"]), 
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
                arr = np.pad(arr, (0, block_frame - arr.size)).astype(np.float32) if arr.size < block_frame else arr[:block_frame].astype(np.float32)

            audio_output, _, perf = vc_instance.on_request(
                arr * (params["input_audio_gain"] / 100.0), 
                f0_up_key=params["pitch"], 
                index_rate=params["index_rate"], 
                protect=params["protect"], 
                volume_envelope=params["volume_envelope"], 
                f0_autotune=params["autotune"], 
                f0_autotune_strength=params["autotune_strength"], 
                proposed_pitch=params["proposed_pitch"], 
                proposed_pitch_threshold=params["proposed_pitch_threshold"]
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