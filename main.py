from fastapi import FastAPI, Security, Depends, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from gradio.routes import mount_gradio_app
import uvicorn
from app import Applio
from core import run_tts_script
import os

app = FastAPI(title="Applio API", version="1.0.0")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.environ.get("APPLIO_API_WRITE"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.post("/api/tts", dependencies=[Security(get_api_key)])
async def tts_endpoint(
    request: Request,
    text: str,
    voice: str,
    rate: int = 0,
    pitch: int = 0,
    filter_radius: int = 3,
    index_rate: float = 0.75,
    protect: float = 0.33,
    export_format: str = "wav"
):
    try:
        output_tts_path = os.path.join("assets", "tts", f"tts_{text[:10]}_{voice}.wav")
        output_rvc_path = os.path.join("assets", "tts", f"rvc_{text[:10]}_{voice}.{export_format}")

        message, output_path = run_tts_script(
            tts_file="",
            tts_text=text,
            tts_voice=voice,
            tts_rate=rate,
            pitch=pitch,
            filter_radius=filter_radius,
            index_rate=index_rate,
            volume_envelope=1,
            protect=protect,
            hop_length=128,
            f0_method="rmvpe",
            output_tts_path=output_tts_path,
            output_rvc_path=output_rvc_path,
            pth_path="",  # Will use default from config
            index_path="",  # Will use default from config
            split_audio=False,
            f0_autotune=False,
            f0_autotune_strength=0.0,
            clean_audio=False,
            clean_strength=0.0,
            export_format=export_format,
            f0_file="",
            embedder_model="hubert_base",
            sid=0
        )

        return JSONResponse({
            "status": "success",
            "message": message,
            "audio_path": output_path
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Mount Gradio app
gradio_app = Applio().launch(share=False, server_name="0.0.0.0", server_port=6969, prevent_thread_lock=True)
mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
