from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from core import run_tts_script
from tabs.inference.inference import (
    extract_model_and_epoch,
    names,
    match_index,
    get_speakers_id,
)

# Initialize FastAPI app
app = FastAPI()

# Default values
default_voice_name = "ur-PK-AsadNeural"
default_model_file = sorted(names, key=lambda x: extract_model_and_epoch(x))[0]
default_index_file = match_index(default_model_file)
default_sid = get_speakers_id(default_model_file)[0] if get_speakers_id(default_model_file) else 0
output_tts_path = os.path.join(os.getcwd(), "assets", "audios", "tts_output.wav")
output_rvc_path = os.path.join(os.getcwd(), "assets", "audios", "tts_rvc_output.wav")

# Input model
class TTSRequest(BaseModel):
    tts_text: str  # Only text is required


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        # Run the TTS script with default parameters
        output_info, audio_file_path = run_tts_script(
            tts_file=None,
            tts_text=request.tts_text,
            tts_voice=default_voice_name,
            tts_rate=0,  # Default TTS speed
            pitch=0,  # Default pitch
            filter_radius=3,
            index_rate=0.75,
            volume_envelope=1,
            protect=0.5,
            hop_length=128,
            f0_method="rmvpe",
            output_tts_path=output_tts_path,
            output_rvc_path=output_rvc_path,
            pth_path=default_model_file,
            index_path=default_index_file,
            split_audio=False,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            clean_audio=True,
            clean_strength=0.5,
            export_format="WAV",
            f0_file=None,
            embedder_model="contentvec",
            embedder_model_custom=None,
            sid=default_sid,
        )
        return {
            "message": "TTS conversion successful",
            "output_info": output_info,
            "audio_file_path": audio_file_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during TTS conversion: {str(e)}")
