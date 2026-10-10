import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import traceback

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

# Keep a reference to real standard streams before redirection
real_stdout = sys.stdout
real_stderr = sys.stderr

active_job_id = None


def get_active_job_id():
    return active_job_id


class IPCStdoutRedirector:
    def __init__(self, target, job_id_fn):
        self.target = target
        self.job_id_fn = job_id_fn
        self.buffer = ""

    def write(self, s):
        if not s:
            return
        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                send_ipc(
                    {
                        "type": "log",
                        "id": self.job_id_fn(),
                        "message": line,
                    }
                )

    def flush(self):
        if self.buffer.strip():
            send_ipc(
                {
                    "type": "log",
                    "id": self.job_id_fn(),
                    "message": self.buffer.strip(),
                }
            )
        self.buffer = ""


def send_ipc(data: dict):
    payload = {"_applio_ipc": True, **data}
    real_stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    real_stdout.flush()


def to_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)


def map_params(params: dict, input_path: str, output_path: str) -> dict:
    return {
        "pitch": int(params.get("pitch", 0)),
        "index_rate": float(params.get("indexRate", 0.75)),
        "volume_envelope": float(params.get("volumeEnvelope", 1.0)),
        "protect": float(params.get("protect", 0.5)),
        "f0_method": str(params.get("f0Method", "rmvpe")),
        "input_path": input_path,
        "output_path": output_path,
        "pth_path": params.get("pthPath", ""),
        "index_path": params.get("indexPath", "") or "",
        "split_audio": to_bool(params.get("splitAudio", False)),
        "f0_autotune": to_bool(params.get("f0Autotune", False)),
        "f0_autotune_strength": float(params.get("f0AutotuneStrength", 1.0)),
        "proposed_pitch": to_bool(params.get("proposedPitch", False)),
        "proposed_pitch_threshold": float(params.get("proposedPitchThreshold", 155.0)),
        "clean_audio": to_bool(params.get("cleanAudio", False)),
        "clean_strength": float(params.get("cleanStrength", 0.5)),
        "export_format": str(params.get("exportFormat", "WAV")),
        "embedder_model": str(params.get("embedderModel", "contentvec")),
        "embedder_model_custom": params.get("embedderModelCustom") or None,
        "formant_shifting": to_bool(params.get("formantShifting", False)),
        "formant_qfrency": float(params.get("formantQfrency", 1.0)),
        "formant_timbre": float(params.get("formantTimbre", 1.0)),
        "post_process": to_bool(params.get("postProcess", False)),
        "reverb": to_bool(params.get("reverb", False)),
        "pitch_shift": to_bool(params.get("pitchShift", False)),
        "limiter": to_bool(params.get("limiter", False)),
        "gain": to_bool(params.get("gain", False)),
        "distortion": to_bool(params.get("distortion", False)),
        "chorus": to_bool(params.get("chorus", False)),
        "bitcrush": to_bool(params.get("bitcrush", False)),
        "clipping": to_bool(params.get("clipping", False)),
        "compressor": to_bool(params.get("compressor", False)),
        "delay": to_bool(params.get("delay", False)),
        "reverb_room_size": float(params.get("reverbRoomSize", 0.5)),
        "reverb_damping": float(params.get("reverbDamping", 0.5)),
        "reverb_wet_gain": float(params.get("reverbWetGain", 0.33)),
        "reverb_dry_gain": float(params.get("reverbDryGain", 0.4)),
        "reverb_width": float(params.get("reverbWidth", 1.0)),
        "reverb_freeze_mode": float(params.get("reverbFreezeMode", 0.0)),
        "pitch_shift_semitones": float(params.get("pitchShiftSemitones", 0.0)),
        "limiter_threshold": float(params.get("limiterThreshold", -6.0)),
        "limiter_release_time": float(params.get("limiterReleaseTime", 0.05)),
        "gain_db": float(params.get("gainDb", 0.0)),
        "distortion_gain": float(params.get("distortionGain", 25.0)),
        "chorus_rate": float(params.get("chorusRate", 1.0)),
        "chorus_depth": float(params.get("chorusDepth", 0.25)),
        "chorus_center_delay": float(params.get("chorusCenterDelay", 7.0)),
        "chorus_feedback": float(params.get("chorusFeedback", 0.0)),
        "chorus_mix": float(params.get("chorusMix", 0.5)),
        "bitcrush_bit_depth": int(params.get("bitcrushBitDepth", 8)),
        "clipping_threshold": float(params.get("clippingThreshold", -6.0)),
        "compressor_threshold": float(params.get("compressorThreshold", 0.0)),
        "compressor_ratio": float(params.get("compressorRatio", 1.0)),
        "compressor_attack": float(params.get("compressorAttack", 1.0)),
        "compressor_release": float(params.get("compressorRelease", 100.0)),
        "delay_seconds": float(params.get("delaySeconds", 0.5)),
        "delay_feedback": float(params.get("delayFeedback", 0.0)),
        "delay_mix": float(params.get("delayMix", 0.5)),
        "sid": int(params.get("sid", 0)),
    }


def run_worker():
    global active_job_id

    # Install redirectors
    sys.stdout = IPCStdoutRedirector(real_stdout, get_active_job_id)
    sys.stderr = IPCStdoutRedirector(real_stdout, get_active_job_id)

    from rvc.infer.infer import VoiceConverter

    # Preload singleton VoiceConverter in background
    vc = VoiceConverter()

    send_ipc({"type": "ready", "pid": os.getpid()})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            send_ipc({"type": "error", "id": None, "error": f"Invalid JSON: {e}"})
            continue

        cmd = req.get("command")
        job_id = req.get("id")
        active_job_id = job_id

        try:
            if cmd == "ping":
                send_ipc({"type": "pong", "id": job_id})
            elif cmd == "warmup":
                # Warm up CUDA, PyTorch, audio libraries, embedder, and the most recent model
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.init()

                    # Pre-import heavy numerical and DSP libraries
                    import librosa
                    import soundfile

                    vc.load_hubert("contentvec", None)

                    # Look for the latest model in logs/ to preload so the first user conversion is instant
                    preloaded_name = None
                    from rvc.lib.user_config import get_logs_dir

                    logs_dir = get_logs_dir()
                    if os.path.isdir(logs_dir):
                        candidates = []
                        for r, _, files in os.walk(logs_dir):
                            for f in files:
                                if f.lower().endswith(".pth") and not f.startswith(
                                    ("G_", "D_")
                                ):
                                    full_p = os.path.join(r, f)
                                    try:
                                        candidates.append(
                                            (os.path.getmtime(full_p), full_p)
                                        )
                                    except Exception:
                                        pass
                        if candidates:
                            candidates.sort(reverse=True)
                            latest_model = candidates[0][1]
                            try:
                                vc.get_vc(latest_model, 0)
                                preloaded_name = os.path.basename(latest_model)
                            except Exception:
                                pass

                    msg = f"Worker warmed up (embedder: contentvec" + (
                        f", active model: {preloaded_name})" if preloaded_name else ")"
                    )
                    send_ipc({"type": "log", "id": job_id, "message": msg})
                except Exception as e:
                    send_ipc(
                        {"type": "log", "id": job_id, "message": f"Warmup notice: {e}"}
                    )
            elif cmd == "unload":
                vc.cleanup_model()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                send_ipc({"type": "unloaded", "id": job_id})
            elif cmd == "infer":
                params = req.get("params", {})
                input_path = req.get("inputPath")
                output_path = req.get("outputPath")
                kwargs = map_params(params, input_path, output_path)

                export_format = kwargs.get("export_format", "WAV")
                kwargs["audio_input_path"] = kwargs.pop("input_path", input_path)
                kwargs["audio_output_path"] = kwargs.pop("output_path", output_path)
                kwargs["model_path"] = kwargs.pop("pth_path", "")

                vc.convert_audio(**kwargs)
                final_out = output_path.replace(".wav", f".{export_format.lower()}")
                info_msg = f"File {input_path} inferred successfully."
                sys.stdout.flush()
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "outputPath": final_out,
                        "info": info_msg,
                    }
                )
            elif cmd == "infer_batch":
                params = req.get("params", {})
                input_folder = req.get("inputFolder")
                output_folder = req.get("outputFolder")
                kwargs = map_params(params, "", "")
                kwargs.pop("input_path", None)
                kwargs.pop("output_path", None)
                kwargs["model_path"] = kwargs.pop("pth_path", "")

                vc.convert_audio_batch(
                    audio_input_paths=input_folder,
                    audio_output_path=output_folder,
                    **kwargs,
                )
                sys.stdout.flush()
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "info": f"Batch conversion in {input_folder} completed.",
                    }
                )
            elif cmd == "tts":
                import asyncio
                import edge_tts

                tts_text = req.get("ttsText", "")
                tts_file = req.get("ttsFile", "")
                tts_voice = req.get("ttsVoice", "en-US-AnaNeural")
                tts_rate = int(req.get("ttsRate", 0))
                output_tts_path = req.get("outputTtsPath")
                output_rvc_path = req.get("outputRvcPath")
                params = req.get("params", {})

                if tts_file and os.path.exists(tts_file):
                    try:
                        with open(tts_file, "r", encoding="utf-8") as f:
                            tts_text = f.read()
                    except Exception:
                        with open(tts_file, "r") as f:
                            tts_text = f.read()

                if not tts_text.strip():
                    raise ValueError("No text provided for TTS synthesis.")

                os.makedirs(
                    os.path.dirname(os.path.abspath(output_tts_path)), exist_ok=True
                )
                rate_str = f"+{tts_rate}%" if tts_rate >= 0 else f"{tts_rate}%"
                asyncio.run(
                    edge_tts.Communicate(tts_text, tts_voice, rate=rate_str).save(
                        output_tts_path
                    )
                )
                print(f"TTS audio generated at '{output_tts_path}'")

                final_out = output_tts_path
                if output_rvc_path and params.get("pthPath"):
                    os.makedirs(
                        os.path.dirname(os.path.abspath(output_rvc_path)), exist_ok=True
                    )
                    kwargs = map_params(params, output_tts_path, output_rvc_path)
                    kwargs["audio_input_path"] = output_tts_path
                    kwargs["audio_output_path"] = output_rvc_path
                    kwargs["model_path"] = kwargs.pop("pth_path", "")
                    vc.convert_audio(**kwargs)
                    export_format = kwargs.get("export_format", "WAV")
                    final_out = output_rvc_path.replace(
                        ".wav", f".{export_format.lower()}"
                    )
                    print(f"TTS RVC conversion completed at '{final_out}'")

                sys.stdout.flush()
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "outputTtsPath": output_tts_path,
                        "outputRvcPath": final_out if output_rvc_path else None,
                        "outputPath": final_out,
                        "info": "TTS synthesis completed.",
                    }
                )
            elif cmd == "preload_model":
                pth_path = req.get("pthPath")
                sid = int(req.get("sid", 0))
                if pth_path and os.path.isfile(pth_path):
                    vc.get_vc(pth_path, sid)
                    send_ipc(
                        {
                            "type": "done",
                            "id": job_id,
                            "success": True,
                            "info": f"Model '{os.path.basename(pth_path)}' preloaded.",
                        }
                    )
                else:
                    send_ipc(
                        {
                            "type": "error",
                            "id": job_id,
                            "error": f"Model file not found: {pth_path}",
                        }
                    )
            elif cmd == "inspect_model":
                pth_path = req.get("pthPath")
                if not pth_path or not os.path.isfile(pth_path):
                    raise ValueError(f"Model file not found: {pth_path}")
                from rvc.lib.tools.inspect_checkpoint import inspect_checkpoint

                meta = inspect_checkpoint(pth_path)
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "metadata": meta,
                    }
                )
            elif cmd == "analyze_audio":
                input_path = req.get("inputPath")
                plot_path = req.get("plotPath")
                if not input_path or not os.path.isfile(input_path):
                    raise ValueError(f"Audio file not found: {input_path}")
                from rvc.lib.tools.analyzer import analyze_audio

                os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
                info, plot = analyze_audio(input_path, plot_path)
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "info": info,
                        "plot": plot,
                    }
                )
            elif cmd == "f0_curve":
                input_path = req.get("inputPath")
                method = req.get("method", "rmvpe")
                output_image = req.get("outputImage")
                output_txt = req.get("outputTxt")
                if not input_path or not os.path.isfile(input_path):
                    raise ValueError(f"Audio file not found: {input_path}")
                from rvc.lib.tools.f0_curve import extract_f0_curve

                os.makedirs(
                    os.path.dirname(os.path.abspath(output_image)), exist_ok=True
                )
                os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)
                img, txt = extract_f0_curve(
                    input_path, method, output_image, output_txt
                )
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "outputImage": img,
                        "outputTxt": txt,
                    }
                )
            elif cmd == "model_blender":
                model_name = req.get("modelName")
                pth1 = req.get("pth1")
                pth2 = req.get("pth2")
                ratio = float(req.get("ratio", 0.5))
                from rvc.train.process.model_blender import model_blender

                r = model_blender(model_name, pth1, pth2, ratio)
                msg, f = r if isinstance(r, tuple) else (str(r), None)
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "message": msg,
                        "file": f,
                    }
                )
            else:
                send_ipc(
                    {
                        "type": "error",
                        "id": job_id,
                        "error": f"Unknown command: {cmd}",
                    }
                )
        except Exception as e:
            sys.stdout.flush()
            tb = traceback.format_exc()
            send_ipc(
                {
                    "type": "error",
                    "id": job_id,
                    "error": str(e),
                    "traceback": tb,
                }
            )
        finally:
            active_job_id = None


if __name__ == "__main__":
    run_worker()
