import asyncio
import os
import sys
import argparse
import edge_tts


async def generate_tts(text: str, voice: str, rate: int, output_file: str):
    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"
    await edge_tts.Communicate(text, voice, rate=rates).save(output_file)


def run_tts_cli():
    # If legacy 5 positional args: tts.py <tts_file> <text> <voice> <rate> <output_file>
    if len(sys.argv) == 6 and not sys.argv[1].startswith("--"):
        tts_file = str(sys.argv[1])
        text = str(sys.argv[2])
        voice = str(sys.argv[3])
        rate = int(sys.argv[4])
        output_file = str(sys.argv[5])

        if tts_file and os.path.exists(tts_file):
            try:
                with open(tts_file, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                with open(tts_file, "r") as f:
                    text = f.read()

        asyncio.run(generate_tts(text, voice, rate, output_file))
        return

    parser = argparse.ArgumentParser(description="Applio TTS Generator and Converter")
    parser.add_argument("--tts-file", default="")
    parser.add_argument("--tts-text", default="")
    parser.add_argument("--tts-voice", default="en-US-AnaNeural")
    parser.add_argument("--tts-rate", type=int, default=0)
    parser.add_argument(
        "--output-tts-path", "--output-file", dest="output_tts_path", required=True
    )
    parser.add_argument("--output-rvc-path", default=None)
    parser.add_argument("--pth-path", "--model-path", dest="model_path", default=None)
    parser.add_argument("--index-path", default="")

    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--volume-envelope", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--export-format", default="WAV")
    parser.add_argument("--embedder-model", default="contentvec")
    parser.add_argument("--embedder-model-custom", default=None)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--split-audio", action="store_true", default=False)
    parser.add_argument("--f0-autotune", action="store_true", default=False)
    parser.add_argument("--f0-autotune-strength", type=float, default=1.0)
    parser.add_argument("--proposed-pitch", action="store_true", default=False)
    parser.add_argument("--proposed-pitch-threshold", type=float, default=155.0)
    parser.add_argument("--clean-audio", action="store_true", default=False)
    parser.add_argument("--clean-strength", type=float, default=0.5)

    args = parser.parse_args()

    text = args.tts_text
    if args.tts_file and os.path.exists(args.tts_file):
        try:
            with open(args.tts_file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            with open(args.tts_file, "r") as f:
                text = f.read()

    if not text.strip():
        print("Error: No text provided for TTS synthesis.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_tts_path)), exist_ok=True)
    asyncio.run(generate_tts(text, args.tts_voice, args.tts_rate, args.output_tts_path))
    print(f"TTS audio generated at '{args.output_tts_path}'")

    if args.output_rvc_path and args.model_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_rvc_path)), exist_ok=True
        )
        now_dir = os.getcwd()
        if now_dir not in sys.path:
            sys.path.append(now_dir)
        from rvc.infer.infer import VoiceConverter

        vc = VoiceConverter()
        vc.convert_audio(
            audio_input_path=args.output_tts_path,
            audio_output_path=args.output_rvc_path,
            model_path=args.model_path,
            index_path=args.index_path,
            pitch=args.pitch,
            index_rate=args.index_rate,
            volume_envelope=args.volume_envelope,
            protect=args.protect,
            f0_method=args.f0_method,
            export_format=args.export_format,
            embedder_model=args.embedder_model,
            embedder_model_custom=args.embedder_model_custom,
            sid=args.sid,
            split_audio=args.split_audio,
            f0_autotune=args.f0_autotune,
            f0_autotune_strength=args.f0_autotune_strength,
            proposed_pitch=args.proposed_pitch,
            proposed_pitch_threshold=args.proposed_pitch_threshold,
            clean_audio=args.clean_audio,
            clean_strength=args.clean_strength,
        )
        print(f"RVC conversion completed at '{args.output_rvc_path}'")


if __name__ == "__main__":
    run_tts_cli()
