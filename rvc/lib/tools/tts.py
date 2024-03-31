import edge_tts


def tts_pipeline(text, voice, output_file):
    edge_tts.Communicate(text, voice).save(output_file)
    print(f"TTS with {voice} completed. Output TTS file: '{output_file}'")