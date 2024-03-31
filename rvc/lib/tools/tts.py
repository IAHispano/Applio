import sys
import asyncio
import edge_tts


async def main():
    text = sys.argv[1]
    voice = sys.argv[2]
    output_file = sys.argv[3]

    await edge_tts.Communicate(text, voice).save(output_file)
    print(f"TTS with {voice} completed. Output TTS file: '{output_file}'")


if __name__ == "__main__":
    asyncio.run(main())