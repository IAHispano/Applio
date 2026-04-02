import asyncio
import os
import sys
import edge_tts

async def main():
    tts_file = str(sys.argv[1])
    text = str(sys.argv[2])
    voice = str(sys.argv[3])
    rate = int(sys.argv[4])
    output_file = str(sys.argv[5])

    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"

    if tts_file and os.path.exists(tts_file):
        try:
            with open(tts_file, "r", encoding="utf-8") as file:
                text = file.read()
        except:
            with open(tts_file, "r") as file:
                text = file.read()

    try:
        await edge_tts.Communicate(text, voice, rate=rates).save(output_file)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
