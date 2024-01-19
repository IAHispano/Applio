import argparse
import os 
import json


def validate_sampling_rate(value):
    valid_sampling = [
        "32000",
        "40000",
        "48000",
    ]
    if value in valid_sampling:
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid sampling_rate. Please choose from {valid_sampling} not {value}"
        )


def validate_f0up_key(value):
    f0up_key = int(value)
    if -12 <= f0up_key <= 12:
        return f0up_key
    else:
        raise argparse.ArgumentTypeError(f"f0up_key must be in the range of -12 to +12")

def validate_true_false(value):
    valid_tf = [
        "True",
        "False",
    ]
    if value in valid_tf:
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid true_false. Please choose from {valid_tf} not {value}"
        )

def validate_f0method(value):
    valid_f0methods = [
        "pm",
        "dio",
        "crepe",
        "crepe-tiny",
        "harvest",
        "rmvpe",
    ]
    if value in valid_f0methods:
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid f0method. Please choose from {valid_f0methods} not {value}"
        )

def validate_tts_voices(value):
    json_path = os.path.join("rvc", "lib", "tools", "tts_voices.json")
    with open(json_path, 'r') as file:
        tts_voices_data = json.load(file)

    # Extrae los valores de "ShortName" del JSON
    short_names = [voice.get("ShortName", "") for voice in tts_voices_data]
    if value in short_names:
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid voice. Please choose from {short_names} not {value}"
        )