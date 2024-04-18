import torch
from datetime import datetime


def prettify_date(date_str):
    if date_str is None:
        return "None"
    try:
        date_time_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        return date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Invalid date format"



def model_information(path):
    model_data = torch.load(path, map_location="cpu")

    print(f"Loaded model from {path}")

    epochs = model_data.get("epoch", "None")
    steps = model_data.get("step", "None")
    sr = model_data.get("sr", "None")
    f0 = model_data.get("f0", "None")
    version = model_data.get("version", "None")
    creation_date = model_data.get(
        "creation_date", "None"
    )
    model_hash = model_data.get("model_hash", None)

    pitch_guidance = "True" if f0 == 1 else "False"

    creation_date_str = (
        prettify_date(creation_date) if creation_date else "None"
    )

    return (
        f"Epochs: {epochs}\n"
        f"Steps: {steps}\n"
        f"RVC Version: {version}\n"
        f"Sampling Rate: {sr}\n"
        f"Pitch Guidance: {pitch_guidance}\n"
        f"Creation Date: {creation_date_str}\n"
        f"Hash (ID): {model_hash}"
    )
