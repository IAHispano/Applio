import torch


def model_information(path):
    model_data = torch.load(path, map_location="cpu")

    print(f"Loaded model from {path}")

    data = model_data

    epochs = data.get("info", "None")
    sr = data.get("sr", "None")
    f0 = data.get("f0", "None")
    version = data.get("version", "None")

    return f"Epochs: {epochs}\nSampling rate: {sr}\nPitch guidance: {f0}\nVersion: {version}"
