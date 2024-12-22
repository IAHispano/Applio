import os

def pretrained_selector(version, vocoder, pitch_guidance, sample_rate):
    base_path = os.path.join("rvc", "models", "pretraineds", f"pretrained_{version}")
    f0 = "f0" if pitch_guidance else ""

    if vocoder == "HiFi-GAN":
        vocoder_path = ""
    elif vocoder == "MRF HiFi-GAN":
        vocoder_path = "HiFiGAN_"
    elif vocoder == "RefineGAN":
        vocoder_path = "RefineGAN_"
    else:
        vocoder_path = ""

    path_g = os.path.join(base_path, f"{vocoder_path}{f0}G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"{vocoder_path}{f0}D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""