import os


def pretrained_selector(vocoder, sample_rate):
    base_path = os.path.join("rvc", "models", "pretraineds", f"{vocoder.lower()}")

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""


def rectified_flow_selector(embedder_model):
    base_path = os.path.join("rvc", "models", "pretraineds", "rectified-flow")

    embedder_name = str(embedder_model).replace("-", "_")
    path_flow = os.path.join(base_path, f"pretrain_flow_{embedder_name}.pth")
    if not os.path.exists(path_flow):
        path_flow = ""

    # OpenVPI NSF-HiFiGAN, renders the mel of the flow
    path_vocoder = ""
    if os.path.isdir(base_path):
        for name in sorted(os.listdir(base_path)):
            if name.endswith(".ckpt") or "_vocoder" in name:
                path_vocoder = os.path.join(base_path, name)
                break

    return path_flow, path_vocoder
