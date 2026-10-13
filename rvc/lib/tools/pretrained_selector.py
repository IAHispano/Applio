import os


def pretrained_selector(vocoder, sample_rate):
    base_path = os.path.join("rvc", "models", "pretraineds", f"{vocoder.lower()}")

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""


# Vocoder of Rectified Flow when none is picked
rectified_flow_vocoder = "nsf_pupu_small_44.1k_hop512_128bin_vocoder.pth"


def rectified_flow_selector(embedder_model):
    base_path = os.path.join("rvc", "models", "pretraineds", "rectified-flow")

    embedder_name = str(embedder_model).replace("-", "_")
    path_flow = os.path.join(base_path, f"pretrain_flow_{embedder_name}.pth")
    if not os.path.exists(path_flow):
        path_flow = ""

    # Pupu-Vocoder, renders the mel of the flow: the default one, else the first
    path_vocoder = ""
    if os.path.isdir(base_path):
        names = sorted(name for name in os.listdir(base_path) if "_vocoder" in name)
        if rectified_flow_vocoder in names:
            path_vocoder = os.path.join(base_path, rectified_flow_vocoder)
        elif names:
            path_vocoder = os.path.join(base_path, names[0])

    return path_flow, path_vocoder
