"""Compatibility entry point for callers using the standalone selector."""

import os


def pretrained_selector(vocoder, sample_rate):
    from rvc.lib.tools.prerequisites_download import ensure_pretrained

    return ensure_pretrained(vocoder, int(sample_rate))


# Vocoder of Rectified Flow at each sampling rate when none is picked
rectified_flow_vocoders = {
    32000: "chouwagan_32k_hop320_128bin_vocoder.pth",
    40000: "chouwagan_40k_hop400_128bin_vocoder.pth",
    44100: "chouwagan_44.1k_hop512_128bin_vocoder.pth",
    48000: "chouwagan_48k_hop480_128bin_vocoder.pth",
}


def rectified_flow_selector(embedder_model, sample_rate=44100):
    base_path = os.path.join("rvc", "models", "pretraineds", "rectified-flow")
    sample_rate = int(sample_rate)

    # The 44100 Hz pretrain predates the others and keeps its name
    embedder_name = str(embedder_model).replace("-", "_")
    suffix = "" if sample_rate == 44100 else f"_{str(sample_rate)[:2]}k"
    path_flow = os.path.join(base_path, f"pretrain_flow_{embedder_name}{suffix}.pth")
    if not os.path.exists(path_flow):
        path_flow = ""

    # ChouwaGAN, renders the mel of the flow: the default one of the sampling
    # rate, else the first named for that rate, else the first
    default_vocoder = rectified_flow_vocoders.get(sample_rate, "")
    rate_name = "_44.1k_" if sample_rate == 44100 else f"_{str(sample_rate)[:2]}k_"
    path_vocoder = ""
    if os.path.isdir(base_path):
        names = sorted(name for name in os.listdir(base_path) if "_vocoder" in name)
        same_rate = [name for name in names if rate_name in name]
        if default_vocoder in names:
            path_vocoder = os.path.join(base_path, default_vocoder)
        elif same_rate or names:
            path_vocoder = os.path.join(base_path, (same_rate or names)[0])

    return path_flow, path_vocoder
