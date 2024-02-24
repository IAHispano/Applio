import torch
from collections import OrderedDict


def extract(ckpt):
    model = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {key: value for key, value in model.items() if "enc_q" not in key}
    return opt


def model_fusion(model_name, pth_path_1, pth_path_2):
    ckpt1 = torch.load(pth_path_1, map_location="cpu")
    ckpt2 = torch.load(pth_path_2, map_location="cpu")
    if "model" in ckpt1:
        ckpt1 = extract(ckpt1)
    else:
        ckpt1 = ckpt1["weight"]
    if "model" in ckpt2:
        ckpt2 = extract(ckpt2)
    else:
        ckpt2 = ckpt2["weight"]
    if sorted(ckpt1.keys()) != sorted(ckpt2.keys()):
        return "Fail to merge the models. The model architectures are not the same."
    opt = OrderedDict(
        weight={
            key: 1 * value.float() + (1 - 1) * ckpt2[key].float()
            for key, value in ckpt1.items()
        }
    )
    opt["info"] = f"Model fusion of {pth_path_1} and {pth_path_2}"
    torch.save(opt, f"logs/{model_name}.pth")
    print(f"Model fusion of {pth_path_1} and {pth_path_2} is done.")
