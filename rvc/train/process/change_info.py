import os
import torch


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, f"logs/{name}/{name}")
        return "Success."
    except Exception as error:
        print(error)
