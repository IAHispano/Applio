import os
import torch
#start Zluda changes
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
#end Zluda changes


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
