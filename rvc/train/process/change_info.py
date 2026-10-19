import os
import torch
from rvc.lib.user_config import get_logs_dir


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ckpt["info"] = info

        if not name:
            name = os.path.splitext(os.path.basename(path))[0]

        target_dir = os.path.join(get_logs_dir(), name)
        os.makedirs(target_dir, exist_ok=True)

        torch.save(ckpt, os.path.join(target_dir, f"{name}.pth"))

        return "Success."

    except Exception as error:
        print(f"An error occurred while changing the info: {error}")
        return f"Error: {error}"
