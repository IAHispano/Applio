import argparse
import sys
import torch
import json
from multiprocessing import cpu_count
import os

global usefp16
usefp16 = False

def decide_fp_config():
    global usefp16
    usefp16 = False
    device_capability = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        device_capability = torch.cuda.get_device_capability(device)[0]
        if device_capability >= 7:
            usefp16 = True
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r") as d:
                    data = json.load(d)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = True

                with open(f"configs/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)



            with open(
                "trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.0", "3.7")

            with open(
                "trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
        else:
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r") as f:
                    data = json.load(f)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = False

                with open(f"configs/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)

                print(f"Set fp16_run to false in {config_file}")

            with open(
                "trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.7", "3.0")

            with open(
                "trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
    else:
        print(
            "CUDA is not available. Make sure you have an NVIDIA GPU and CUDA installed."
        )
    return (usefp16, device_capability)

class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.paperspace,
            self.is_cli,
            self.grtheme,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""

        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(  
            "--paperspace",
            action="store_true",
            help="Note that this argument just shares a gradio link for the web UI. Thus can be used on other non-local CLI systems.",
        )
        parser.add_argument(  
            "--is_cli",
            action="store_true",
            help="Use the CLI instead of setting up a gradio UI. This flag will launch an RVC text interface where you can execute functions from infer-web.py!",
        )

        parser.add_argument(
                    "-t",
                    "--theme",
            help    = "Theme for Gradio. Format - `JohnSmith9982/small_and_pretty` (no backticks)",
            default = "JohnSmith9982/small_and_pretty",
            type    = str
        )

        parser.add_argument(
            "--dml",
            action="store_true",
            help="Use DirectML backend instead of CUDA."
        )
        
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.paperspace,
            cmd_opts.is_cli,
            cmd_opts.theme,
            cmd_opts.dml,
        )

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                decide_fp_config()
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            print("No supported Nvidia GPU found, using MPS instead")
            self.device = "mps"
            self.device = self.instead = "mps"
            self.is_half = False
            decide_fp_config()
        else:
            print("No supported Nvidia GPU found, using CPU instead")
            self.device = "cpu"
            self.device = self.instead = "cpu"
            self.is_half = False
            decide_fp_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        
        if self.dml:
            print("use DirectML instead")
            try:
                os.rename("runtime\Lib\site-packages\onnxruntime","runtime\Lib\site-packages\onnxruntime-cuda")
            except:
                pass
            try:
                os.rename("runtime\Lib\site-packages\onnxruntime-dml","runtime\Lib\site-packages\onnxruntime")
            except:
                pass
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            print(self.device)
            self.is_half = False
        else:
            if self.instead:
                print(f"use {self.instead} instead")
            try:
                os.rename("runtime\Lib\site-packages\onnxruntime","runtime\Lib\site-packages\onnxruntime-dml")
            except:
                pass
            try:
                os.rename("runtime\Lib\site-packages\onnxruntime-cuda","runtime\Lib\site-packages\onnxruntime")
            except:
                pass

        return x_pad, x_query, x_center, x_max