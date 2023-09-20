import argparse
import getpass
import sys
sys.path.append('..')
import json
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
    if torch.xpu.is_available():
        from lib.infer.modules.ipex import ipex_init
        ipex_init()
except Exception:
    pass

import logging

logger = logging.getLogger(__name__)

import os
import sys

# Nombre de usuario actual
current_user = os.getlogin() or getpass.getuser()
logger.info("Current user: %s" % current_user)
# Ruta de Python 3.9 para el usuario actual
python_version = "39"
#C:\Users\USUARIO\AppData\Local\Programs\Python\Python39
python_path = os.path.join("C:\\Users", current_user, "AppData", "Local", "Programs", "Python", f"Python{python_version}")


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
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
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            with open(f"./assets/configs/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

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

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
        
    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("lib/infer/modules/train/preprocess.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("lib/infer/modules/train/preprocess.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            if self.gpu_mem == 4:
                x_pad = 1
                x_query = 5
                x_center = 30
                x_max = 32
            elif self.gpu_mem <= 3:
                x_pad = 1
                x_query = 2
                x_center = 16
                x_max = 18
        
        if self.dml:
            logger.info("Use DirectML instead")
            directml_dll_path = os.path.join(python_path, "Lib", "site-packages", "onnxruntime", "capi", "DirectML.dll")
            if (
                os.path.exists(
                    directml_dll_path
                )
                == False
            ):
                try:
                    os.rename(
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime"),
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime-cuda"),
                    )
                except:
                    pass
                try:
                    os.rename(
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime-dml"),
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime"),
                    )
                except:
                    pass
            # if self.device != "cpu":
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
            providers_cuda_dll_path = os.path.join(python_path, "Lib", "site-packages", "onnxruntime", "capi", "onnxruntime_providers_cuda.dll")
            if (
                os.path.exists(
                    providers_cuda_dll_path
                )
                == False
            ):
                try:
                    os.rename(
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime"),
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime-dml"),
                    )
                except:
                    pass
                try:
                    os.rename(
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime-cuda"),
                        os.path.join(python_path, "Lib", "site-packages", "onnxruntime"),
                    )
                except:
                    pass
        return x_pad, x_query, x_center, x_max
