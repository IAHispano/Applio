import os
import re
import subprocess
import sys

LEGACY_GPU_PATTERN = re.compile(
    r"\b(?:gtx\s*(?!16\d\d)\d+|gt\s*\d+|p10[0-9]|p40|p4\b|cmp\s*(?:30|40|50)hx|titan\s*[xv]|tesla\s*[pmk]\d+)\b",
    re.IGNORECASE,
)


def get_nvidia_gpus():
    """
    Query nvidia-smi for all installed NVIDIA GPUs with their name and compute capability.
    Returns a list of dicts: [{'name': str, 'compute_cap': float}]
    """
    gpus = []
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            **kwargs,
        )
        if res.returncode == 0 and res.stdout:
            for line in res.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        cap = float(parts[1])
                    except ValueError:
                        cap = 0.0
                    gpus.append({"name": name, "compute_cap": cap})
                elif len(parts) == 1:
                    gpus.append({"name": parts[0], "compute_cap": 0.0})
    except Exception:
        pass

    # If nvidia-smi was unavailable or returned nothing, check via torch if already loaded
    if not gpus and "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    try:
                        cap_tuple = torch.cuda.get_device_capability(i)
                        cap = float(f"{cap_tuple[0]}.{cap_tuple[1]}")
                    except Exception:
                        cap = 0.0
                    gpus.append({"name": name, "compute_cap": cap})
        except Exception:
            pass

    return gpus


def is_legacy_nvidia_gpu(gpus=None):
    """
    Determine if any installed NVIDIA GPU belongs to a legacy architecture:
    - Maxwell (5.x), Pascal (6.x), Volta (7.0), Kepler (3.x) -> compute_cap < 7.5
    - Famous legacy cards: GTX 1080/1070/1060/1050, P104-100, P106-100, Titan X, etc.
    """
    if gpus is None:
        gpus = get_nvidia_gpus()
    if not gpus:
        return False

    for gpu in gpus:
        cap = gpu.get("compute_cap", 0.0)
        name = gpu.get("name", "")
        if 0.0 < cap < 7.5:
            return True
        if LEGACY_GPU_PATTERN.search(name):
            return True
    return False


def check_torch_compatibility():
    """
    Check if the currently installed PyTorch is compatible with the detected GPU hardware.
    PyTorch >= 2.8 dropped Maxwell (5.x), Pascal (6.x), and Volta (7.0) architecture kernels.
    Older GPUs require PyTorch 2.7.1 with cu126.
    Returns: (is_compatible: bool, message: str, is_legacy: bool)
    """
    gpus = get_nvidia_gpus()
    legacy = is_legacy_nvidia_gpu(gpus)

    if not legacy:
        return (
            True,
            "GPU is modern or non-NVIDIA; current PyTorch is compatible.",
            False,
        )

    # Check installed torch version
    try:
        import torch

        torch_ver = getattr(torch, "__version__", "")
    except ImportError:
        return (
            False,
            "PyTorch is not installed. For older GPUs, install PyTorch 2.7.1 (cu126).",
            True,
        )

    # PyTorch 2.7.x is the legacy-compatible series
    if torch_ver.startswith("2.7."):
        return (
            True,
            f"PyTorch {torch_ver} is compatible with legacy GPU hardware.",
            True,
        )

    # Test if CUDA kernel execution actually fails
    kernel_error = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")
    except Exception as e:
        kernel_error = str(e)

    gpu_desc_items = []
    for g in gpus:
        cap_val = g.get("compute_cap", 0.0)
        c_str = f"sm_{cap_val}" if cap_val > 0 else "legacy"
        gpu_desc_items.append(f"{g.get('name', 'NVIDIA GPU')} ({c_str})")
    gpu_desc = ", ".join(gpu_desc_items)

    reason = (
        f"Detected legacy NVIDIA GPU ({gpu_desc}), but PyTorch {torch_ver} is installed. "
        "PyTorch >= 2.8 removed support for these architectures (Pascal/Maxwell/Volta/P104-100). "
        f"{f'CUDA kernel error: {kernel_error}. ' if kernel_error else ''}"
        "Downgrade to PyTorch 2.7.1 (cu126) is required for CUDA acceleration."
    )
    return False, reason, True


def downgrade_torch(python_exe=None, cuda_tag="cu126"):
    """
    Uninstall existing torch, torchvision, torchaudio and install PyTorch 2.7.1 (cu126).
    """
    if python_exe is None:
        python_exe = sys.executable

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    print(
        f"[*] Downgrading PyTorch to 2.7.1 ({cuda_tag}) for legacy GPU compatibility..."
    )
    print(f"[*] Target Python: {python_exe}")
    print(f"[*] Index URL: {index_url}\n")

    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # 1. Uninstall torch torchvision torchaudio
    uninstall_cmd = [
        python_exe,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torch",
        "torchvision",
        "torchaudio",
    ]
    print(f"$ {' '.join(uninstall_cmd)}")
    uninst = subprocess.run(uninstall_cmd, **kwargs)
    if uninst.returncode != 0:
        print("[!] Warning: pip uninstall reported a non-zero exit code, continuing...")

    # 2. Install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
    install_cmd = [
        python_exe,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "--extra-index-url",
        index_url,
    ]
    print(f"$ {' '.join(install_cmd)}")
    inst = subprocess.run(install_cmd, **kwargs)
    if inst.returncode == 0:
        print(f"\n[✓] PyTorch 2.7.1 ({cuda_tag}) successfully installed!")
        return True
    else:
        print(f"\n[✗] Failed to install PyTorch 2.7.1 (exit code {inst.returncode}).")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and manage PyTorch compatibility for legacy NVIDIA GPUs."
    )
    parser.add_argument(
        "--downgrade",
        action="store_true",
        help="Downgrade PyTorch to 2.7.1 (cu126) if an older GPU is detected or forced.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force downgrade to PyTorch 2.7.1 regardless of detected GPU.",
    )
    parser.add_argument(
        "--cuda",
        default="cu126",
        help="CUDA wheel tag for PyTorch download (default: cu126).",
    )
    args = parser.parse_args()

    gpus = get_nvidia_gpus()
    print("Detected GPUs:")
    if gpus:
        for g in gpus:
            cap_str = f"sm_{g['compute_cap']}" if g["compute_cap"] > 0 else "unknown"
            print(f"  • {g['name']} ({cap_str})")
    else:
        print("  • None detected via nvidia-smi")

    legacy = is_legacy_nvidia_gpu(gpus)
    print(f"Legacy GPU architecture (< sm_75 / Pascal / Maxwell / Volta): {legacy}")

    compatible, msg, _ = check_torch_compatibility()
    print(f"Compatibility status: {'Compatible' if compatible else 'Incompatible'}")
    print(f"Details: {msg}\n")

    if args.downgrade or args.force:
        if args.force or not compatible or legacy:
            success = downgrade_torch(cuda_tag=args.cuda)
            sys.exit(0 if success else 1)
        else:
            print(
                "GPU is modern and PyTorch is already compatible. Use --force to override."
            )
    elif not compatible:
        print("To downgrade now, run:")
        print(f"  {sys.executable} -m rvc.lib.tools.gpu_checker --downgrade")
        sys.exit(1)


if __name__ == "__main__":
    main()
