import os
import sys
import torch

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_DEBUG_DISABLE_FIND_DB", "1")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("AMD_COMGR_CACHE", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")


def setup_windows_msvc_env():
    """
    Configures MSVC and Windows SDK include and bin paths so hiprtc and MIOpen
    can find standard C++ headers (e.g. type_traits, ucrt) during runtime kernel JIT compilation.
    """
    if sys.platform != "win32":
        return
    import subprocess

    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, r"Microsoft Visual Studio\Installer\vswhere.exe")
    if not os.path.exists(vswhere):
        return

    try:
        flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        vs_path = subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            timeout=5,
            creationflags=flags,
        ).strip()
    except Exception:
        return

    if not vs_path or not os.path.isdir(vs_path):
        return

    # Add MSVC bin and include
    msvc_dir = os.path.join(vs_path, "VC", "Tools", "MSVC")
    if os.path.isdir(msvc_dir):
        versions = sorted(os.listdir(msvc_dir), reverse=True)
        for v in versions:
            ver_path = os.path.join(msvc_dir, v)
            bin_dir = os.path.join(ver_path, "bin", "Hostx64", "x64")
            inc_dir = os.path.join(ver_path, "include")
            if (
                os.path.isdir(bin_dir)
                and bin_dir.lower() not in os.environ.get("PATH", "").lower()
            ):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            if os.path.isdir(inc_dir):
                current_inc = os.environ.get("INCLUDE", "")
                if inc_dir.lower() not in current_inc.lower():
                    os.environ["INCLUDE"] = (
                        (inc_dir + os.pathsep + current_inc) if current_inc else inc_dir
                    )
            break

    # Add Windows SDK include (ucrt, shared, um)
    sdk_inc_base = os.path.join(pf86, r"Windows Kits\10\Include")
    if os.path.isdir(sdk_inc_base):
        try:
            sdk_versions = sorted(os.listdir(sdk_inc_base), reverse=True)
            for v in sdk_versions:
                sdk_v_path = os.path.join(sdk_inc_base, v)
                for sub in ("ucrt", "shared", "um"):
                    sub_path = os.path.join(sdk_v_path, sub)
                    if os.path.isdir(sub_path):
                        current_inc = os.environ.get("INCLUDE", "")
                        if sub_path.lower() not in current_inc.lower():
                            os.environ["INCLUDE"] = (
                                (sub_path + os.pathsep + current_inc)
                                if current_inc
                                else sub_path
                            )
                break
        except Exception:
            pass


def is_amd_device() -> bool:
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is not None:
        return True
    try:
        dev_name = torch.cuda.get_device_name().upper()
        if "AMD" in dev_name or "RADEON" in dev_name or dev_name.endswith("[ZLUDA]"):
            return True
    except Exception:
        pass
    return False


if is_amd_device():
    # Setup MSVC paths if present on Windows to prevent hiprtc compilation errors
    setup_windows_msvc_env()
    # Disabling MIOpen (cuDNN) forces PyTorch to use its native ATen precompiled
    # C++/HIP kernels for BatchNorm and Convolutions, completely avoiding
    # MIOpen JIT compilation failures (e.g. fatal error: 'type_traits' file not found
    # in hiprtc during MIOpenBatchNormFwdInferSpatial) on Windows and unstable MIOpen solvers.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):

    class STFT:
        def __init__(self):
            self.device = "cuda"
            self.fourier_bases = {}  # Cache for Fourier bases

        def _get_fourier_basis(self, n_fft):
            # Check if the basis for this n_fft is already cached
            if n_fft in self.fourier_bases:
                return self.fourier_bases[n_fft]
            fourier_basis = torch.fft.fft(torch.eye(n_fft, device="cpu")).to(
                self.device
            )
            # stack separated real and imaginary components and convert to torch tensor
            cutoff = n_fft // 2 + 1
            fourier_basis = torch.cat(
                [fourier_basis.real[:cutoff], fourier_basis.imag[:cutoff]], dim=0
            )
            # cache the tensor and return
            self.fourier_bases[n_fft] = fourier_basis
            return fourier_basis

        def transform(self, input, n_fft, hop_length, window):
            # fetch cached Fourier basis
            fourier_basis = self._get_fourier_basis(n_fft)
            # apply hann window to Fourier basis
            fourier_basis = fourier_basis * window
            # pad input to center with reflect
            pad_amount = n_fft // 2
            input = torch.nn.functional.pad(
                input, (pad_amount, pad_amount), mode="reflect"
            )
            # separate input into n_fft-sized frames
            input_frames = input.unfold(1, n_fft, hop_length).permute(0, 2, 1)
            # apply fft to each frame
            fourier_transform = torch.matmul(fourier_basis, input_frames)
            cutoff = n_fft // 2 + 1
            return torch.complex(
                fourier_transform[:, :cutoff, :], fourier_transform[:, cutoff:, :]
            )

    stft = STFT()
    _torch_stft = torch.stft

    def z_stft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        # only optimizing a specific call from rvc.train.mel_processing.MultiScaleMelSpectrogramLoss
        if (
            kwargs.get("win_length") == None
            and kwargs.get("center") == None
            and kwargs.get("return_complex") == True
        ):
            # use GPU accelerated calculation
            return stft.transform(
                input, kwargs.get("n_fft"), kwargs.get("hop_length"), window
            )
        else:
            # simply do the operation on CPU
            return _torch_stft(
                input=input.cpu(), window=window.cpu(), *args, **kwargs
            ).to(input.device)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    # hijacks
    torch.stft = z_stft
    torch.jit.script = z_jit

# MIOpen has no usable dilated 1D convolution kernel on some AMD architectures. On gfx1100 the
# identical FLOPs run ~30x slower dilated than undilated, and the HiFi-GAN / NSF ResBlocks are built
# on dilations (1, 3, 5), so this dominates both training and inference.
#
# A dilation-D convolution is exactly D independent undilated convolutions over the strided phases
# x[..., i::D], which avoids the kernel entirely. Whether that wins is a property of the card and
# the ROCm build rather than of the vendor, so it is measured once here at startup and the native
# kernel keeps ties. Patching F.conv1d rather than the models means every dilated conv is covered,
# including ones outside the ResBlocks.
if is_amd_device():
    import time

    _conv1d = torch.nn.functional.conv1d

    def _conv1d_phases(input, weight, bias, pad, dilation, groups):
        length = input.shape[-1]
        input = torch.nn.functional.pad(input, (pad, pad))
        # phases must divide evenly; the remainder is padded off the end, so it only ever lands
        # beyond `length` and is dropped by the final slice
        rem = (-input.shape[-1]) % dilation
        if rem:
            input = torch.nn.functional.pad(input, (0, rem))
        n, c, total = input.shape
        phases = (
            input.view(n, c, total // dilation, dilation)
            .permute(0, 3, 1, 2)
            .reshape(n * dilation, c, total // dilation)
        )
        out = _conv1d(phases, weight, bias, 1, 0, 1, groups)
        chan, per = weight.shape[0], out.shape[-1]
        out = (
            out.view(n, dilation, chan, per)
            .permute(0, 2, 3, 1)
            .reshape(n, chan, per * dilation)
        )
        return out[..., :length]

    def z_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        first = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        d, s, p = first(dilation), first(stride), first(padding)
        # only the plain dilated case is rewritten; everything else falls through untouched
        if d == 1 or s != 1 or not isinstance(p, int) or not input.is_cuda:
            return _conv1d(input, weight, bias, stride, padding, dilation, groups)
        return _conv1d_phases(input, weight, bias, p, d, groups)

    def _dilated_conv_is_slow():
        conv = torch.nn.Conv1d(192, 192, 3, dilation=3, padding=3).cuda().eval()
        x = torch.randn(4, 192, 4096, device="cuda")

        def timed(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        with torch.no_grad():
            native = timed(lambda: _conv1d(x, conv.weight, conv.bias, 1, 3, 3, 1))
            phases = timed(lambda: _conv1d_phases(x, conv.weight, conv.bias, 3, 3, 1))
        # a clear margin, so timing noise cannot pick the rewrite on a card whose kernel is fine
        return phases * 1.5 < native

    try:
        if _dilated_conv_is_slow():
            torch.nn.functional.conv1d = z_conv1d
    except Exception:
        pass  # any failure here just leaves the stock kernel in use
