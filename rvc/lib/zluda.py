import torch

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
    # disabling unsupported cudnn
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
