import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
import numpy as np

class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    References:
        - https://github.com/gemelo-ai/vocos
        - https://github.com/echocatzh/torch-mfcc
    """

    def __init__(
        self, n_fft: int, n_bins: int, hop_length: int, window: str = "hann_window"
    ):
        """
        Initialize the STFT module.

        Args:
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            window (str, optional): Name of the window function (default: "hann_window").
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_bins
        self.hop_length = hop_length

        # Create the window function and its squared values for normalization
        window = torch.from_numpy(get_window('hann', self.n_fft, fftbins=True).astype(np.float32))
        self.register_buffer("window", window.reshape(1, n_fft, 1))
        window_envelope = window.square()
        self.register_buffer("window_envelope", window_envelope.reshape(1, n_fft, 1))

        # Create the kernel for enframe operation (sliding windows)
        enframe_kernel = torch.eye(self.n_fft).unsqueeze(1)
        self.register_buffer("enframe_kernel", enframe_kernel)

    def forward(self, x, norm: str = None):
        """
        Perform the forward Short-Time Fourier Transform (STFT) on the input waveform.

        Args:
            x (Tensor): Input waveform with shape (batch, samples) or (batch, 1, samples).
            norm (str, optional): Normalization mode for the FFT (default: None).

        Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the STFT result.
        """
        # Apply zero-padding to the input signal
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))

        # Enframe the padded waveform (sliding windows)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel, stride=self.hop_length)

        # Perform the forward real-valued DFT on each frame
        x = x * self.window
        x_stft = torch.fft.rfft(x, dim=1, norm=norm)
        
        return x_stft

    def inverse(self, complex, norm = None):
        frames = complex.shape[2]
        samples = frames * self.hop_length
        x = torch.fft.irfft(complex, dim=1, norm=norm)
        x = x * self.window
        x = F.conv_transpose1d(x, self.enframe_kernel, stride=self.hop_length)
        
        # Compute window envelope for normalization
        window_envelope = F.conv_transpose1d(
            self.window_envelope.repeat(1, 1, frames),
            self.enframe_kernel,
            stride=self.hop_length,
        )

        # Remove padding
        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]

        # Normalize the output by the window envelope
        assert (window_envelope > 1e-11).all()
        x = x / window_envelope

        return x