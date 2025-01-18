import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, Tuple

def generate_pcph(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn((batch, 1, frames * hop_length), device=device)
    if torch.all(f0 == 0.0):
        return noise

    vuv = f0 > 0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = sample_rate / 2.0 / f0[vuv]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    # Compute harmonic amplitude
    harmonic_amplitude = vuv * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to the harmonic signal
    harmonics = harmonic_mask * harmonics
    harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise

class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    References:
        - https://github.com/gemelo-ai/vocos
        - https://github.com/echocatzh/torch-mfcc
    """

    def __init__(
        self, n_fft: int, n_bins: int, hop_length: int, window: Optional[str] = "hann_window"
    ) -> None:
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
        window = getattr(torch, window)(self.n_fft).reshape(1, n_fft, 1)
        self.register_buffer("window", window.reshape(1, n_fft, 1))
        window_envelope = window.square()
        self.register_buffer("window_envelope", window_envelope.reshape(1, n_fft, 1))

        # Create the kernel for enframe operation (sliding windows)
        enframe_kernel = torch.eye(self.n_fft).unsqueeze(1)
        self.register_buffer("enframe_kernel", enframe_kernel)

    def forward(self, x: Tensor, norm: Optional[str] = None) -> Tuple[Tensor, Tensor]:
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
        real, imag = x_stft.real, x_stft.imag

        return real, imag

    def inverse(self, real: Tensor, imag: Tensor, norm: Optional[str] = None) -> Tensor:
        """
        Perform the inverse Short-Time Fourier Transform (iSTFT) to reconstruct the waveform from the complex spectrogram.

        Args:
            real (Tensor): Real part of the complex spectrogram with shape (batch, n_bins, frames).
            imag (Tensor): Imaginary part of the complex spectrogram with shape (batch, n_bins, frames).
            norm (str, optional): Normalization mode for the inverse FFT (default: None).

        Returns:
            Tensor: Reconstructed waveform with shape (batch, 1, samples).
        """
        # Validate shape and dimensionality
        assert real.shape == imag.shape and real.ndim == 3

        # Ensure the input represents a one-sided spectrogram
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length

        # Inverse RDFT and apply windowing, followed by overlap-add
        x = torch.fft.irfft(torch.complex(real, imag), dim=1, norm=norm)
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

class NormLayer(nn.Module):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the NormLayer module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        x: Tensor,
        dim: int,
        mean: Optional[Tensor] = None,
        var: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, ...).
            dim (int): Dimensions along which statistics are calculated.
            mean (Tensor, optional): Mean tensor (default: None).
            var (Tensor, optional): Variance tensor (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized tensor and statistics.
        """
        # Calculate the mean along dimensions to be reduced
        if mean is None:
            mean = x.mean(dim, keepdim=True)

        # Centerize the input tensor
        x = x - mean

        # Calculate the variance
        if var is None:
            var = (x**2).mean(dim=dim, keepdim=True)

        # Normalize
        x = x / torch.sqrt(var + self.eps)

        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)

        return x, mean, var


class LayerNorm2d(NormLayer):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the LayerNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x

def drop_path(
    x: Tensor,
    drop_prob: Optional[float] = 0.0,
    training: Optional[bool] = False,
    scale_by_keep: Optional[bool] = True,
) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of drop_prob in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(
        self, drop_prob: Optional[float] = 0.0, scale_by_keep: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class ConvNeXtBlock2d(nn.Module):
    """
    A 2D residual block module based on ConvNeXt architecture.

    Reference:
        - https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        channels: int,
        layer_scale_init_value: float = None,
        mult_channels: int = 3,
        kernel_size: int = [13, 7],
        drop_prob: float = 0.0,
    ) -> None:
        """
        Initialize the ConvNeXtBlock2d module.

        Args:
            channels (int): Number of input and output channels for the block.
            mult_channels (int): Channel expansion factor used in pointwise convolutions.
            kernel_size (int): Size of the depthwise convolution kernel.
            drop_prob (float, optional): Probability of dropping paths for stochastic depth (default: 0.0).
            layer_scale_init_value (float, optional): Initial value for the learnable layer scale parameter.
                If None, no scaling is applied (default: None).
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel_size[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_size[1] % 2 == 1, "Kernel size must be odd number."
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False, padding_mode="reflect",)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * mult_channels, 1)
        self.nonlinear = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * mult_channels, channels, 1)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(1, channels, 1, 1),
                requires_grad=True,
            )
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Output tensor of the same shape (batch, channels, height, width).
        """
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.nonlinear(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = residual + self.drop_path(x)
        return x

class WavehaxGenerator(nn.Module):
    """
    Wavehax generator module.

    This module produces time-domain waveforms through complex spectrogram estimation
    based on the integration of 2D convolution and harmonic prior spectrograms.
    """

    def __init__(
        self,
        in_channels: int = 192,
        channels: int = 16  ,
        num_blocks: int = 8,
        sample_rate: int = 48000,
        prior_type: str = "pcph",
        gin_channels: int = 256,
    ) -> None:
        """
        Initialize the WavehaxGenerator module.

        Args:
            in_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used.
            drop_prob (float): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            use_logmag_phase (bool): Whether to use log-magnitude and phase for STFT (default: False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = sample_rate//25
        self.n_bins = self.n_fft // 2 + 1
        self.hop_length = sample_rate // 100
        self.sample_rate = sample_rate

        # STFT layer
        self.stft = STFT(
            n_fft = self.n_fft,
            n_bins = self.n_bins,
            hop_length = self.hop_length
        )

        # Input projection layers
        self.prior_proj1 = nn.Conv1d(self.n_bins, self.n_bins, 7, padding=3, padding_mode="reflect")
        self.prior_proj2 = nn.Conv1d(self.n_bins, self.n_bins, 7, padding=3, padding_mode="reflect")
        self.cond_proj = nn.Conv1d(in_channels, self.n_bins, 7, padding=3, padding_mode="reflect")
        self.cond = nn.Conv1d(gin_channels, self.n_bins, 1)

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock2d(channels, layer_scale_init_value=1 / num_blocks,) for _ in range(num_blocks)]
        )

        # Output normalization and projection layers
        self.output_norm = LayerNorm2d(channels)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor, f0: Tensor, g: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Encoded spectrogram with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = generate_pcph(f0.unsqueeze(1),
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
            real, imag = self.stft(prior)
            prior1, prior2 = real, imag
        
        # Apply input projection
        prior1_proj = self.prior_proj1(prior1)
        prior2_proj = self.prior_proj2(prior2)
        
        
        x = self.cond_proj(x)
        x += self.cond(g)

        # Convert to 2d representation
        x = torch.stack([prior1, prior2, prior1_proj, prior2_proj, x], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            x = f(x)
        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Apply iSTFT followed by overlap and add
        real, imag = x[:, 0], x[:, 1]
        
        x = self.stft.inverse(real, imag)

        return x
