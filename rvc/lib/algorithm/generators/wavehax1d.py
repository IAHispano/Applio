import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_pcph(
    f0,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: float = 0.01,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
    max_frequency: float = None,
):
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
        window = getattr(torch, window)(self.n_fft).reshape(1, n_fft, 1)
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
        real, imag = x_stft.real, x_stft.imag

        return real, imag

    def inverse(self, real, imag, norm=None):
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
        
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # x: [BatchSize, cnannels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x.mT, (self.channels,), self.gamma, self.beta, self.eps)
        return x.mT

# Global Resnponse Normalization for 1d Sequence (shape=[BatchSize, Channels, Length])
class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    # x: [batchsize, channels, length]
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ConvNeXt v2
class ConvNeXtLayer(nn.Module):
    def __init__(self, channels=512, kernel_size=7, mlp_mul=2):
        super().__init__()
        padding = kernel_size // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, groups=channels)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.grn = GRN(channels * mlp_mul)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    # x: [batchsize, channels, length]
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.c3(x)
        x = x + res
        return x
        
class WaveHaxGenerator1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        channels: int = 16,
        num_layers: int = 8,
        sample_rate: int = 48000,
        gin_channels: int = 256,
    ):
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
        
        self.input_proj = nn.Conv1d(self.n_bins * 5, self.n_bins, 1, bias=False)
        
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(self.n_bins) for _ in range(num_layers)])
        
        self.output_proj = nn.Conv1d(self.n_bins, self.n_bins * 2, 1)
  
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
  
    def forward(self, x, f0, g):
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
        
        x = torch.cat([prior1, prior2, prior1_proj, prior2_proj,x], dim=1)
      
        x = self.input_proj(x)
        
        for layer in self.mid_layers:
            x = layer(x)
        
        x = self.output_proj(x)
        
        real, imag = x[:, :self.n_bins], x[:, self.n_bins:]
        
        x = self.stft.inverse(real, imag)
        
        return x
