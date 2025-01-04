import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from rvc.lib.algorithm.commons import get_padding, init_weights
from rvc.lib.algorithm.generators.conformer import Conformer
import einops

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class RingFormerGenerator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int =192,
        upsample_initial_channel: int = 512,
        upsample_rates = [12, 10, 2, 2],
        resblock_kernel_sizes = [3, 7, 11],
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5],],
        sample_rate = 48000,
        gin_channels=0,
    ):
        super(RingFormerGenerator, self).__init__()

        # standard HiFiGAN uses 4 upsamplers
        # this one uses first two upsamples, then istft the final x2 x2 expansion
        
        self.n_fft = 64
        self.hop_size =  upsample_rates[2] * upsample_rates[3]
        self.fft_window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        upsample_rates = upsample_rates[:2]
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, u in enumerate(upsample_rates):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size = u * 2,
                        stride = u,
                        padding = (u + 1) // 2,
                        output_padding = u % 2
                    )
                )
            )
        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conformers = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** i)
            self.conformers.append(
                Conformer(
                    dim=ch, 
                    depth=2,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor = 2,
                    conv_kernel_size=31,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1
                )
            )
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(Conv1d(128, self.n_fft + 2, 7, 1, padding=3))
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0, g=None):
        # x: [b,d,t]
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = einops.rearrange(x, 'b f t -> b t f')
            x = self.conformers[i](x)
            x = einops.rearrange(x, 'b t f -> b f t')
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
                
        x = x + (1 / self.alphas[i+1]) * (torch.sin(self.alphas[i+1] * x) ** 2)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        
        mag = torch.exp(x[:,:self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1:, :])
        
        out = torch.istft(
            mag * torch.exp(phase * 1j),
            self.n_fft,
            self.hop_size,
            self.n_fft,
            window=self.fft_window.to(mag.device)
        )

        return out.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
