import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 groups: int = 1, device=None, dtype=None) -> None:
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        super().__init__(in_features // groups, out_features, bias, device, dtype)

    def forward(self, input):
        if self.groups == 1:
            return super().forward(input)
        else:
            sh = input.shape[:-1]
            input = input.view(*sh, self.groups, -1)
            weight = self.weight.view(self.groups, -1, self.weight.shape[-1])
            output = torch.einsum('...gi,...goi->...go', input, weight)
            output = output.reshape(*sh, -1) + self.bias
            return output

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim, groups):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.groups = groups

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        if self.groups == 1:
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        else:
            Gx = Gx.view(*Gx.shape[:2], self.groups, -1)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            Nx = Nx.view(*Nx.shape[:2], -1)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = GroupLinear(dim, intermediate_dim, groups=groups)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim, groups=groups)
        self.pwconv2 = GroupLinear(intermediate_dim, dim, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = residual + x
        return x
		
class WaveNextGenerator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 192,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        gin_channels: int = 256,
        sample_rate: int = 48000,
    ):
        super().__init__()

        self.n_fft = 1024
        self.hop_length = sample_rate // 100

        self.input_channels = input_channels

        self.in_conv = torch.nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)

        self.layers = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_last = torch.nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

        self.linear_1 = torch.nn.Linear(dim, self.n_fft + 2)
        self.linear_2 = torch.nn.Linear(self.n_fft + 2, self.hop_length, bias = False)
        
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=0.02)		
        
        self.cond = None
        if gin_channels > 0:
            self.cond = torch.nn.Conv1d(gin_channels, dim, kernel_size=1)	

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor):
        # x = mel spec (8, 192, 36), f0 = (8, 36), g = (8, 256, 1)
        # f0 not used

        x = self.in_conv(x)

        if g is not None and self.cond:
            x +=self.cond(g)

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2))
        #print(x.shape)
        B, C, T = x.shape
        x = self.linear_1(x)
        x = self.linear_2(x)
        audio = x.view(B,-1) # / 100
        audio = torch.clip(audio, min=-1.0, max=1.0)
        return audio.unsqueeze(1)