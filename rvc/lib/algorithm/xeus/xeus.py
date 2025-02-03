import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.xeus.encoder import EBranchformerEncoder

class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor):
        x = input.transpose(-2, -1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x
        
class ConvLayerBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = LayerNorm(normalized_shape=out_channels, elementwise_affine=True,)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,)

    def forward(self, x: torch.Tensor, length: torch.Tensor,):
        x = self.conv(x)
        x = torch.utils.checkpoint.checkpoint(self.layer_norm, x, use_reentrant=False)
        x = nn.functional.gelu(x)
        length = (torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1)
        length = torch.max(torch.zeros_like(length), length)
        return x, length

class Frontend(nn.Module):
    def __init__(self,):
        super().__init__()
        shapes = [ [512, 10, 5], [512, 3, 2], [512, 3, 2], [512, 3, 2], [512, 3, 2], [512, 2, 2], [512, 2, 2],]
        blocks = []
        in_channels = 1
        self.downsampling_factor = 1
        for i, (out_channels, kernel_size, stride) in enumerate(shapes):
            blocks.append(ConvLayerBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,))
            in_channels = out_channels
            self.downsampling_factor *= stride
        self.layers = nn.Sequential(*blocks)

    def forward(
        self,
        x: torch.Tensor,                    # (1, T)
        length: torch.Tensor,               # (1)
    ):
        x = F.layer_norm(x, x.shape)
        x = x.unsqueeze(1)                  # (batch, channel==1, frame)
        for layer in self.layers:
            x, length = layer(x, length)    # (batch, feature, frame)
        x = x.transpose(1, 2)               # (batch, frame, feature)
        return x, length

class LinearProjection(torch.nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.output_dim = output_size
        self.linear_out = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        output = self.linear_out(self.dropout(input))
        return output, input_lengths

class XeusModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = Frontend()
        self.preencoder = LinearProjection()
        self.encoder = EBranchformerEncoder()
        self.final_proj = nn.Linear(1024, 768)
        
    def load_checkpoint(self, model_path:str):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=True)

        
    def forward(self, speech):
        with torch.no_grad():
            speech_lengths = torch.LongTensor([speech.shape[-1]]).to(speech.device)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
            encoder_out = self.encoder(feats, feats_lengths)
            encoder_out = self.final_proj(encoder_out)
            del feats, feats_lengths
        return {"last_hidden_state": encoder_out}
