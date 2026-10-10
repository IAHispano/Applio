from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend

from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange
import torchaudio


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(Module):

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_head=32,
        heads=8,
        scale=8,
        flash=True,
        dropout=0.0,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(scale=scale, dropout=dropout, flash=flash)

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                )

            self.layers.append(
                ModuleList(
                    [attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):

        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        x = torch.stack(outs, dim=-2)

        return x


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.InstanceNorm2d(c2, affine=True, eps=1e-8)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.InstanceNorm2d(c2, affine=True, eps=1e-8)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConv(c1, c_, k=3, s=1)
        self.dsconv2 = DSConv(c_, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.dsconv2(self.dsconv1(x))
            if self.shortcut
            else self.dsconv2(self.dsconv1(x))
        )


class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(
            *[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        x_ = self.cv1(x)
        x_ = self.m(x_)
        return self.cv2(x_)


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads=8):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))

        self.context_mapper = nn.Linear(
            2 * in_channels, num_hyperedges * in_channels, bias=False
        )

        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.scale = self.head_dim**-0.5

    def forward(self, x):
        B, N, C = x.shape

        f_avg = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_ctx = torch.cat((f_avg, f_max), dim=1)

        delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
        P = self.global_proto.unsqueeze(0) + delta_P

        z = self.query_proj(x)

        z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(
            0, 2, 3, 1
        )

        sim = (z @ P) * self.scale

        s_bar = sim.mean(dim=1)

        A = F.softmax(s_bar.permute(0, 2, 1), dim=-1)

        return A


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        f_m = torch.bmm(A, x)
        f_m = self.act(self.W_e(f_m))

        x_out = torch.bmm(A.transpose(1, 2), f_m)
        x_out = self.act(self.W_v(x_out))

        return x + x_out


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=8, num_heads=8):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(
            in_channels, num_hyperedges, num_heads
        )
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)

        A = self.adaptive_hyperedge_gen(x_flat)

        x_out_flat = self.hypergraph_conv(x_flat, A)

        x_out = x_out_flat.permute(0, 2, 1).view(B, -1, H, W)
        return x_out


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges=8, num_heads=8, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))


class HyperACE(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_hyperedges=8,
        num_heads=8,
        k=2,
        l=1,
        c_h=0.5,
        c_l=0.25,
    ):
        super().__init__()

        c2, c3, c4, c5 = in_channels
        c_mid = c4

        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)

        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        assert self.c_s > 0, "Channel split error"

        self.high_order_branch = nn.ModuleList(
            [
                C3AH(self.c_h, self.c_h, num_hyperedges, num_heads, e=1.0)
                for _ in range(k)
            ]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)

        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )

        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        B2, B3, B4, B5 = x

        B, _, H4, W4 = B4.shape

        B2_resized = F.interpolate(
            B2, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B3_resized = F.interpolate(
            B3, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B5_resized = F.interpolate(
            B5, size=(H4, W4), mode="bilinear", align_corners=False
        )

        x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1))

        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))

        x_l_out = self.low_order_branch(x_l)

        y = self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))

        return y


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        if f_in.shape[1] != h.shape[1]:
            raise ValueError(f"Channel mismatch: f_in={f_in.shape}, h={h.shape}")
        return f_in + self.gamma * h


class Backbone(nn.Module):
    def __init__(self, in_channels=256, base_channels=64, base_depth=3):
        super().__init__()
        c = base_channels
        c2 = base_channels
        c3 = 256
        c4 = 384
        c5 = 512
        c6 = 768

        self.stem = DSConv(in_channels, c2, k=3, s=(2, 1), p=1)

        self.p2 = nn.Sequential(
            DSConv(c2, c3, k=3, s=(2, 1), p=1), DS_C3k2(c3, c3, n=base_depth)
        )

        self.p3 = nn.Sequential(
            DSConv(c3, c4, k=3, s=(2, 1), p=1), DS_C3k2(c4, c4, n=base_depth * 2)
        )

        self.p4 = nn.Sequential(
            DSConv(c4, c5, k=3, s=(2, 1), p=1), DS_C3k2(c5, c5, n=base_depth * 2)
        )

        self.p5 = nn.Sequential(
            DSConv(c5, c6, k=3, s=(2, 1), p=1), DS_C3k2(c6, c6, n=base_depth)
        )

        self.out_channels = [c3, c4, c5, c6]

    def forward(self, x):
        x = self.stem(x)
        x2 = self.p2(x)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)
        return [x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        hyperace_out_c: int,
        decoder_channels: List[int],
    ):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d2, c_d3, c_d4, c_d5 = decoder_channels

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)

    def forward(self, enc_feats: List[torch.Tensor], h_ace: torch.Tensor):
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)
        h_d5 = self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode="bilinear"))
        d5 = self.fusion_d5(d5, h_d5)

        d5_up = F.interpolate(d5, size=p4.shape[2:], mode="bilinear")
        d4_skip = self.skip_p4(p4)
        d4 = self.up_d5(d5_up) + d4_skip

        h_d4 = self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode="bilinear"))
        d4 = self.fusion_d4(d4, h_d4)

        d4_up = F.interpolate(d4, size=p3.shape[2:], mode="bilinear")
        d3_skip = self.skip_p3(p3)
        d3 = self.up_d4(d4_up) + d3_skip

        h_d3 = self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode="bilinear"))
        d3 = self.fusion_d3(d3, h_d3)

        d3_up = F.interpolate(d3, size=p2.shape[2:], mode="bilinear")
        d2_skip = self.skip_p2(p2)
        d2 = self.up_d3(d3_up) + d2_skip

        h_d2 = self.h_to_d2(F.interpolate(h_ace, size=d2.shape[2:], mode="bilinear"))
        d2 = self.fusion_d2(d2, h_d2)

        d2_final = self.final_d2(d2)

        return d2_final


class FreqPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = DSConv(in_channels, out_channels * scale, k=3, s=1, p=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        B, C_r, H, W = x.shape
        out_c = C_r // self.scale

        x = x.view(B, out_c, self.scale, H, W)

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, out_c, H, W * self.scale)

        return x


class ProgressiveUpsampleHead(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025):
        super().__init__()
        self.target_bins = target_bins

        c = in_channels

        self.block1 = FreqPixelShuffle(c, c, scale=2)
        self.block2 = FreqPixelShuffle(c, c // 2, scale=2)
        self.block3 = FreqPixelShuffle(c // 2, c // 2, scale=2)
        self.block4 = FreqPixelShuffle(c // 2, c // 4, scale=2)

        self.final_conv = nn.Conv2d(c // 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if x.shape[-1] != self.target_bins:
            x = F.interpolate(
                x,
                size=(x.shape[2], self.target_bins),
                mode="bilinear",
                align_corners=False,
            )

        x = self.final_conv(x)
        return x


class SegmModel(nn.Module):
    def __init__(
        self,
        in_bands=62,
        in_dim=256,
        out_bins=1025,
        out_channels=4,
        base_channels=64,
        base_depth=2,
        num_hyperedges=16,
        num_heads=8,
    ):
        super().__init__()

        self.backbone = Backbone(
            in_channels=in_dim, base_channels=base_channels, base_depth=base_depth
        )
        enc_channels = self.backbone.out_channels
        c2, c3, c4, c5 = enc_channels

        hyperace_in_channels = enc_channels
        hyperace_out_channels = c4
        self.hyperace = HyperACE(
            hyperace_in_channels,
            hyperace_out_channels,
            num_hyperedges,
            num_heads,
            k=3,
            l=2,
        )

        decoder_channels = [c2, c3, c4, c5]
        self.decoder = Decoder(enc_channels, hyperace_out_channels, decoder_channels)

        self.upsample_head = ProgressiveUpsampleHead(
            in_channels=decoder_channels[0],
            out_channels=out_channels,
            target_bins=out_bins,
        )

    def forward(self, x):
        H, W = x.shape[2:]

        enc_feats = self.backbone(x)

        h_ace_feats = self.hyperace(enc_feats)

        dec_feat = self.decoder(enc_feats, h_ace_feats)

        feat_time_restored = F.interpolate(
            dec_feat, size=(H, dec_feat.shape[-1]), mode="bilinear", align_corners=False
        )

        out = self.upsample_head(feat_time_restored)

        return out


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

        self.segm = SegmModel(
            in_bands=len(dim_inputs), in_dim=dim, out_bins=sum(dim_inputs) // 4
        )

    def forward(self, x):
        y = rearrange(x, "b t f c -> b c t f")
        y = self.segm(y)
        y = rearrange(y, "b c t f -> b t (f c)")

        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1) + y


DEFAULT_FREQS_PER_BANDS = (
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    128,
    129,
)


class BSRoformerHyperACE(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        **kwargs,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(
            default(stft_window_fn, torch.hann_window), stft_win_length
        )

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True,
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert (
            sum(freqs_per_bands) == freqs
        ), f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}"

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in freqs_per_bands
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):

        device = raw_audio.device

        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(
                raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except:
            stft_repr = torch.stft(
                raw_audio.cpu() if x_is_mps else raw_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")

        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        x = rearrange(stft_repr, "b f t c -> b t (f c)")

        x = self.band_split(x)

        for i, transformer_block in enumerate(self.layers):

            time_transformer, freq_transformer = transformer_block

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        try:
            recon_audio = torch.istft(
                stft_repr,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio.shape[-1],
            )
        except:
            recon_audio = torch.istft(
                stft_repr.cpu() if x_is_mps else stft_repr,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio.shape[-1],
            ).to(device)

        recon_audio = rearrange(
            recon_audio, "(b n s) t -> b n s t", s=self.audio_channels, n=num_stems
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[..., : recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(
                rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs
            )
            target_Y = torch.stft(
                rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs
            )

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(
                recon_Y, target_Y
            )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
