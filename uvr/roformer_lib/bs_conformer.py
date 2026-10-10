from functools import partial

import torch
from torch import nn, einsum, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# helper functions


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


# norm


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# feedforward


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


# attention


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


# optional linear attention block


class LinearAttention(Module):
    """
    https://arxiv.org/abs/2106.09681 (El-Nouby et al.)
    """

    @beartype
    def __init__(
        self, *, dim, dim_head=32, heads=8, scale=8, flash=True, dropout=0.0, **kwargs
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


# transformer (kept for optional initial linear blocks)


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


# conformer


class MacaronFF(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.ff = FeedForward(dim=dim, mult=mult, dropout=dropout)
        self.scale = 0.5

    def forward(self, x):
        return self.ff(x) * self.scale


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner = dim * expansion_factor
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.net = nn.Sequential(
            RMSNorm(dim),
            Rearrange("b n d -> b d n"),
            nn.Conv1d(dim, inner * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                inner, inner, kernel_size, padding=(kernel_size - 1) // 2, groups=inner
            ),
            nn.BatchNorm1d(inner),
            nn.SiLU(inplace=True),
            nn.Conv1d(inner, dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        rotary_embed=None,
        flash_attn=True,
    ):
        super().__init__()
        self.ff1 = MacaronFF(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=attn_dropout,
            rotary_embed=rotary_embed,
            flash=flash_attn,
        )
        self.conv = ConformerConvModule(
            dim=dim,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=ff_dropout,
        )
        self.ff2 = MacaronFF(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.out_norm = RMSNorm(dim)

    def forward(self, x):
        x = x + self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + self.ff2(x)
        return self.out_norm(x)


class Conformer(Module):
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
        rotary_embed=None,
        flash_attn=True,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        norm_output=True,
    ):
        super().__init__()
        self.layers = ModuleList(
            [
                ConformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    rotary_embed=rotary_embed,
                    flash_attn=flash_attn,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)


# bandsplit module


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
        return torch.stack(outs, dim=-2)


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
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)
        outs = []
        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)
        return torch.cat(outs, dim=-1)


# main class

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


class BSConformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_conformer_depth=2,
        freq_conformer_depth=2,
        linear_conformer_depth=0,
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
        zero_dc=True,
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
        # conformer-specific
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
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

        # rotary embeddings per axis
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        # build per-depth blocks: optional linear -> time conformer -> freq conformer
        for _ in range(depth):
            modules = []

            if linear_conformer_depth > 0:
                modules.append(
                    Transformer(
                        depth=linear_conformer_depth,
                        linear_attn=True,
                        **transformer_kwargs,
                    )
                )

            modules.append(
                Conformer(
                    depth=time_conformer_depth,
                    rotary_embed=time_rotary_embed,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    **transformer_kwargs,
                )
            )

            modules.append(
                Conformer(
                    depth=freq_conformer_depth,
                    rotary_embed=freq_rotary_embed,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    **transformer_kwargs,
                )
            )

            self.layers.append(nn.ModuleList(modules))

        self.final_norm = RMSNorm(dim)

        # STFT parameters
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        self.stft_window_fn = partial(
            default(stft_window_fn, torch.hann_window), stft_win_length
        )

        # derive number of freq bins from STFT to validate band split
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

        # band split and mask estimator
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

        # options and multi-res STFT loss settings
        self.zero_dc = zero_dc
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """
        einops
        b - batch
        f - freq
        t - time
        s - audio channel (1 mono, 2 stereo)
        n - number of stems
        c - complex (2)
        d - feature dimension
        """
        device = raw_audio.device
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), "stereo True requires 2 channels; mono requires 1 channel"

        # STFT
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

        # merge stereo / mono into frequency leading dimension for band splitting
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        # flatten complex into "frequency"
        x = rearrange(stft_repr, "b f t c -> b t (f c)")

        # band split to per-band features
        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial hierarchical encoding: [optional linear] -> time conformer -> freq conformer
        store = [None] * len(self.layers)

        for i, block in enumerate(self.layers):
            if len(block) == 3:
                linear_transformer, time_encoder, freq_encoder = block

                x, ft_ps = pack([x], "b * d")
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_encoder, freq_encoder = block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # time-axis
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            if self.use_torch_checkpoint:
                x = checkpoint(time_encoder, x, use_reentrant=False)
            else:
                x = time_encoder(x)
            (x,) = unpack(x, ps, "* t d")

            # freq-axis
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            if self.use_torch_checkpoint:
                x = checkpoint(freq_encoder, x, use_reentrant=False)
            else:
                x = freq_encoder(x)
            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # masks
        num_stems = len(self.mask_estimators)
        if self.use_torch_checkpoint:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        # modulate complex STFT with masks
        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)
        stft_repr = stft_repr * mask

        # iSTFT
        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        if self.zero_dc:
            stft_repr = stft_repr.index_fill(1, tensor(0, device=device), 0.0)

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

        # optional loss
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
