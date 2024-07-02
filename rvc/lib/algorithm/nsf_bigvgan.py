import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from rvc.lib.algorithm.alias.act import SnakeAlias


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
            self,
            samp_rate,
            harmonic_num=0,
            sine_amp=0.1,
            noise_std=0.003,
            voiced_threshold=0,
            flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
            self,
            sampling_rate=32000,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()
        harmonic_num = 10
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_tanh = torch.nn.Tanh()
        self.register_buffer('merge_w', torch.FloatTensor([[
            0.2942, -0.2243, 0.0033, -0.0056, -0.0020, -0.0046,
            0.0221, -0.0083, -0.0241, -0.0036, -0.0581]]))
        self.register_buffer('merge_b', torch.FloatTensor([0.0008]))

    def forward(self, x):
        """
        Sine_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        """
        # source for harmonic branch
        sine_wavs = self.l_sin_gen(x)
        self.merge_w = self.merge_w.to(sine_wavs.dtype) #added
        sine_wavs = nn.functional.linear(
            sine_wavs, self.merge_w) + self.merge_b
        sine_merge = self.l_tanh(sine_wavs)
        return sine_merge


class AMPBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        # total number of conv layers
        self.num_layers = len(self.convs1) + len(self.convs2)

        # periodic nonlinearity with snakebeta function and anti-aliasing
        self.activations = nn.ModuleList([
            SnakeAlias(channels) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class SpeakerAdapter(nn.Module):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        y = y.transpose(1, -1)
        return y


class GeneratorBigVgan(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_kernel_sizes, upsample_input,
                 upsample_initial_channel, sampling_rate, spk_dim):
        super(GeneratorBigVgan, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = SpeakerAdapter(spk_dim, upsample_input)
        # pre conv
        self.conv_pre = Conv1d(upsample_input,
                               upsample_initial_channel, 7, 1, padding=3)
        # nsf
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=sampling_rate)
        self.noise_convs = nn.ModuleList()
        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2)
                )
            )
            # nsf
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    Conv1d(
                        1,
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    Conv1d(1, upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=1)
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.activation_post = SnakeAlias(ch)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # weight initialization
        self.ups.apply(init_weights)

    def forward(self, x, f0, g):
        if g.size(-1) == 1:
            speaker_embedding = g.squeeze(-1)
        else:
            speaker_embedding = g
        # Perturbation
        x = x + torch.randn_like(x)
        # adapter
        x = self.adapter(x, speaker_embedding=speaker_embedding)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)
            # nsf
            #har_source = har_source.to(torch.float32)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def eval(self, inference=False):
        super(GeneratorBigVgan, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, x, har_source, g):
        # adapter
        x = self.adapter(x, g)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
