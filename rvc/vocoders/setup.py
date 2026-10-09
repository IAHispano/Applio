"""Assembling a run: what happens once, before the loop in ``trainer.py``."""

import os

import torch


def loader_workers(requested: int) -> int:
    """``requested`` DataLoader workers, capped at the CPUs this process may
    use, the limit PyTorch warns past (Colab has 2)."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(0, min(int(requested), available))


def get_d_model(settings: dict, sample_rate: int):
    """The v3 discriminator (period and spectrogram branches), as the config's
    ``discriminator`` section sets it. An absent key keeps the preset's value;
    an empty ``d_periods`` or ``d_resolutions`` builds none of those branches."""
    from rvc.vocoders.models.discriminator import MPD_MSD_Combined

    def setting(name, default=None):
        value = settings.get(name)
        return default if value is None else value

    def optional_float(name):
        return None if setting(name) is None else float(setting(name))

    return MPD_MSD_Combined(
        bool(setting("use_spectral_norm", False)),
        use_checkpointing=False,
        version="v3",
        periods=[] if not setting("d_use_periods", True) else setting("d_periods"),
        resolutions=[] if not setting("d_use_resolutions", True) else setting("d_resolutions"),
        frequency_strides=setting("d_frequency_strides"),
        use_msd=bool(setting("d_use_msd", True)),
        use_fast_mpd=bool(setting("d_use_fast_mpd", False)),
        mrd_fp32_input=bool(setting("d_mrd_fp32_input", True)),
        wave_fp32_input=bool(setting("d_wave_fp32_input", False)),
        mrd_channels=int(setting("d_mrd_channels", 32)),
        mpd_pre_emphasis=setting("d_mpd_pre_emphasis"),
        r1_gamma=setting("d_r1_gamma"),
        r1_interval=int(setting("d_r1_interval", 16)),
        r1_batch_fraction=float(setting("d_r1_batch_fraction", 0.5)),
        r1_segment=int(setting("d_r1_segment", 6400)),
        sample_rate=int(sample_rate),
        use_univhd=bool(setting("d_use_univhd", False)),
        use_san=bool(setting("d_use_san", False)),
        univhd_n_fft=int(setting("d_univhd_n_fft", 2048)),
        univhd_hop_length=int(setting("d_univhd_hop_length", 256)),
        univhd_harmonics=int(setting("d_univhd_harmonics", 10)),
        univhd_bins_per_octave=int(setting("d_univhd_bins_per_octave", 24)),
        univhd_f_min=float(setting("d_univhd_f_min", 80.0)),
        univhd_channels=int(setting("d_univhd_channels", 32)),
        univhd_half_harmonic=bool(setting("d_univhd_half_harmonic", True)),
        univhd_max_hz=optional_float("d_univhd_max_hz"),
        univhd_weight=optional_float("d_univhd_weight"),
        msd_weight=float(setting("d_msd_weight", 1.0)),
    )


def apply_precision_policy(net_g, amp_dtype):
    """Under autocast, keep the generator's precision-critical paths in FP32:
    BF16 needs it for mantissa, FP16 for range. Sets ``fp32_residuals`` on
    every module that declares it."""
    enabled = amp_dtype in (torch.bfloat16, torch.float16)
    model = net_g.module if hasattr(net_g, "module") else net_g
    for module in model.modules():
        if hasattr(module, "fp32_residuals"):
            module.fp32_residuals = enabled


def normalize_san_weights(net_d):
    """Put every SAN direction back on the unit sphere after ``optim_d.step``,
    which moves it off. A no-op when no branch carries a SAN head."""
    model = net_d.module if hasattr(net_d, "module") else net_d
    for module in model.modules():
        normalize = getattr(module, "normalize_weight", None)
        if normalize is not None:
            normalize()
