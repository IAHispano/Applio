"""Spectral DSP helpers for Applio UVR (vendored, VR/MDX/Demucs paths only)."""

import io
import math
import traceback

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, hilbert


class InvalidAudioDataError(Exception):
    pass


def crop_center(h1, h2):
    """
    This function crops the center of the first input tensor to match the size of the second input tensor.
    It is used to ensure that the two tensors have the same size in the time dimension.
    """
    h1_shape = h1.size()
    h2_shape = h2.size()

    # If the time dimensions are already equal, return the first tensor as is
    if h1_shape[3] == h2_shape[3]:
        return h1
    # If the time dimension of the first tensor is smaller, raise an error
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # Calculate the start and end indices for cropping
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    # Crop the first tensor
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def preprocess(X_spec):
    """
    This function preprocesses a spectrogram by separating it into magnitude and phase components.
    This is a common preprocessing step in audio processing tasks.
    """
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    return X_mag, X_phase


def make_padding(width, cropsize, offset):
    """
    This function calculates the padding needed to make the width of an image divisible by the crop size.
    It is used in the process of splitting an image into smaller patches.
    """
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def normalize(wave, max_peak=1.0, min_peak=None):
    """Normalize (or amplify) audio waveform to a specified peak value.

    Args:
        wave (array-like): Audio waveform.
        max_peak (float): Maximum peak value for normalization.

    Returns:
        array-like: Normalized or original waveform.
    """
    wave = np.asarray(wave)
    if wave.size == 0:
        raise InvalidAudioDataError("Audio data is empty")

    maxv = np.abs(wave).max()
    if not np.isfinite(maxv):
        raise InvalidAudioDataError("Audio data must contain only finite values")

    if maxv > max_peak:
        wave *= max_peak / maxv
    elif min_peak is not None and 0 < maxv < min_peak:
        wave *= min_peak / maxv

    return wave


def auto_transpose(audio_array: np.ndarray):
    """
    Ensure that the audio array is in the (channels, samples) format.

    Parameters:
        audio_array (ndarray): Input audio array.

    Returns:
        ndarray: Transposed audio array if necessary.
    """

    # If the second dimension is 2 (indicating stereo channels), transpose the array
    if audio_array.shape[1] == 2:
        return audio_array.T
    return audio_array


def write_array_to_mem(audio_data, subtype):
    if isinstance(audio_data, np.ndarray):
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, 44100, subtype=subtype, format="WAV")
        audio_buffer.seek(0)
        return audio_buffer
    else:
        return audio_data


def reduce_vocal_aggressively(X, y, softmask):
    v = X - y
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(v)

    v_mask = v_mag_tmp > y_mag_tmp
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)

    return y_mag * np.exp(1.0j * np.angle(y))


def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    mask = y_mask

    try:
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)
        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight[:, :, s : s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size

                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size : e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size

                weight[:, :, s + fade_size : e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask

        mask = y_mask
    except Exception as e:
        error_name = f"{type(e).__name__}"
        traceback_text = "".join(traceback.format_tb(e.__traceback__))
        message = f'{error_name}: "{e}"\n{traceback_text}"'
        print("Post Process Failed: ", message)

    return mask


def convert_channels(spec, mp, band):
    cc = mp.param["band"][band].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else:
        return spec

    return np.asfortranarray([spec_left, spec_right])


def combine_spectrograms(specs, mp, is_v51_model=False):
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter

    if mp.param["pre_filter_start"] > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(
                spec_c.shape[1],
                mp.param["pre_filter_start"],
                mp.param["pre_filter_stop"],
            )
        else:
            if bands_n == 1:
                spec_c = fft_lp_filter(
                    spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
                )
            else:
                gp = 1
                for b in range(
                    mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
                ):
                    g = math.pow(
                        10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                    )
                    gp = g
                    spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):

    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])

    if not is_v51_model:
        if mp.param["reverse"]:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp.param["mid_side"]:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param["mid_side_b2"]:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model:
        spec = convert_channels(spec, mp, band)

    return spec


def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if is_v51_model:
        cc = mp.param["band"][band].get("convert_channels")
        if "mid_side_c" == cc:
            return np.asfortranarray(
                [
                    np.subtract(wave_left / 1.0625, wave_right / 4.25),
                    np.add(wave_right / 1.0625, wave_left / 4.25),
                ]
            )
        elif "mid_side" == cc:
            return np.asfortranarray(
                [
                    np.add(wave_left, wave_right / 2),
                    np.subtract(wave_left, wave_right / 2),
                ]
            )
        elif "stereo_n" == cc:
            return np.asfortranarray(
                [
                    np.subtract(wave_left, wave_right * 0.25),
                    np.subtract(wave_right, wave_left * 0.25),
                ]
            )
    else:
        if mp.param["reverse"]:
            return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp.param["mid_side"]:
            return np.asfortranarray(
                [
                    np.add(wave_left, wave_right / 2),
                    np.subtract(wave_left, wave_right / 2),
                ]
            )
        elif mp.param["mid_side_b2"]:
            return np.asfortranarray(
                [
                    np.add(wave_right / 1.25, 0.4 * wave_left),
                    np.subtract(wave_left / 1.25, 0.4 * wave_right),
                ]
            )

    return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(
    spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False
):
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.zeros(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp["hpf_start"] > 0:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(
                        spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
            else:
                wave = np.add(
                    wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                if is_v51_model:
                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"]
                    )
                else:
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                try:
                    wave = librosa.resample(
                        spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model),
                        orig_sr=bp["sr"],
                        target_sr=sr,
                        res_type=wav_resolution,
                    )
                except ValueError as e:
                    print(f"Error during resampling: {e}")
                    print(
                        f"Spec_s shape: {spec_s.shape}, SR: {sr}, Res type: {wav_resolution}"
                    )

            else:  # mid
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(
                        spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                    spec_s *= get_lp_filter_mask(
                        spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"]
                    )
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                wave2 = np.add(
                    wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                )

                try:
                    wave = librosa.resample(
                        wave2, orig_sr=bp["sr"], target_sr=sr, res_type=wav_resolution
                    )
                except ValueError as e:
                    print(f"Error during resampling: {e}")
                    print(
                        f"Spec_s shape: {spec_s.shape}, SR: {sr}, Res type: {wav_resolution}"
                    )

    return wave


def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate(
        [
            np.ones((bin_start - 1, 1)),
            np.linspace(1, 0, bin_stop - bin_start + 1)[:, None],
            np.zeros((n_bins - bin_stop, 1)),
        ],
        axis=0,
    )

    return mask


def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate(
        [
            np.zeros((bin_stop + 1, 1)),
            np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None],
            np.ones((n_bins - bin_start - 2, 1)),
        ],
        axis=0,
    )

    return mask


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr

        if np.any(aggr > 10) or np.any(aggr < -10):
            print(f"Warning: Extreme aggressiveness values detected: {aggr}")

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        for ch in range(2):
            mask[ch, : aggressiveness["split_bin"]] = np.power(
                mask[ch, : aggressiveness["split_bin"]], 1 + aggr[ch] / 3
            )
            mask[ch, aggressiveness["split_bin"] :] = np.power(
                mask[ch, aggressiveness["split_bin"] :], 1 + aggr[ch]
            )

    return mask


def spectrogram_to_wave_no_mp(spec, n_fft=2048, hop_length=1024):
    wave = librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)

    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])

    return wave


def wave_to_spectrogram_no_mp(wave):

    spec = librosa.stft(wave, n_fft=2048, hop_length=1024)

    if spec.ndim == 1:
        spec = np.asfortranarray([spec, spec])

    return spec


def invert_audio(specs, invert_p=True):

    ln = min([specs[0].shape[2], specs[1].shape[2]])
    specs[0] = specs[0][:, :, :ln]
    specs[1] = specs[1][:, :, :ln]

    if invert_p:
        X_mag = np.abs(specs[0])
        y_mag = np.abs(specs[1])
        max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)
        v_spec = specs[1] - max_mag * np.exp(1.0j * np.angle(specs[0]))
    else:
        specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
        v_spec = specs[0] - specs[1]

    return v_spec


def invert_stem(mixture, stem):
    mixture = wave_to_spectrogram_no_mp(mixture)
    stem = wave_to_spectrogram_no_mp(stem)
    output = spectrogram_to_wave_no_mp(invert_audio([mixture, stem]))

    return -output.T


def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = target_dim - x_dim
        pad_tuple = (0, pad_value)
        padding_list.append(pad_tuple)

    return np.pad(x, tuple(padding_list), mode="constant")


def reshape_sources(wav_1: np.ndarray, wav_2: np.ndarray):

    if wav_1.shape > wav_2.shape:
        wav_2 = to_shape(wav_2, wav_1.shape)
    if wav_1.shape < wav_2.shape:
        ln = min([wav_1.shape[1], wav_2.shape[1]])
        wav_2 = wav_2[:, :ln]

    ln = min([wav_1.shape[1], wav_2.shape[1]])
    wav_1 = wav_1[:, :ln]
    wav_2 = wav_2[:, :ln]

    return wav_2
