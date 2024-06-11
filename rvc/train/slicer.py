import numpy as np


class Slicer:
    """
    A class for slicing audio waveforms into segments based on silence detection.

    Attributes:
        sr (int): Sampling rate of the audio waveform.
        threshold (float): RMS threshold for silence detection, in dB.
        min_length (int): Minimum length of a segment, in milliseconds.
        min_interval (int): Minimum interval between segments, in milliseconds.
        hop_size (int): Hop size for RMS calculation, in milliseconds.
        max_sil_kept (int): Maximum length of silence to keep at the beginning or end of a segment, in milliseconds.

    Methods:
        slice(waveform): Slices the given waveform into segments.
    """

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        """
        Initializes a Slicer object.

        Args:
            sr (int): Sampling rate of the audio waveform.
            threshold (float, optional): RMS threshold for silence detection, in dB. Defaults to -40.0.
            min_length (int, optional): Minimum length of a segment, in milliseconds. Defaults to 5000.
            min_interval (int, optional): Minimum interval between segments, in milliseconds. Defaults to 300.
            hop_size (int, optional): Hop size for RMS calculation, in milliseconds. Defaults to 20.
            max_sil_kept (int, optional): Maximum length of silence to keep at the beginning or end of a segment, in milliseconds. Defaults to 5000.

        Raises:
            ValueError: If the input parameters are not valid.
        """
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size is required")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size is required")

        # Convert time-based parameters to sample-based parameters
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        """
        Applies a slice to the waveform.

        Args:
            waveform (numpy.ndarray): The waveform to slice.
            begin (int): Start frame index.
            end (int): End frame index.

        Returns:
            numpy.ndarray: The sliced waveform.
        """
        start_idx = begin * self.hop_size
        if len(waveform.shape) > 1:
            end_idx = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, start_idx:end_idx]
        else:
            end_idx = min(waveform.shape[0], end * self.hop_size)
            return waveform[start_idx:end_idx]

    def slice(self, waveform):
        """
        Slices the given waveform into segments.

        Args:
            waveform (numpy.ndarray): The waveform to slice.

        Returns:
            list: A list of sliced waveforms.
        """
        # Calculate RMS for each frame
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)

        # Detect silence segments and mark them
        sil_tags = []
        silence_start, clip_start = None, 0
        for i, rms in enumerate(rms_list):
            # If current frame is silent
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            # If current frame is not silent
            if silence_start is None:
                continue

            # Check if current silence segment is leading silence or need to slice
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )

            # If not leading silence and not need to slice middle
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            # Handle different cases of silence segments
            if i - silence_start <= self.max_sil_kept:
                # Short silence
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                # Medium silence
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                # Long silence
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        # Handle trailing silence
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Extract segments based on silence tags
        if not sil_tags:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )

            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )

            return chunks


def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    """
    Calculates the root mean square (RMS) of a waveform.

    Args:
        y (numpy.ndarray): The waveform.
        frame_length (int, optional): The length of the frame in samples. Defaults to 2048.
        hop_length (int, optional): The hop length between frames in samples. Defaults to 512.
        pad_mode (str, optional): The padding mode used for the waveform. Defaults to "constant".

    Returns:
        numpy.ndarray: The RMS values for each frame.
    """
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)
