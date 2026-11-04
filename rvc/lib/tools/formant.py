"""Phase-preserving formant filtering for files and continuous audio streams."""

import math

import numpy as np


class FormantShifter:
    """Warp the cepstral envelope while preserving the original FFT phase.

    Retain analysis and overlap-add state across callbacks of any size. Active
    processing has one frame of latency; neutral settings return the input
    unchanged. File processing flushes and removes that latency.
    """

    frame_size = 1024
    hop_size = 256

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._window = np.hanning(self.frame_size + 1)[:-1]
        self.reset()

    def reset(self):
        self._input = np.zeros(self.frame_size - self.hop_size)
        self._overlap = np.zeros(self.frame_size)
        self._weights = np.zeros(self.frame_size)
        self._output = np.zeros(self.hop_size)

    def process(self, audio, quefrency=1.0, timbre=1.0):
        quefrency, timbre = float(quefrency), float(timbre)
        if not math.isfinite(quefrency) or quefrency < 0:
            raise ValueError("Formant quefrency must be finite and non-negative")
        if not math.isfinite(timbre) or timbre <= 0:
            raise ValueError("Formant timbre must be finite and positive")
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("Formant audio must be a one-dimensional float array")
        if not audio.size:
            return audio

        # Quefrency only selects the envelope lifter; it does not itself shift
        # formants. A unity timbre (or disabled lifter) is exactly neutral.
        cutoff = min(int(quefrency * 1e-3 * self.sample_rate), self.frame_size // 2)
        if timbre == 1.0 or cutoff == 0:
            self.reset()
            return audio

        self._input = np.concatenate((self._input, audio))
        emitted = [self._output]
        bins = np.arange(self.frame_size // 2 + 1)
        window_power = self._window**2
        while self._input.size >= self.frame_size:
            spectrum = np.fft.rfft(self._input[: self.frame_size] * self._window)
            magnitude = np.abs(spectrum)
            # The envelope calculation must remain finite for silence and
            # empty spectral bins; filtering must never create signal there.
            cepstrum = np.fft.irfft(np.log(np.maximum(magnitude, 1e-12)))
            cepstrum[1:cutoff] *= 2
            cepstrum[cutoff + 1 :] = 0
            envelope = np.exp(np.fft.rfft(cepstrum).real)
            warped = np.interp(bins / timbre, bins, envelope, right=0)
            gain = warped / np.maximum(envelope, 1e-12)
            frame = np.fft.irfft(spectrum * gain)
            self._overlap += frame * self._window
            self._weights += window_power
            emitted.append(
                self._overlap[: self.hop_size]
                / np.maximum(self._weights[: self.hop_size], 1e-12)
            )
            self._overlap[: -self.hop_size] = self._overlap[self.hop_size :]
            self._overlap[-self.hop_size :] = 0
            self._weights[: -self.hop_size] = self._weights[self.hop_size :]
            self._weights[-self.hop_size :] = 0
            self._input = self._input[self.hop_size :]

        output = np.concatenate(emitted)
        self._output = output[audio.size :].copy()
        result = output[: audio.size].astype(audio.dtype)
        if not np.isfinite(result).all():
            raise ValueError("Formant processing produced non-finite audio")
        return result

    def process_file(self, audio, quefrency=1.0, timbre=1.0):
        self.reset()
        shifted = self.process(audio, quefrency, timbre)
        if shifted is audio or not len(audio):
            return shifted
        tail = self.process(
            np.zeros(self.frame_size, dtype=audio.dtype), quefrency, timbre
        )
        return np.concatenate((shifted, tail))[
            self.frame_size : self.frame_size + len(audio)
        ]
