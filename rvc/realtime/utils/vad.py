import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(
        self,
        sensitivity_mode=1,
        sample_rate=16000,
        frame_duration_ms=30,
        hangover_ms=200,
    ):
        """
        Initializes the VADProcessor.

        Args:
            sensitivity_mode (int): VAD sensitivity (0-3). 3 is most aggressive.
            sample_rate (int): Sample rate of the audio. Must be 8000, 16000, 32000, or 48000 Hz.
                               WebRTC VAD internally works best with 16000 Hz.
            frame_duration_ms (int): Duration of each audio frame in ms. Must be 10, 20, or 30.
        """

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("VAD sample rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")

        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.hangover_samples = int(sample_rate * hangover_ms / 1000)
        self.remaining_samples = 0
        self.noise_rms = 0.0001
        self.window = np.hanning(self.frame_length)
        self.fft_size = 1 << (2 * self.frame_length - 1).bit_length()
        frequencies = np.fft.rfftfreq(self.fft_size, 1 / sample_rate)
        self.voice_band = (frequencies >= 70) & (frequencies <= 4000)
        self.lags = np.arange(
            max(1, sample_rate // 1100), min(self.frame_length - 1, sample_rate // 70)
        )

    def _is_sustained_voice(self, frame):
        """Keep periodic vowels/singing which conversational WebRTC VAD rejects."""
        frame = frame.astype(np.float64) / 32767
        frame -= frame.mean()
        rms = np.sqrt(np.mean(frame**2))
        if rms < max(0.001, self.noise_rms * 3):
            return False
        spectrum = np.abs(np.fft.rfft(frame * self.window, self.fft_size)) ** 2
        if spectrum[self.voice_band].sum() < spectrum.sum() * 0.75:
            return False  # Reject DC and low-frequency electrical/fan hum.
        transformed = np.fft.rfft(frame, self.fft_size)
        correlation = np.fft.irfft(np.abs(transformed) ** 2, self.fft_size)
        energy = np.concatenate(([0.0], np.cumsum(frame**2)))
        denominator = np.sqrt(energy[-self.lags - 1] * (energy[-1] - energy[self.lags]))
        periodicity = correlation[self.lags] / np.maximum(denominator, 1e-12)
        return bool(periodicity.max() >= 0.8)

    def is_speech(self, audio_chunk_float32):
        """
        Detects if the given audio chunk contains speech.

        Args:
            audio_chunk_float32 (np.ndarray): A chunk of audio data in float32 format, mono.
                                              The sample rate must match the one VAD was initialized with.

        Returns:
            bool: True if speech is detected in the chunk, False otherwise.
        """

        if audio_chunk_float32.ndim > 1 and audio_chunk_float32.shape[1] == 1:
            audio_chunk_float32 = audio_chunk_float32.flatten()
        elif audio_chunk_float32.ndim > 1:
            # If stereo, average to mono. This is a simple approach.
            # For better results, ensure mono input from the source.
            print("VAD Warning: Received stereo audio, averaging to mono.")
            audio_chunk_float32 = np.mean(audio_chunk_float32, axis=1)

        if audio_chunk_float32.size == 0:
            return False
        chunk_samples = len(audio_chunk_float32)
        # Convert float32 audio to int16 PCM
        # WebRTC VAD expects 16-bit linear PCM audio.
        if np.max(np.abs(audio_chunk_float32)) > 1.0:
            # print(
            #     f"VAD Warning: Input audio chunk has values outside [-1.0, 1.0]: min={np.min(audio_chunk_float32)}, max={np.max(audio_chunk_float32)}. Clipping."
            # )
            audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)

        audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

        # Analyze the last partial frame as well, so word endings are retained.
        padding_size = (-len(audio_chunk_int16)) % self.frame_length
        if padding_size:
            audio_chunk_int16 = np.pad(audio_chunk_int16, (0, padding_size))
        num_frames = len(audio_chunk_int16) // self.frame_length
        try:
            for i in range(num_frames):
                start = i * self.frame_length
                end = start + self.frame_length
                frame = audio_chunk_int16[start:end]
                # The VAD expects bytes, not a NumPy array.
                if self.vad.is_speech(
                    frame.tobytes(), self.sample_rate
                ) or self._is_sustained_voice(frame):
                    self.remaining_samples = self.hangover_samples
                    return True  # Speech detected in at least one frame
            keep_tail = self.remaining_samples > 0
            self.remaining_samples = max(0, self.remaining_samples - chunk_samples)
            if not keep_tail:
                rms = float(np.sqrt(np.mean(audio_chunk_float32**2)))
                self.noise_rms = 0.95 * self.noise_rms + 0.05 * min(rms, 0.01)
            return keep_tail
        except Exception as e:
            # webrtcvad can sometimes throw "Error talking to VAD" or similar
            # if frame length is not perfect.
            print(
                f"VAD processing error: {e}. Chunk length: {len(audio_chunk_int16)}, Frame length: {self.frame_length}"
            )
            # Fallback: assume no speech on error to avoid processing noise
            return False
