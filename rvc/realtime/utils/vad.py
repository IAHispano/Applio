import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(self, sensitivity_mode=3, sample_rate=16000, frame_duration_ms=30):
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
        # print(f"VAD Initialized: SR={sample_rate}, Frame Duration={frame_duration_ms}ms, Frame Length={self.frame_length} samples")

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

        # Convert float32 audio to int16 PCM
        # WebRTC VAD expects 16-bit linear PCM audio.
        if np.max(np.abs(audio_chunk_float32)) > 1.0:
            print(
                f"VAD Warning: Input audio chunk has values outside [-1.0, 1.0]: min={np.min(audio_chunk_float32)}, max={np.max(audio_chunk_float32)}. Clipping."
            )
            audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)

        audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

        num_frames = len(audio_chunk_int16) // self.frame_length
        if num_frames == 0 and len(audio_chunk_int16) > 0:
            # If the chunk is smaller than one frame, pad it for VAD analysis
            # This might not be ideal but handles small initial chunks
            padding = np.zeros(
                self.frame_length - len(audio_chunk_int16), dtype=np.int16
            )
            audio_chunk_int16 = np.concatenate((audio_chunk_int16, padding))
            num_frames = 1
        elif num_frames == 0 and len(audio_chunk_int16) == 0:
            return False  # Empty chunk

        try:
            for i in range(num_frames):
                start = i * self.frame_length
                end = start + self.frame_length
                frame = audio_chunk_int16[start:end]
                # The VAD expects bytes, not a NumPy array.
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    return True  # Speech detected in at least one frame
            return False  # No speech detected in any frame
        except Exception as e:
            # webrtcvad can sometimes throw "Error talking to VAD" or similar
            # if frame length is not perfect.
            print(
                f"VAD processing error: {e}. Chunk length: {len(audio_chunk_int16)}, Frame length: {self.frame_length}"
            )
            # Fallback: assume no speech on error to avoid processing noise
            return False
