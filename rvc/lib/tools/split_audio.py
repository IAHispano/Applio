import numpy as np
import librosa


def process_audio(audio, sr=16000, silence_thresh=-60, min_silence_len=250, lookahead=200, falloff=200):
    """
    Splits an audio signal into segments using a fixed frame size and hop size with lookahead and falloff.

    Parameters:
    - audio (np.ndarray): The audio signal to split.
    - sr (int): The sample rate of the input audio (default is 16000).
    - silence_thresh (int): Silence threshold (default = -60 dB).
    - min_silence_len (int): Minimum silence duration (default 250 ms).
    - lookahead (int): Lookahead duration in milliseconds (default 200 ms).
    - falloff (int): Falloff duration in milliseconds (default 200 ms).

    Returns:
    - list of np.ndarray: A list of audio segments.
    - np.ndarray: The intervals where the audio was split.
    """
    # Calculate frame and hop lengths
    frame_length = int(min_silence_len / 1000 * sr)
    hop_length = frame_length // 2
    lookahead_samples = int(lookahead / 1000 * sr)  # Convert lookahead to samples
    falloff_samples = int(falloff / 1000 * sr)  # Convert falloff to samples

    # Detect non-silent intervals
    intervals = librosa.effects.split(
        audio, top_db=-silence_thresh, frame_length=frame_length, hop_length=hop_length
    )

    # Adjust intervals with lookahead and falloff
    adjusted_intervals = []
    for start, end in intervals:
        adjusted_start = max(0, start - lookahead_samples)  # Avoid negative index
        adjusted_end = end  # Start with the original end

        # Check for silence after the current end
        silence_duration = 0  # Duration of silence below threshold
        for i in range(end, len(audio), hop_length):
            if audio[i:i + hop_length].mean() < librosa.db_to_amplitude(-silence_thresh):
                silence_duration += hop_length
            else:
                silence_duration = 0  # Reset if audio goes above threshold
            
            # If silence exceeds falloff duration, cut the segment
            if silence_duration >= falloff_samples:
                adjusted_end = i
                break

        adjusted_intervals.append((adjusted_start, adjusted_end))

    # Extract audio segments using the adjusted intervals
    audio_segments = [audio[start:end] for start, end in adjusted_intervals]

    return audio_segments, adjusted_intervals


def merge_audio(audio_segments, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence.

    Parameters:
    - audio_segments (list of np.ndarray): The non-silent audio segments.
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio
    - sr_new (int): The sample rate of the model

    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    sr_ratio = sr_new / sr_orig if sr_new > sr_orig else 1.0

    merged_audio = np.zeros(
        int(intervals[0][0] * sr_ratio if intervals[0][0] > 0 else 0),
        dtype=audio_segments[0].dtype
    )

    merged_audio = np.concatenate((merged_audio, audio_segments[0]))

    for i in range(1, len(intervals)):
        silence_duration = int((intervals[i][0] - intervals[i - 1][1]) * sr_ratio)
        silence = np.zeros(silence_duration, dtype=audio_segments[0].dtype)
        merged_audio = np.concatenate((merged_audio, silence, audio_segments[i]))

    return merged_audio
