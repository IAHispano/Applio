import numpy as np
import librosa


def process_audio(audio, sr=16000, silence_thresh=-60, min_silence_len=250):
    """
    Splits an audio signal into segments using a fixed frame size and hop size.

    Parameters:
    - audio (np.ndarray): The audio signal to split.
    - sr (int): The sample rate of the input audio (default is 16000).
    - silence_thresh (int): Silence threshold (default =-60dB)
    - min_silence_len (int): Minimum silence duration (default 250ms).

    Returns:
    - list of np.ndarray: A list of audio segments.
    - np.ndarray: The intervals where the audio was split.
    """
    frame_length = int(min_silence_len / 1000 * sr)
    hop_length = frame_length // 2
    intervals = librosa.effects.split(
        audio, top_db=-silence_thresh, frame_length=frame_length, hop_length=hop_length
    )
    audio_segments = [audio[start:end] for start, end in intervals]

    return audio_segments, intervals


def merge_audio(audio_segments_org, audio_segments_new, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence.
    Assumes audio segments are already at sr_new.

    Parameters:
    - audio_segments_org (list of np.ndarray): The non-silent audio segments (at sr_orig).
    - audio_segments_new (list of np.ndarray): The non-silent audio segments (at sr_new).
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio
    - sr_new (int): The sample rate of the model
    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    # Collect all segments in a list first to avoid O(n²) concatenation complexity
    segments_to_merge = []
    sr_ratio = sr_new / sr_orig
    dtype = audio_segments_new[0].dtype

    for i, (start, end) in enumerate(intervals):

        start_new = int(start * sr_ratio)
        end_new = int(end * sr_ratio)

        original_duration = len(audio_segments_org[i]) / sr_orig
        new_duration = len(audio_segments_new[i]) / sr_new
        duration_diff = new_duration - original_duration

        silence_samples = int(abs(duration_diff) * sr_new)

        if i == 0 and start_new > 0:
            segments_to_merge.append(np.zeros(start_new, dtype=dtype))

        if duration_diff > 0:
            segments_to_merge.append(np.zeros(silence_samples, dtype=dtype))

        segments_to_merge.append(audio_segments_new[i])

        if duration_diff < 0:
            segments_to_merge.append(np.zeros(silence_samples, dtype=dtype))

        if i < len(intervals) - 1:
            next_start_new = int(intervals[i + 1][0] * sr_ratio)
            silence_duration = next_start_new - end_new
            if silence_duration > 0:
                segments_to_merge.append(np.zeros(silence_duration, dtype=dtype))

    # Single concatenation at the end - much more efficient
    return np.concatenate(segments_to_merge) if segments_to_merge else np.array([], dtype=dtype)
