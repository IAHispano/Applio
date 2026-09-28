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
    - audio_segments_org (list of np.ndarray): Original segments. Callers still pass them. Timing comes from the intervals.
    - audio_segments_new (list of np.ndarray): The non-silent audio segments (at sr_new).
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio
    - sr_new (int): The sample rate of the model
    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    merged_audio = np.array([], dtype=audio_segments_new[0].dtype)
    sr_ratio = sr_new / sr_orig

    for i, (start, end) in enumerate(intervals):
        start_new = int(start * sr_ratio)
        end_new = int(end * sr_ratio)

        # A longer converted chunk stays at its start. Padding in front of it
        # delayed the voice by the extra length.
        if start_new > len(merged_audio):
            gap = np.zeros(start_new - len(merged_audio), dtype=merged_audio.dtype)
            merged_audio = np.concatenate((merged_audio, gap))

        merged_audio = np.concatenate((merged_audio, audio_segments_new[i]))

        if len(merged_audio) < end_new:
            missing = np.zeros(end_new - len(merged_audio), dtype=merged_audio.dtype)
            merged_audio = np.concatenate((merged_audio, missing))

    return merged_audio
