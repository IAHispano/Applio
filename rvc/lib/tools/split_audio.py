import numpy as np
import librosa

def apply_fade(audio_segment, sr, fade_ms=5):
    """
    Applies a fade-in and fade-out effect to the audio segment.

    Parameters:
    - audio_segment (np.ndarray): The audio segment to process.
    - sr (int): The sample rate of the audio.
    - fade_ms (int): Duration of the fade-in and fade-out in milliseconds.

    Returns:
    - np.ndarray: The audio segment with fades applied.
    """
    fade_samples = int(fade_ms / 1000 * sr)
    
    # Fade-in: Linear ramp from 0 to 1 over the first `fade_samples`
    fade_in_curve = np.linspace(0, 1, fade_samples)
    audio_segment[:fade_samples] *= fade_in_curve
    
    # Fade-out: Linear ramp from 1 to 0 over the last `fade_samples`
    fade_out_curve = np.linspace(1, 0, fade_samples)
    audio_segment[-fade_samples:] *= fade_out_curve
    
    return audio_segment

def process_audio(audio, sr=16000, silence_thresh=-60, min_silence_len=250):
    """
    Splits an audio signal into segments using a fixed frame size and hop size,
    and applies a 5ms fade-in and fade-out to each segment.

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
    
    audio_segments = []
    for start, end in intervals:
        segment = audio[start:end]
        # Apply fade-in and fade-out to each segment
        segment = apply_fade(segment, sr)
        audio_segments.append(segment)

    return audio_segments, intervals

def merge_audio(audio_segments, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence,
    and applies a 5ms fade-in and fade-out to each segment before merging.

    Parameters:
    - audio_segments (list of np.ndarray): The non-silent audio segments.
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio.
    - sr_new (int): The sample rate of the model.

    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    sr_ratio = sr_new / sr_orig if sr_new > sr_orig else 1.0

    merged_audio = np.zeros(
        int(intervals[0][0] * sr_ratio if intervals[0][0] > 0 else 0),
        dtype=audio_segments[0].dtype,
    )

    merged_audio = np.concatenate((merged_audio, apply_fade(audio_segments[0], sr_orig)))

    for i in range(1, len(intervals)):
        silence_duration = int((intervals[i][0] - intervals[i - 1][1]) * sr_ratio)
        silence = np.zeros(silence_duration, dtype=audio_segments[0].dtype)
        merged_audio = np.concatenate((merged_audio, silence, apply_fade(audio_segments[i], sr_orig)))

    return merged_audio