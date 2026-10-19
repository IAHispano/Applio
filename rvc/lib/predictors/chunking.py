"""Bound offline pitch inference without changing the recording's frame clock."""

import numpy as np


def extract_pitch(audio, infer, sample_rate=16000, hop_size=160, chunk_seconds=10):
    """Stitch contextual windows on hop boundaries, retaining one final frame.

    Only the small CPU contour grows with duration. One second of context on
    either side keeps spectral padding, smoothing and recurrent state away
    from internal joins; context frames are discarded rather than averaged
    across voiced/unvoiced transitions. Short inputs retain their native path.
    """
    chunk_frames = max(1, int(chunk_seconds * sample_rate) // hop_size)
    if len(audio) <= chunk_frames * hop_size:
        return infer(audio)

    context = max(1, sample_rate // hop_size)
    frame_count = len(audio) // hop_size + 1
    result = np.empty(frame_count, dtype=np.float32)
    for start in range(0, frame_count - 1, chunk_frames):
        end = min(start + chunk_frames, frame_count - 1)
        if end == frame_count - 1:
            end = frame_count
        left = max(0, start - context)
        right = min(len(audio), (end + context) * hop_size)
        values = np.asarray(infer(audio[left * hop_size : right])).reshape(-1)
        offset = start - left
        if len(values) < offset + end - start:
            raise ValueError("Pitch extractor returned fewer frames than its audio window")
        result[start:end] = values[offset : offset + end - start]
    return result
