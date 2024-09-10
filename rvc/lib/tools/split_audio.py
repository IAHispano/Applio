from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import numpy as np
import re
import os
import librosa

from rvc.lib.utils import format_title


def process_audio(audio_path, sr=44100, silence_thresh=-70, min_silence_len=750):
    try:
        # Convert min_silence_len from ms to frames
        min_silence_frames = int((min_silence_len / 1000) * sr)

        # Detect non-silent parts
        intervals = librosa.effects.split(
            audio_path,
            top_db=-silence_thresh,
            frame_length=min_silence_frames,
            hop_length=min_silence_frames // 2,
        )

        segments = []
        timestamps = []

        # Add the first silence segment if any
        if intervals[0][0] > 0:
            segments.append(audio_path[: intervals[0][0]])
            timestamps.append((0, intervals[0][0] / sr))

        for i, interval in enumerate(intervals):
            start, end = interval
            chunk = audio_path[start:end]

            segments.append(chunk)
            timestamps.append((start / sr, end / sr))  # Convert to seconds

            print(f"Segment {i} created!")

            # Add the next silence segment if any
            if i < len(intervals) - 1 and end < intervals[i + 1][0]:
                segments.append(audio_path[end : intervals[i + 1][0]])
                timestamps.append((end / sr, intervals[i + 1][0] / sr))

        # Add the last silence segment if any
        if intervals[-1][1] < len(audio_path):
            segments.append(audio_path[intervals[-1][1] :])
            timestamps.append((intervals[-1][1] / sr, len(audio_path) / sr))

        print(f"Total segments created: {len(segments)}")

        return "Finish", segments, timestamps

    except Exception as error:
        print(f"An error occurred while splitting the audio: {error}")
        return "Error", None, None


def merge_audio(segments, timestamps, sample_rate=44100):
    try:
        audio_segments = []
        last_end_time = 0

        print("Starting audio merging process")

        for i, (start_time, end_time) in enumerate(timestamps):
            silence_duration = max(start_time - last_end_time, 0)
            silence = AudioSegment.silent(
                duration=silence_duration, frame_rate=sample_rate
            )
            audio_segments.append(silence)

            segment_audio = AudioSegment(
                segments[i].tobytes(),
                frame_rate=sample_rate,
                sample_width=segments[i].dtype.itemsize,
                channels=1 if segments[i].ndim == 1 else segments[i].shape[1],
            )
            audio_segments.append(segment_audio)

            last_end_time = end_time

            print(f"Processed segment {i+1}/{len(timestamps)}")

        merged_audio = sum(audio_segments)

        merged_audio_np = np.array(merged_audio.get_array_of_samples())

        print("Audio merging completed successfully")
        return sample_rate, merged_audio_np

    except Exception as error:
        print(f"An error occurred during audio merging: {error}")
        return None, None
