from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import numpy as np
import re
import os

from rvc.lib.utils import format_title


def process_audio(np_audio, sample_rate=44100, sample_width=4, channels=1):
    try:
        # Convert numpy array to AudioSegment
        audio_segment = AudioSegment(
            np_audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        # Set silence threshold and duration
        silence_thresh = -70  # dB
        min_silence_len = 750  # ms, adjust as needed

        # Detect non-silent parts
        nonsilent_parts = detect_nonsilent(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )

        segments = []
        timestamps = []

        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = audio_segment[start_i:end_i]

            # Convert chunk to numpy array
            chunk_np = np.array(chunk.get_array_of_samples())

            segments.append(chunk_np)
            timestamps.append((start_i, end_i))

            print(f"Segment {i} created!")

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
