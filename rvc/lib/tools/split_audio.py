from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import numpy as np
import re
import os

from rvc.lib.utils import format_title


def process_audio(file_path):
    try:
        # load audio file
        song = AudioSegment.from_file(file_path)

        # set silence threshold and duration
        silence_thresh = -70  # dB
        min_silence_len = 750  # ms, adjust as needed

        # detect nonsilent parts
        nonsilent_parts = detect_nonsilent(song, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

        # Create a new directory to store chunks
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        file_name = format_title(file_name)
        new_dir_path = os.path.join(file_dir, file_name)
        os.makedirs(new_dir_path, exist_ok=True)

        # Check if timestamps file exists, if so delete it
        timestamps_file = os.path.join(file_dir, f"{file_name}_timestamps.txt")
        if os.path.isfile(timestamps_file):
            os.remove(timestamps_file)

        # export chunks and save start times
        segment_count = 0
        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = song[start_i:end_i]
            chunk_file_path = os.path.join(new_dir_path, f"chunk{i}.wav")
            chunk.export(chunk_file_path, format="wav")

            print(f"Segment {i} created!")
            segment_count += 1

            # write start times to file
            with open(timestamps_file, "a", encoding="utf-8") as f:
                f.write(f"{chunk_file_path} starts at {start_i} ms\n")

        print(f"Total segments created: {segment_count}")
        print(f"Split all chunks for {file_path} successfully!")

        return "Finish", new_dir_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", None


def merge_audio(timestamps_file):
    try:
        # Extract prefix from the timestamps filename
        prefix = os.path.basename(timestamps_file).replace('_timestamps.txt', '')
        timestamps_dir = os.path.dirname(timestamps_file)

        # Open the timestamps file
        with open(timestamps_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Initialize empty list to hold audio segments
        audio_segments = []
        last_end_time = 0

        print(f"Processing file: {timestamps_file}")

        for line in lines:
            # Extract filename and start time from line
            match = re.search(r"(chunk\d+.wav) starts at (\d+) ms", line)
            if match:
                filename, start_time = match.groups()
                start_time = int(start_time)

                # Construct the complete path to the chunk file
                chunk_file = os.path.join(timestamps_dir, prefix, filename)

                # Add silence from last_end_time to start_time
                silence_duration = max(start_time - last_end_time, 0)
                silence = AudioSegment.silent(duration=silence_duration)
                audio_segments.append(silence)

                # Load audio file and append to list
                audio = AudioSegment.from_wav(chunk_file)
                audio_segments.append(audio)

                # Update last_end_time
                last_end_time = start_time + len(audio)

                print(f"Processed chunk: {chunk_file}")

        # Concatenate all audio_segments and export
        merged_audio = sum(audio_segments)
        merged_audio_np = np.array(merged_audio.get_array_of_samples())
        #print(f"Exported merged file: {merged_filename}\n")
        return merged_audio.frame_rate, merged_audio_np

    except Exception as e:
        print(f"An error occurred: {e}")