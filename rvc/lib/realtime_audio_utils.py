import sounddevice as sd
import numpy as np

def list_audio_devices():
    """Lists available audio input and output devices."""
    devices = sd.query_devices()
    input_devices = []
    output_devices = []
    for i, device in enumerate(devices):
        device_label = f"{i}: {device['name']} (Channels: {device['max_input_channels']} in, {device['max_output_channels']} out)"
        if device['max_input_channels'] > 0:
            input_devices.append({'id': i, 'name': device['name'], 'channels': device['max_input_channels'], 'label': device_label})
        if device['max_output_channels'] > 0:
            output_devices.append({'id': i, 'name': device['name'], 'channels': device['max_output_channels'], 'label': device_label})
    return input_devices, output_devices

def get_device_info(device_id):
    """Gets detailed information for a specific device."""
    try:
        return sd.query_devices(device_id)
    except Exception as e:
        print(f"Error querying device {device_id}: {e}")
        return None

_stream = None
_is_streaming = False

def audio_callback(indata, outdata, frames, time, status):
    """
    This is called (from a separate thread) for each audio block.
    """
    global process_audio_chunk_callback
    if status:
        print(status, flush=True)

    if process_audio_chunk_callback:
        try:
            processed_data = process_audio_chunk_callback(indata.copy()) # Pass a copy
            if processed_data is not None and processed_data.shape == outdata.shape:
                outdata[:] = processed_data
            else:
                # print(f"Processed data is None or shape mismatch: {processed_data.shape if processed_data is not None else 'None'} vs {outdata.shape}")
                outdata.fill(0)  # Fill with silence if no data or shape mismatch
        except Exception as e:
            print(f"Error in process_audio_chunk_callback: {e}")
            import traceback
            traceback.print_exc()
            outdata.fill(0) # Fill with silence on error
    else:
        outdata.fill(0) # Fill with silence if no callback is set

# This will be set by the main application logic
process_audio_chunk_callback = None

def set_process_audio_chunk_callback(callback_function):
    """Sets the callback function for processing audio chunks."""
    global process_audio_chunk_callback
    process_audio_chunk_callback = callback_function

def start_audio_stream(input_device_id, output_device_id, sample_rate, chunk_size):
    """
    Starts the audio stream for real-time input and output.
    The actual audio processing is done in the `process_audio_chunk_callback`.
    """
    global _stream, _is_streaming

    if _is_streaming:
        print("Stream is already running.")
        return False

    try:
        print(f"Attempting to start stream with Input ID: {input_device_id}, Output ID: {output_device_id}, SR: {sample_rate}, Chunk: {chunk_size}")

        input_device_info = get_device_info(input_device_id)
        output_device_info = get_device_info(output_device_id)

        if not input_device_info:
            print(f"Invalid input device ID: {input_device_id}")
            return False
        if not output_device_info:
            print(f"Invalid output device ID: {output_device_id}")
            return False

        input_channels = 1 # For now, let's assume mono input. Can be made configurable.
        # input_channels = input_device_info['max_input_channels']
        output_channels = 1 # For now, let's assume mono output. Can be made configurable.
        # output_channels = output_device_info['max_output_channels']

        print(f"Input device: {input_device_info['name']}, Max input channels: {input_device_info['max_input_channels']}")
        print(f"Output device: {output_device_info['name']}, Max output channels: {output_device_info['max_output_channels']}")
        print(f"Using {input_channels} input channels and {output_channels} output channels.")

        _stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            device=(input_device_id, output_device_id),
            channels=(input_channels, output_channels), # (input_channels, output_channels)
            dtype='float32', # RVC pipeline expects float32
            callback=audio_callback,
            latency='low' # Request low latency
        )
        _stream.start()
        _is_streaming = True
        print("Audio stream started successfully.")
        return True
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        import traceback
        traceback.print_exc()
        _is_streaming = False
        return False

def stop_audio_stream():
    """Stops the currently active audio stream."""
    global _stream, _is_streaming
    if _stream is not None and _is_streaming:
        try:
            _stream.stop()
            _stream.close()
            _is_streaming = False
            print("Audio stream stopped.")
        except Exception as e:
            print(f"Error stopping audio stream: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Stream is not running or already stopped.")
    _stream = None # Ensure it's reset

def is_streaming():
    """Checks if the audio stream is currently active."""
    return _is_streaming

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    print("Available audio devices:")
    inputs, outputs = list_audio_devices()
    print("\nInput Devices:")
    for device in inputs:
        print(device['label'])
    print("\nOutput Devices:")
    for device in outputs:
        print(device['label'])

    if not inputs or not outputs:
        print("\nNot enough input or output devices found to run test.")
    else:
        # Dummy processing callback for testing
        def my_dummy_processor(indata):
            # Simple passthrough for testing
            print(f"Processing chunk of shape: {indata.shape}, dtype: {indata.dtype}, min: {np.min(indata)}, max: {np.max(indata)}")
            # Make sure output shape matches outdata in the stream
            # If stream is (1,1) channels, and indata is (chunk_size, 1)
            # then processed_data should also be (chunk_size, 1)
            return indata

        set_process_audio_chunk_callback(my_dummy_processor)

        # Use default devices or let user choose
        # For testing, let's try to pick the first available ones if they exist
        # Or better, let sounddevice pick default if IDs are not specified, but we need IDs for the dropdowns.

        # For this test, you might need to manually set valid device IDs.
        # For example, input_dev_id = inputs[0]['id'] output_dev_id = outputs[0]['id']
        # Or, find your virtual cable and microphone IDs from the printed list.

        # --- IMPORTANT: Manually set these for your system for this test ---
        # Example:
        # input_dev_id = 1 # Replace with your microphone ID
        # output_dev_id = 5 # Replace with your virtual cable ID
        # sample_rate_test = 48000 # RVC typically uses 16000 for Hubert, but output SR can vary.
        # chunk_size_test = 512  # Common chunk size

        # if start_audio_stream(input_dev_id, output_dev_id, sample_rate_test, chunk_size_test):
        #     print(f"Stream started with Input ID {input_dev_id} and Output ID {output_dev_id}. Press Enter to stop...")
        #     input() # Keep stream running until Enter is pressed
        #     stop_audio_stream()
        # else:
        #     print("Failed to start stream.")
        print("\nSkipping direct stream test in this script. Test through the main application UI.")

print("realtime_audio_utils.py loaded")
