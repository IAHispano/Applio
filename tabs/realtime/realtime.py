import gradio as gr
import numpy as np
import librosa
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.realtime_audio_utils import (
    list_audio_devices,
    start_audio_stream,
    stop_audio_stream,
    set_process_audio_chunk_callback,
    is_streaming as is_audio_stream_running,
    get_device_info,
)
from rvc.lib.vad_utils import VADProcessor
from rvc.infer.realtime_infer import RealtimeVoiceConverter
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Global state for the Realtime tab
rt_converter = RealtimeVoiceConverter()
vad_processor = None
is_realtime_processing_active = False # Our top-level state

realtime_settings = {
    "pitch": 0,
    "index_rate": 0.75,
    "protect_val": 0.33,
    "f0_method": "rmvpe",
    "vad_enabled": True,
    "vad_sensitivity": 3, # Default, will be updated from UI
    "embedder_name": "contentvec",
    "f0_autotune_enabled": False,
    "f0_autotune_strength": 0.8,
    "output_volume_envelope_mix": 1.0,
}

# Default stream parameters (can be made configurable later)
REALTIME_STREAM_SAMPLE_RATE = 48000  # For mic input and speaker output
REALTIME_CHUNK_SIZE = 1024           # Samples per callback from sounddevice
HUBERT_SAMPLE_RATE = 16000           # RVC and VAD expect this

# Buffers for resampling
resample_input_buffer = np.array([], dtype=np.float32)
resample_output_buffer = np.array([], dtype=np.float32)

# Store last parameters to avoid reloading models unnecessarily
last_loaded_model_path = ""
last_loaded_index_path = ""
last_loaded_sid = -1
last_loaded_embedder = ""


def refresh_audio_devices():
    input_devices, output_devices = list_audio_devices()
    input_dev_labels = [d['label'] for d in input_devices]
    output_dev_labels = [d['label'] for d in output_devices]
    return gr.update(choices=input_dev_labels), gr.update(choices=output_dev_labels)

def get_device_id_from_label(label, device_list):
    if label is None: return None
    for device in device_list:
        if device['label'] == label:
            return device['id']
    return None

def update_status(message, is_error=False):
    prefix = "Error: " if is_error else "Status: "
    return prefix + message

# The core audio processing callback for sounddevice
def audio_processing_callback(mic_chunk_float32):
    global vad_processor, rt_converter, resample_input_buffer, resample_output_buffer, realtime_settings

    if not rt_converter.is_ready() or not is_realtime_processing_active:
        # Return silence if not ready or stopped
        return np.zeros_like(mic_chunk_float32, dtype=np.float32)

    # 1. Resample input chunk from stream SR to Hubert SR (16kHz)
    # This simple resampling might introduce some latency. For lower latency,
    # process in blocks that are multiples of resampling ratio.
    mic_chunk_16khz = librosa.resample(
        mic_chunk_float32,
        orig_sr=REALTIME_STREAM_SAMPLE_RATE,
        target_sr=HUBERT_SAMPLE_RATE
    )

    # 2. VAD
    vad_active = realtime_settings["vad_enabled"]
    processed_16khz_chunk = None # This will store the audio data post-VAD, pre-RVC (or post-RVC)

    if vad_active and vad_processor:
        try:
            is_speech = vad_processor.is_speech(mic_chunk_16khz.copy()) # VAD might modify array if not copied
            if not is_speech:
                # If no speech, output silence (at Hubert SR for now, will be resampled)
                processed_16khz_chunk = np.zeros_like(mic_chunk_16khz, dtype=np.float32)
            # If speech, RVC processing will happen next (processed_16khz_chunk remains None)
        except Exception as e:
            print(f"VAD error: {e}")
            # Fallback to processing if VAD errors, processed_16khz_chunk remains None
            pass

    # 3. RVC processing (if speech, or VAD inactive/errored)
    if processed_16khz_chunk is None: # Indicates speech or VAD not silencing
        try:
            # Use parameters from realtime_settings
            processed_16khz_chunk = rt_converter.process_chunk( # This output is at rt_converter.tgt_sr
                mic_chunk_16khz,
                pitch_change=realtime_settings["pitch"],
                f0_method=realtime_settings["f0_method"],
                index_rate=realtime_settings["index_rate"],
                protect_val=realtime_settings["protect_val"],
                f0_autotune=realtime_settings["f0_autotune_enabled"],
                f0_autotune_strength=realtime_settings["f0_autotune_strength"],
                output_volume_envelope_mix=realtime_settings["output_volume_envelope_mix"]
            )
        except Exception as e:
            print(f"RVC error in process_chunk: {e}")
            import traceback
            traceback.print_exc()
            processed_16khz_chunk = np.zeros_like(mic_chunk_16khz, dtype=np.float32) # Silence on error

    # At this point, processed_16khz_chunk is at Hubert SR (16kHz) but needs to be converted to RVC's tgt_sr
    # However, rt_converter.process_chunk already outputs at rt_converter.tgt_sr

    # 4. Resample output chunk from RVC tgt_sr back to stream SR
    if rt_converter.tgt_sr is None: # Should not happen if model is loaded
        print("Error: rt_converter.tgt_sr is None. Cannot resample output.")
        return np.zeros_like(mic_chunk_float32, dtype=np.float32)

    output_chunk_stream_sr = librosa.resample(
        processed_16khz_chunk, # This is actually at tgt_sr from RVC
        orig_sr=rt_converter.tgt_sr,
        target_sr=REALTIME_STREAM_SAMPLE_RATE
    )

    # Ensure output shape matches input shape (sounddevice expects this)
    if output_chunk_stream_sr.shape != mic_chunk_float_32.shape:
        # Basic padding/truncating to match length.
        # This can happen due to slight variations in resampling.
        target_len = mic_chunk_float32.shape[0]
        current_len = output_chunk_stream_sr.shape[0]
        if current_len < target_len:
            output_chunk_stream_sr = np.pad(output_chunk_stream_sr, (0, target_len - current_len), mode='constant')
        elif current_len > target_len:
            output_chunk_stream_sr = output_chunk_stream_sr[:target_len]

        if mic_chunk_float32.ndim > 1: # If input was stereo (e.g. (1024,2))
            if output_chunk_stream_sr.ndim == 1: # and output is mono
                 # Duplicate mono to stereo to match outdata shape
                output_chunk_stream_sr = np.tile(output_chunk_stream_sr[:, np.newaxis], (1, mic_chunk_float32.shape[1]))


    return output_chunk_stream_sr.astype(np.float32)


def start_realtime_conversion(
    input_dev_label, output_dev_label,
    model_path, index_path, sid_value,
    f0_method_val, pitch_val, index_rate_val, protect_val,
    f0_autotune_enabled_val, f0_autotune_strength_val, output_volume_envelope_mix_val, # New options
    vad_enabled_val, vad_sensitivity_val, embedder_name_val
):
    global rt_converter, vad_processor, is_realtime_processing_active, realtime_settings
    global last_loaded_model_path, last_loaded_index_path, last_loaded_sid, last_loaded_embedder

    if is_realtime_processing_active or is_audio_stream_running():
        return update_status("Processing is already active. Please stop it first.", is_error=True)

    input_devices, output_devices = list_audio_devices()
    input_device_id = get_device_id_from_label(input_dev_label, input_devices)
    output_device_id = get_device_id_from_label(output_dev_label, output_devices)

    if input_device_id is None: return update_status("Invalid input device selected.", is_error=True)
    if output_device_id is None: return update_status("Invalid output device selected.", is_error=True)
    if not model_path: return update_status("RVC Model path is required.", is_error=True)

    try:
        sid_int = int(sid_value)
    except ValueError:
        return update_status("Speaker ID must be an integer.", is_error=True)

    # Load/reload models only if parameters changed
    if (model_path != last_loaded_model_path or \
        (index_path if index_path else "") != (last_loaded_index_path if last_loaded_index_path else "") or \
        sid_int != last_loaded_sid or \
        embedder_name_val != last_loaded_embedder):

        status_msg = update_status("Loading models...")
        print(status_msg) # also print to console
        # TODO: Expose embedder_custom_path if embedder_name_val is "custom"
        loaded_ok, message = rt_converter.load_resources(model_path, index_path, sid_int, embedder_name_val)
        if not loaded_ok:
            return update_status(f"Failed to load models: {message}", is_error=True)

        last_loaded_model_path = model_path
        last_loaded_index_path = index_path if index_path else ""
        last_loaded_sid = sid_int
        last_loaded_embedder = embedder_name_val
        status_msg = update_status("Models loaded.")
        print(status_msg)
    else:
        status_msg = update_status("Models already loaded.")
        print(status_msg)

    if not rt_converter.is_ready():
        return update_status("Converter not ready after attempting to load models.", is_error=True)

    # Store settings for the audio callback to use
    realtime_settings["pitch"] = pitch_val
    realtime_settings["index_rate"] = index_rate_val
    realtime_settings["protect_val"] = protect_val
    realtime_settings["f0_method"] = f0_method_val
    realtime_settings["vad_enabled"] = vad_enabled_val
    realtime_settings["vad_sensitivity"] = int(vad_sensitivity_val)
    realtime_settings["embedder_name"] = embedder_name_val
    realtime_settings["f0_autotune_enabled"] = f0_autotune_enabled_val
    realtime_settings["f0_autotune_strength"] = f0_autotune_strength_val
    realtime_settings["output_volume_envelope_mix"] = output_volume_envelope_mix_val

    if realtime_settings["vad_enabled"]:
        try:
            vad_processor = VADProcessor(sensitivity_mode=realtime_settings["vad_sensitivity"], sample_rate=HUBERT_SAMPLE_RATE)
            print(f"VAD Processor initialized with sensitivity {realtime_settings['vad_sensitivity']} for 16kHz audio.")
        except Exception as e:
            return update_status(f"Failed to initialize VAD: {str(e)}", is_error=True)
    else:
        vad_processor = None
        print("VAD is disabled.")

    # Set the main processing callback for sounddevice
    set_process_audio_chunk_callback(audio_processing_callback)

    if start_audio_stream(input_device_id, output_device_id, REALTIME_STREAM_SAMPLE_RATE, REALTIME_CHUNK_SIZE):
        is_realtime_processing_active = True
        return update_status("Realtime processing started.")
    else:
        is_realtime_processing_active = False
        return update_status("Failed to start audio stream.", is_error=True)


def stop_realtime_conversion():
    global is_realtime_processing_active
    if is_audio_stream_running():
        stop_audio_stream()
    is_realtime_processing_active = False
    # `set_process_audio_chunk_callback(None)` will be called by stop_audio_stream if we add it there, or do it explicitly.
    # For now, audio_processing_callback checks `is_realtime_processing_active`.
    return update_status("Realtime processing stopped.")


# Gradio UI layout
def realtime_tab():
    input_devices, output_devices = list_audio_devices()
    input_device_labels = [d['label'] for d in input_devices]
    output_device_labels = [d['label'] for d in output_devices]

    # Get model choices from inference tab or define here
    # This is a simplified version; ideally, share with inference_tab.py
    model_root_relative = os.path.join(now_dir, "logs")
    model_choices = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))
    ]
    index_choices = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    with gr.Column() as layout:
        status_label = gr.Label("Status: Idle")

        with gr.Row():
            input_device_dropdown = gr.Dropdown(choices=input_device_labels, label=i18n("Input Microphone"))
            output_device_dropdown = gr.Dropdown(choices=output_device_labels, label=i18n("Output Device (Virtual Cable)"))
            refresh_devices_button = gr.Button(i18n("Refresh Devices"))

        refresh_devices_button.click(
            refresh_audio_devices,
            outputs=[input_device_dropdown, output_device_dropdown]
        )

        with gr.Accordion(i18n("RVC Model Settings"), open=True):
            model_path_dropdown = gr.Dropdown(choices=model_choices, label=i18n("RVC Model (.pth)"), allow_custom_value=True)
            index_path_dropdown = gr.Dropdown(choices=index_choices, label=i18n("Index File (.index)"), allow_custom_value=True)
            # TODO: Add embedder model dropdown (contentvec, custom, etc.)
            embedder_model_name = gr.Radio( # Placeholder, use same as inference tab
                label=i18n("Embedder Model"),
                choices=["contentvec", "chinese-hubert-base", "custom"], # Simplified
                value="contentvec",
                interactive=True
            )
            speaker_id_slider = gr.Slider(minimum=0, maximum=100, step=1, label=i18n("Speaker ID (SID)"), value=0) # Max needs to be dynamic

        with gr.Accordion(i18n("Realtime Conversion Settings"), open=True):
            f0_method_dropdown = gr.Dropdown(
                choices=["rmvpe", "fcpe"], # Simplified for real-time
                value="rmvpe",
                label=i18n("F0 Method")
            )
            pitch_slider = gr.Slider(minimum=-24, maximum=24, step=1, label=i18n("Pitch (semitones)"), value=0)
            index_rate_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label=i18n("Index Rate"), value=0.75)
            protect_slider = gr.Slider(minimum=0.0, maximum=0.5, step=0.01, label=i18n("Protect Voiceless"), value=0.33)
            volume_mix_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label=i18n("Output Volume Mix (Envelope)"), value=1.0, info=i18n("0:Input envelope, 1:Output envelope"))

            with gr.Row():
                f0_autotune_checkbox = gr.Checkbox(label=i18n("F0 Autotune"), value=False)
                f0_autotune_strength_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label=i18n("Autotune Strength"), value=0.8, interactive=True, visible=False) # Initially hidden

            f0_autotune_checkbox.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[f0_autotune_checkbox],
                outputs=[f0_autotune_strength_slider],
            )

        with gr.Accordion(i18n("VAD Settings"), open=False):
            vad_enabled_checkbox = gr.Checkbox(label=i18n("Enable VAD"), value=True)
            vad_sensitivity_slider = gr.Slider(minimum=0, maximum=3, step=1, label=i18n("VAD Sensitivity (0-3, 3=Aggressive)"), value=3)

        with gr.Row():
            start_button = gr.Button(i18n("Start Realtime"))
            stop_button = gr.Button(i18n("Stop Realtime"))

        start_button.click(
            start_realtime_conversion,
            inputs=[
                input_device_dropdown, output_device_dropdown,
                model_path_dropdown, index_path_dropdown, speaker_id_slider,
                f0_method_dropdown, pitch_slider, index_rate_slider, protect_slider,
                f0_autotune_checkbox, f0_autotune_strength_slider, volume_mix_slider, # New inputs
                vad_enabled_checkbox, vad_sensitivity_slider, embedder_model_name
            ],
            outputs=[status_label]
        )
        stop_button.click(
            stop_realtime_conversion,
            outputs=[status_label]
        )
    return layout

if __name__ == '__main__':
    # For testing this tab in isolation
    app = gr.Blocks()
    with app:
        gr.Markdown("# Realtime RVC Test Interface")
        realtime_tab()
    app.launch(debug=True)

print("realtime.py loaded")
