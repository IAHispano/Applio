import gradio as gr
import os
import sys
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

from .callbacks import AudioCallbacks
from .audio import list_audio_device
from .core import AUDIO_SAMPLE_RATE

from tabs.inference.inference import (
    i18n,
    names,
    default_weight,
    get_indexes,
    extract_model_and_epoch,
    match_index,
    get_speakers_id,
    create_folder_and_move_files,
    refresh_embedders_folders,
    custom_embedder_root_relative,
    model_root_relative
)

PASS_THROUGH = False
interactive_true = gr.update(interactive=True)
interactive_false = gr.update(interactive=False)
running, callbacks, audio_manager = False, None, None

def start_realtime(
    input_audio_device: str,
    input_audio_gain: int,
    input_asio_channels: int,
    output_audio_device: str,
    output_audio_gain: int,
    output_asio_channels: int,
    monitor_output_device: str,
    monitor_audio_gain: int,
    monitor_asio_channels: int,
    use_monitor_device: bool,
    exclusive_mode: bool,
    vad_enabled: bool,
    chunk_size: float,
    cross_fade_overlap_size: float,
    extra_convert_size: float,
    silent_threshold: int,
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    pth_path: str,
    index_path: str,
    sid: int,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    embedder_model: str,
    embedder_model_custom: str = None,
):
    global running, callbacks, audio_manager
    running = True

    if not input_audio_device or not output_audio_device:
        return "Please select valid input/output devices!", interactive_true, interactive_false
    if use_monitor_device and not monitor_output_device:
        return "Please select a valid monitor device!", interactive_true, interactive_false
    if not pth_path:
        return "Model path not provided. Aborting conversion.", interactive_true, interactive_false

    yield "Starting Realtime...", interactive_false, interactive_true
    
    read_chunk_size = int(chunk_size * AUDIO_SAMPLE_RATE / 1000 / 128)

    sid = int(sid) if sid is not None else 0

    input_audio_gain /= 100.0
    output_audio_gain /= 100.0
    monitor_audio_gain /= 100.0

    try:
        input_device_id = int(input_audio_device.split(":")[0])
        output_device_id = int(output_audio_device.split(":")[0])
        output_monitor_id = int(monitor_output_device.split(":")[0]) if use_monitor_device else None
    except (ValueError, IndexError):
        yield "Incorrectly formatted audio device. Stopping.", interactive_true, interactive_false
        return

    callbacks = AudioCallbacks(
        pass_through=PASS_THROUGH,
        read_chunk_size=read_chunk_size,
        cross_fade_overlap_size=cross_fade_overlap_size,
        extra_convert_size=extra_convert_size,
        model_path=pth_path,
        index_path=index_path,
        f0_method=f0_method,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        silent_threshold=silent_threshold,
        f0_up_key=pitch,
        index_rate=index_rate,
        protect=protect,
        volume_envelope=volume_envelope,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        input_audio_gain=input_audio_gain,
        output_audio_gain=output_audio_gain,
        monitor_audio_gain=monitor_audio_gain,
        monitor=use_monitor_device,
        vad_enabled=vad_enabled,
        vad_sensitivity=3,
        vad_frame_ms=30,
        sid=sid,
    )

    audio_manager = callbacks.audio
    audio_manager.start(
        input_device_id=input_device_id,
        output_device_id=output_device_id,
        output_monitor_id=output_monitor_id,
        exclusive_mode=exclusive_mode,
        asio_input_channel=input_asio_channels,
        asio_output_channel=output_asio_channels,
        asio_output_monitor_channel=monitor_asio_channels,
        read_chunk_size=read_chunk_size,
    )

    yield "Realtime is ready!", interactive_false, interactive_true

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)
        if hasattr(audio_manager, "latency"):
            yield f"Latency: {audio_manager.latency:.2f} ms", interactive_false, interactive_true

    return gr.update(), gr.update(), gr.update()

def stop_realtime():
    global running, callbacks, audio_manager
    if running and audio_manager is not None and callbacks is not None:
        audio_manager.stop()
        running = False
        if hasattr(audio_manager, "latency"): del audio_manager.latency
        audio_manager = callbacks = None

        return gr.update(value="Stopping..."), gr.update(), gr.update()
    else:
        return "Realtime pipeline not found!", interactive_true, interactive_false

def get_audio_devices_formatted():
    try:
        input_devices, output_devices = list_audio_device()

        def priority(name: str) -> int:
            n = name.lower()
            if 'virtual' in n:
                return 0 
            if 'vb' in n:
                return 1
            return 2

        output_sorted = sorted(output_devices, key=lambda d: priority(d.name))
        input_sorted = sorted(input_devices, key=lambda d: priority(d.name), reverse=True)

        input_device_list = [f"{d.index}: {d.name} ({d.host_api})" for d in input_sorted]
        output_device_list = [f"{d.index}: {d.name} ({d.host_api})" for d in output_sorted]

        return input_device_list, output_device_list
    except Exception:
        return [], []

def realtime_tab():
    gr.Markdown("## Realtime Voice Changer")
    input_devices, output_devices = get_audio_devices_formatted()

    with gr.Blocks() as ui:
        with gr.Row():
            start_button = gr.Button("Start", variant="primary")
            stop_button = gr.Button("Stop", interactive=False)
        latency_info = gr.Label(label=i18n("Status"), value="Realtime not started.")

        with gr.Tabs():
            with gr.TabItem("Audio Settings"):
                with gr.Row():
                    refresh_devices_button = gr.Button("Refresh Audio Devices")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Device")
                        input_audio_device = gr.Dropdown(label="Input Device", info="Select the microphone or audio interface you will be speaking into.", choices=input_devices, interactive=True)
                        input_audio_gain = gr.Slider(minimum=0, maximum=200, value=100, label="Input Gain (%)", info="Adjusts the input volume before processing. Prevents clipping or boosts a quiet mic.", interactive=True)
                        input_asio_channels = gr.Slider(minimum=-1, maximum=16, value=-1, step=1, label="Input ASIO Channel", info="For ASIO drivers, selects a specific input channel. Leave at -1 for default.", interactive=True)
                    with gr.Column():
                        gr.Markdown("### Output Device")
                        output_audio_device = gr.Dropdown(label="Output Device", info="Select the device where the final converted voice will be sent (e.g., a virtual cable).", choices=output_devices, interactive=True)
                        output_audio_gain = gr.Slider(minimum=0, maximum=200, value=100, label="Output Gain (%)", info="Adjusts the final volume of the converted voice after processing.", interactive=True)
                        output_asio_channels = gr.Slider(minimum=-1, maximum=16, value=-1, step=1, label="Output ASIO Channel", info="For ASIO drivers, selects a specific output channel. Leave at -1 for default.", interactive=True)
                with gr.Accordion("Monitor Device (Optional)", open=False):
                    with gr.Column():
                        use_monitor_device = gr.Checkbox(label="Use Monitor Device", value=False, interactive=True)
                        monitor_output_device = gr.Dropdown(label="Monitor Device", info="Select the device for monitoring your voice (e.g., your headphones).", choices=output_devices, interactive=True)
                        monitor_audio_gain = gr.Slider(minimum=0, maximum=200, value=100, label="Monitor Gain (%)", info="Adjusts the volume of the monitor feed, independent of the main output.", interactive=True)
                        monitor_asio_channels = gr.Slider(minimum=-1, maximum=16, value=-1, step=1, label="Monitor ASIO Channel", info="For ASIO drivers, selects a specific monitor output channel. Leave at -1 for default.", interactive=True)
                with gr.Row():
                    exclusive_mode = gr.Checkbox(label="Exclusive Mode (WASAPI)", info="For WASAPI (Windows), gives the app exclusive control for potentially lower latency.", value=True, interactive=True)
                    vad_enabled = gr.Checkbox(label="Enable VAD", info="Enables Voice Activity Detection to only process audio when you are speaking, saving CPU.", value=True, interactive=True)

            with gr.TabItem("Model Settings"):
                with gr.Row():
                    model_file = gr.Dropdown(label=i18n("Voice Model"), choices=sorted(names, key=extract_model_and_epoch) if names else [], interactive=True, value=default_weight)
                    index_file = gr.Dropdown(label=i18n("Index File"), choices=get_indexes(), value=match_index(default_weight) if default_weight else "", interactive=True)
                    
                with gr.Row():
                    unload_button = gr.Button(i18n("Unload Voice"))
                    refresh_button = gr.Button(i18n("Refresh"))
                with gr.Column():
                    autotune = gr.Checkbox(label=i18n("Autotune"), info=i18n("Apply a soft autotune to your inferences, recommended for singing conversions."), visible=True, value=False, interactive=True)
                    autotune_strength = gr.Slider(minimum=0, maximum=1, label=i18n("Autotune Strength"), info=i18n("Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."), visible=False, value=1, interactive=True)
                    proposed_pitch = gr.Checkbox(label=i18n("Proposed Pitch"), info=i18n("Adjust the input audio pitch to match the voice model range."), visible=True, value=False, interactive=True)
                    proposed_pitch_threshold = gr.Slider(minimum=50.0, maximum=1200.0, label=i18n("Proposed Pitch Threshold"), info=i18n("Male voice models typically use 155.0 and female voice models typically use 255.0."), visible=False, value=155.0, interactive=True)
                    sid = gr.Dropdown(label=i18n("Speaker ID"), choices=get_speakers_id(default_weight) if default_weight else [0], value=0, interactive=True)
                    pitch = gr.Slider(minimum=-24, maximum=24, step=1, label=i18n("Pitch"), info=i18n("Set the pitch of the audio, the higher the value, the higher the pitch."), value=0, interactive=True)
                    index_rate = gr.Slider(minimum=0, maximum=1, label=i18n("Search Feature Ratio"), info=i18n("Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."), value=0.75, interactive=True)
                    volume_envelope = gr.Slider(minimum=0, maximum=1, value=1, label=i18n("Volume Envelope"), info=i18n("Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."), interactive=True)
                    protect = gr.Slider(minimum=0, maximum=0.5, value=0.5, label=i18n("Protect Voiceless Consonants"), info=i18n("Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."), interactive=True)
                    f0_method = gr.Radio(choices=["rmvpe", "fcpe", "swift"], value="swift", label=i18n("Pitch extraction algorithm"), info=i18n("Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."), interactive=True)
                    embedder_model = gr.Radio(choices=["contentvec", "spin", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base", "custom"], value="contentvec", label=i18n("Embedder Model"), info=i18n("Model used for learning speaker embedding."), interactive=True)
                    with gr.Column(visible=False) as embedder_custom:
                        with gr.Accordion(i18n("Custom Embedder"), open=True):
                            with gr.Row():
                                embedder_model_custom = gr.Dropdown(
                                    label=i18n("Select Custom Embedder"),
                                    choices=refresh_embedders_folders(),
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                refresh_embedders_button = gr.Button(
                                    i18n("Refresh embedders")
                                )
                            folder_name_input = gr.Textbox(
                                label=i18n("Folder Name"), interactive=True
                            )
                            with gr.Row():
                                bin_file_upload = gr.File(
                                    label=i18n("Upload .bin"),
                                    type="filepath",
                                    interactive=True,
                                )
                                config_file_upload = gr.File(
                                    label=i18n("Upload .json"),
                                    type="filepath",
                                    interactive=True,
                                )
                            move_files_button = gr.Button(
                                i18n("Move files to custom embedder folder")
                            )

            with gr.TabItem("Performance Settings"):
                chunk_size = gr.Slider(
                    minimum=2.7, 
                    maximum=2730.7, 
                    value=192, 
                    step=1, 
                    label="Chunk Size (ms)", 
                    info="Audio buffer size in milliseconds. Lower values may reduce latency but increase CPU load.",
                    interactive=True
                )
                cross_fade_overlap_size = gr.Slider(minimum=0.05, maximum=0.2, value=0.01, label="Crossfade Overlap Size (s)", info="Duration of the fade between audio chunks to prevent clicks. Higher values create smoother transitions but may increase latency.", interactive=True)
                extra_convert_size = gr.Slider(minimum=0.1, maximum=5, value=0.5, step=0.1, label="Extra Conversion Size (s)", info="Amount of extra audio processed to provide context to the model. Improves conversion quality at the cost of higher CPU usage.",interactive=True)
                silent_threshold = gr.Slider(minimum=-90, maximum=-60, value=-90, step=1, label="Silence Threshold (dB)", info="Volume level below which audio is treated as silence and not processed. Helps to save CPU resources and reduce background noise.", interactive=True)

        def update_on_model_change(model_path):
            new_index = match_index(model_path)
            new_sids = get_speakers_id(model_path)
            return gr.update(value=new_index), gr.update(choices=new_sids, value=0 if new_sids else None)
        
        def refresh_devices():
            input_choices, output_choices = get_audio_devices_formatted()
            return gr.update(choices=input_choices), gr.update(choices=output_choices), gr.update(choices=output_choices)
            
        def toggle_visible(checkbox):
            return {"visible": checkbox, "__type__": "update"}

        def toggle_visible_embedder_custom(embedder_model):
            if embedder_model == "custom":
                return {"visible": True, "__type__": "update"}
            return {"visible": False, "__type__": "update"}

        refresh_devices_button.click(
            fn=refresh_devices,
            outputs=[input_audio_device, output_audio_device, monitor_output_device]
        )

        autotune.change(
            fn=toggle_visible,
            inputs=[autotune],
            outputs=[autotune_strength],
        )

        proposed_pitch.change(
            fn=toggle_visible,
            inputs=[proposed_pitch],
            outputs=[proposed_pitch_threshold],
        )

        embedder_model.change(
            fn=toggle_visible_embedder_custom,
            inputs=[embedder_model],
            outputs=[embedder_custom],
        )

        move_files_button.click(
            fn=create_folder_and_move_files,
            inputs=[folder_name_input, bin_file_upload, config_file_upload],
            outputs=[],
        )
        refresh_embedders_button.click(
            fn=lambda: gr.update(choices=refresh_embedders_folders()),
            inputs=[],
            outputs=[embedder_model_custom],
        )

        start_button.click(
            fn=start_realtime,
            inputs=[
                input_audio_device, input_audio_gain, input_asio_channels,
                output_audio_device, output_audio_gain, output_asio_channels,
                monitor_output_device, monitor_audio_gain, monitor_asio_channels,
                use_monitor_device, exclusive_mode, vad_enabled,
                chunk_size, cross_fade_overlap_size, extra_convert_size, silent_threshold,
                pitch, index_rate, volume_envelope, protect,
                f0_method, model_file, index_file, sid,
                autotune, autotune_strength,
                proposed_pitch, proposed_pitch_threshold,
                embedder_model, embedder_model_custom,
            ],
            outputs=[latency_info, start_button, stop_button],
        )

        stop_button.click(fn=stop_realtime, outputs=[latency_info, start_button, stop_button]).then(
            fn=lambda: (yield gr.update(value="Stopped"), interactive_true, interactive_false),
            inputs=None,
            outputs=[latency_info, start_button, stop_button]
        )
        unload_button.click(fn=lambda: ({"value": "", "__type__": "update"}, {"value": "", "__type__": "update"}), inputs=[], outputs=[model_file, index_file])
        model_file.select(fn=update_on_model_change, inputs=[model_file], outputs=[index_file, sid])

        def refresh_all():
            new_names = [os.path.join(root, file) for root, _, files in os.walk(model_root_relative, topdown=False) for file in files if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))]
            new_indexes = get_indexes()
            input_choices, output_choices = get_audio_devices_formatted()
            return (gr.update(choices=sorted(new_names, key=extract_model_and_epoch)), gr.update(choices=new_indexes), gr.update(choices=input_choices), gr.update(choices=output_choices), gr.update(choices=output_choices))

        refresh_button.click(fn=refresh_all, outputs=[model_file, index_file, input_audio_device, output_audio_device, monitor_output_device])