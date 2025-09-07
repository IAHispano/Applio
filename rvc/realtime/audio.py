import os
import sys
import librosa
import traceback
import numpy as np
import sounddevice as sd
from queue import Queue
from dataclasses import dataclass

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE


@dataclass
class ServerAudioDevice:
    index: int = 0
    name: str = ""
    host_api: str = ""
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: int = 0


def list_audio_device():
    """
    Function to query audio devices and host api.
    """
    try:
        audio_device_list = sd.query_devices()
    except Exception as e:
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []

    input_audio_device_list = [
        d for d in audio_device_list if d["max_input_channels"] > 0
    ]
    output_audio_device_list = [
        d for d in audio_device_list if d["max_output_channels"] > 0
    ]

    try:
        hostapis = sd.query_hostapis()
    except Exception as e:
        print("An error occurred while querying the host api:", e)
        hostapis = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the host api:", e)
        hostapis = []

    audio_input_device = []
    audio_output_device = []

    for d in input_audio_device_list:
        input_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_input_device.append(input_audio_device)

    for d in output_audio_device_list:
        output_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_output_device.append(output_audio_device)

    return audio_input_device, audio_output_device


class Audio:
    def __init__(
        self,
        callbacks,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch=False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
    ):
        self.callbacks = callbacks
        self.mon_queue = Queue()
        self.stream = None
        self.monitor = None
        self.running = False
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold

    def get_input_audio_device(self, index: int):
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def get_output_audio_device(self, index: int):
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def process_data(self, indata: np.ndarray):
        indata = indata * self.input_audio_gain
        unpacked_data = librosa.to_mono(indata.T)

        return self.callbacks.change_voice(
            unpacked_data,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def process_data_with_time(self, indata: np.ndarray):
        out_wav, _, perf, _ = self.process_data(indata)
        performance_ms = perf[1]
        # print(f"real-time voice conversion performance: {performance_ms:.2f} ms")
        self.latency = performance_ms  # latency to display on the application interface

        return out_wav

    def audio_stream_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        try:
            out_wav = self.process_data_with_time(indata)

            output_channels = outdata.shape[1]
            if self.use_monitor:
                self.mon_queue.put(out_wav)

            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_queue(self, outdata: np.ndarray, frames, times, status):
        try:
            mon_wav = self.mon_queue.get()

            while self.mon_queue.qsize() > 0:
                self.mon_queue.get()

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(mon_wav, output_channels).reshape(-1, output_channels)
                * self.monitor_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio queue: {error}")
            print(traceback.format_exc())

    def run_audio_stream(
        self,
        block_frame: int,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int,
        input_max_channel: int,
        output_max_channel: int,
        output_monitor_max_channel: int,
        input_extra_setting,
        output_extra_setting,
        output_monitor_extra_setting,
    ):
        self.stream = sd.Stream(
            callback=self.audio_stream_callback,
            latency="low",
            dtype=np.float32,
            device=(input_device_id, output_device_id),
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=(input_max_channel, output_max_channel),
            extra_settings=(input_extra_setting, output_extra_setting),
        )
        self.stream.start()

        if self.use_monitor:
            self.monitor = sd.OutputStream(
                callback=self.audio_queue,
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=AUDIO_SAMPLE_RATE,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting,
            )
            self.monitor.start()

    def stop(self):
        self.running = False

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None

    def start(
        self,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int = None,
        exclusive_mode: bool = False,
        asio_input_channel: int = -1,
        asio_output_channel: int = -1,
        asio_output_monitor_channel: int = -1,
        read_chunk_size: int = 192,
    ):
        self.stop()

        sd._terminate()
        sd._initialize()

        input_audio_device, output_audio_device = self.get_input_audio_device(
            input_device_id
        ), self.get_output_audio_device(output_device_id)
        input_channels, output_channels = (
            input_audio_device.max_input_channels,
            output_audio_device.max_output_channels,
        )

        (
            input_extra_setting,
            output_extra_setting,
            output_monitor_extra_setting,
            monitor_channels,
        ) = (None, None, None, None)
        wasapi_exclusive_mode = bool(exclusive_mode)

        if input_audio_device and "WASAPI" in input_audio_device.host_api:
            input_extra_setting = sd.WasapiSettings(
                exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode
            )
        elif (
            input_audio_device
            and "ASIO" in input_audio_device.host_api
            and asio_input_channel != -1
        ):
            input_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_input_channel]
            )
            input_channels = 1

        if output_audio_device and "WASAPI" in output_audio_device.host_api:
            output_extra_setting = sd.WasapiSettings(
                exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode
            )
        elif (
            input_audio_device
            and "ASIO" in input_audio_device.host_api
            and asio_output_channel != -1
        ):
            output_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_output_channel]
            )
            output_channels = 1

        if self.use_monitor:
            output_monitor_device = self.get_output_audio_device(output_monitor_id)
            monitor_channels = output_monitor_device.max_output_channels

            if output_monitor_device and "WASAPI" in output_monitor_device.host_api:
                output_monitor_extra_setting = sd.WasapiSettings(
                    exclusive=wasapi_exclusive_mode,
                    auto_convert=not wasapi_exclusive_mode,
                )
            elif (
                output_monitor_device
                and "ASIO" in output_monitor_device.host_api
                and asio_output_monitor_channel != -1
            ):
                output_monitor_extra_setting = sd.AsioSettings(
                    channel_selectors=[asio_output_monitor_channel]
                )
                monitor_channels = 1

        block_frame = int((read_chunk_size * 128 / 48000) * AUDIO_SAMPLE_RATE)

        try:
            self.run_audio_stream(
                block_frame,
                input_device_id,
                output_device_id,
                output_monitor_id,
                input_channels,
                output_channels,
                monitor_channels,
                input_extra_setting,
                output_extra_setting,
                output_monitor_extra_setting,
            )
            self.running = True
        except Exception as error:
            print(f"An error occurred while streaming audio: {error}")
            print(traceback.format_exc())
