"""Audio transports around LiveConverter, not another synthesis architecture.

The Gradio microphone, WebSocket API and native device session feed bounded PCM
packets into the same converter. Rational resampling preserves filter history
across packets; transport buffers/device scheduling add latency beyond model
lookahead. Each session owns its stream and reset/flush lifetime.
"""

import asyncio
import json
import math
import threading
from queue import Empty, Full, Queue

import numpy as np

from rvc.infer.v3 import LiveConverter


class StreamingResampler:
    """Phase-aligned polyphase FIR with a bounded halo and exact flush length."""

    def __init__(self, input_rate, output_rate):
        if input_rate < 1 or output_rate < 1:
            raise ValueError("Sample rates must be positive")
        divisor = math.gcd(input_rate, output_rate)
        self.up, self.down = output_rate // divisor, input_rate // divisor
        self.reset()

    def reset(self):
        self.audio = np.empty(0, dtype=np.float32)
        self.origin = self.received = self.emitted = 0
        self.closed = False

    def push(self, audio, final=False):
        from scipy.signal import resample_poly

        if self.closed:
            raise ValueError("Resampler is flushed")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Resampler requires finite mono PCM")
        self.closed = final
        if self.up == self.down:
            return audio.copy()
        self.audio = np.concatenate([self.audio, audio])
        self.received += len(audio)
        halo = max(2 * self.down, math.ceil(12 * max(self.up, self.down) / self.up))
        end = (
            math.ceil(self.received * self.up / self.down)
            if final
            else max(0, math.floor((self.received - halo) * self.up / self.down))
        )
        if end <= self.emitted:
            return np.empty(0, dtype=np.float32)
        resampled = resample_poly(self.audio, self.up, self.down).astype(np.float32)
        offset = self.origin // self.down * self.up
        output = resampled[self.emitted - offset : end - offset].copy()
        self.emitted = end
        keep = max(
            0, (int(self.emitted * self.down / self.up) - halo) // self.down * self.down
        )
        discard = max(0, keep - self.origin)
        self.audio = self.audio[discard:].copy()
        self.origin += discard
        return output


class BrowserSession:
    """One Gradio recording owns a converter stream and input resampling history.

    Gradio supplies (sample_rate, PCM) chunks. Integer PCM is normalized before
    stereo is mixed to mono; output is float32 PCM at the model's sample rate.
    Keep this object in session-local gr.State and discard it after flushing.
    """

    def __init__(self, converter, sample_rate, speaker=0, semitones=0, steps=0, seed=0):
        self.sample_rate = int(sample_rate)
        if not 8000 <= self.sample_rate <= 192000:
            raise ValueError("Unsupported input sample rate")
        self.output_rate = converter.mel_config.sample_rate
        self.live = LiveConverter(converter, speaker, semitones, steps, seed)
        self.resampler = StreamingResampler(self.sample_rate, self.output_rate)

    def push(self, chunk):
        sample_rate, audio = chunk
        if int(sample_rate) != self.sample_rate:
            raise ValueError("Microphone sample rate changed; start a new recording")
        audio = np.asarray(audio)
        if audio.ndim not in {1, 2} or len(audio) > self.sample_rate * 2:
            raise ValueError(
                "Send mono or stereo microphone chunks of at most two seconds"
            )
        if np.issubdtype(audio.dtype, np.integer):
            limits = np.iinfo(audio.dtype)
            midpoint = (limits.max + limits.min + 1) / 2
            audio = (audio.astype(np.float32) - midpoint) / (
                (limits.max - limits.min + 1) / 2
            )
        else:
            audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        output = self.live.push(self.resampler.push(audio))
        return (self.output_rate, output) if len(output) else None

    def flush(self):
        tail = self.live.push(self.resampler.push(np.empty(0, np.float32), final=True))
        output = np.concatenate([tail, self.live.flush()])
        return (self.output_rate, output) if len(output) else None


def create_app(converter):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app = FastAPI(title="Applio v3 live")
    gpu_lock = asyncio.Lock()

    @app.get("/")
    async def info():
        return {
            "backend": "applio-v3",
            "stream": "/stream",
            "interface": "Applio Gradio Realtime tab",
        }

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            settings = json.loads(await ws.receive_text())
            input_rate = int(settings.get("sample_rate", 48000))
            if not 8000 <= input_rate <= 192000:
                raise ValueError("Unsupported input sample rate")
            session = LiveConverter(
                converter,
                speaker=int(settings.get("speaker", 0)),
                semitones=float(settings.get("semitones", 0)),
                steps=int(settings.get("steps", 4)),
                seed=int(settings.get("seed", 0)),
            )
            resampler = StreamingResampler(input_rate, converter.mel_config.sample_rate)
            await ws.send_json(
                {
                    "sample_rate": converter.mel_config.sample_rate,
                    "latency": session.latency,
                    "encoding": "float32-le",
                    "speakers": converter.acoustic_package["speakers"],
                }
            )
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    if len(data) % 4 or len(data) > input_rate * 4 * 2:
                        raise ValueError(
                            "Send float32 PCM chunks of at most two seconds"
                        )
                    audio = np.frombuffer(data, dtype="<f4")
                    async with gpu_lock:
                        output = await asyncio.to_thread(
                            session.push, resampler.push(audio)
                        )
                    await ws.send_bytes(output.astype("<f4").tobytes())
                else:
                    control = json.loads(message.get("text", "{}"))
                    if control.get("command") == "reset":
                        session.reset()
                        resampler.reset()
                        await ws.send_json({"reset": True})
                    elif control.get("command") == "flush":
                        async with gpu_lock:
                            tail = await asyncio.to_thread(
                                session.push,
                                resampler.push(np.empty(0, np.float32), final=True),
                            )
                            final = await asyncio.to_thread(session.flush)
                        await ws.send_bytes(
                            np.concatenate([tail, final]).astype("<f4").tobytes()
                        )
                        await ws.send_json(
                            {"flushed": True, "samples": session.output_samples}
                        )
                        break
                    else:
                        raise ValueError("Unknown stream command")
        except WebSocketDisconnect:
            pass
        except (ValueError, FloatingPointError) as error:
            await ws.send_json({"error": str(error)})
        finally:
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app


class NativeSession:
    """Audio callbacks only enqueue/dequeue PCM; model work runs in a worker."""

    def __init__(
        self,
        converter,
        input_device=None,
        output_device=None,
        sample_rate=48000,
        block_size=1024,
        speaker=0,
        semitones=0,
        steps=4,
        seed=0,
    ):
        self.live = LiveConverter(converter, speaker, semitones, steps, seed)
        self.in_resample = StreamingResampler(
            sample_rate, converter.mel_config.sample_rate
        )
        self.out_resample = StreamingResampler(
            converter.mel_config.sample_rate, sample_rate
        )
        self.input_device, self.output_device = input_device, output_device
        self.sample_rate, self.block_size = sample_rate, block_size
        self.input_queue = Queue(maxsize=32)
        self.output_queue = Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.error = None
        self.underflows = self.overflows = 0
        self.pending = np.empty(0, np.float32)

    def callback(self, input_data, output_data, frames, timing, status):
        try:
            self.input_queue.put_nowait(input_data[:, 0].copy())
        except Full:
            self.overflows += 1
            self.error = RuntimeError(
                "Live input queue overflow: processing is slower than capture"
            )
            self.stop_event.set()
        while len(self.pending) < frames:
            try:
                self.pending = np.concatenate(
                    [self.pending, self.output_queue.get_nowait()]
                )
            except Empty:
                break
        output_data.fill(0)
        count = min(frames, len(self.pending))
        output_data[:count, 0] = self.pending[:count]
        self.pending = self.pending[count:].copy()
        if count < frames:
            self.underflows += 1

    def worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    audio = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue
                converted = self.live.push(self.in_resample.push(audio))
                output = self.out_resample.push(converted)
                if len(output):
                    self.output_queue.put(output, timeout=1)
        except Exception as error:
            self.error = error
            self.stop_event.set()

    def run(self):
        import sounddevice as sd

        thread = threading.Thread(
            target=self.worker, name="applio-v3-audio", daemon=True
        )
        thread.start()
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                device=(self.input_device, self.output_device),
                callback=self.callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            thread.join(timeout=5)
        if self.error:
            raise self.error
