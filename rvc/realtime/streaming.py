"""Retained V3 state for packet-size-independent bounded conversion.

FeatureStream waits for declared future observations. AcousticStream retains
causal histories independently for every integration evaluation. VocoderStream
retains filter/overlap halos, oscillator phase and absolute noise coordinates.
Finalization emits the withheld tail; reset starts a different recording.
Throughput and these algorithmic buffers are different latency measurements.
"""

import math

import numpy as np
import torch


def coordinate_noise(frames, channels=1, offset=0, seed=0):
    """Generate deterministic Gaussian excitation keyed by absolute position, channel and seed."""
    if frames < 0 or channels < 1 or offset < 0 or seed < 0:
        raise ValueError("Noise coordinates and seed must be nonnegative")
    index = np.arange(offset * channels, (offset + frames) * channels, dtype=np.uint64)

    def uniform(x):
        with np.errstate(over="ignore"):
            z = x + np.uint64(seed & ((1 << 64) - 1)) + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> 31)
        return ((z >> 11).astype(np.float64) + 0.5) / (1 << 53)

    u, v = uniform(index * np.uint64(2)), uniform(index * np.uint64(2) + np.uint64(1))
    return (
        (np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))
        .astype(np.float32)
        .reshape(frames, channels)
    )


def cached_blocks(blocks, x, histories):
    for i, block in enumerate(blocks):
        x, histories[i] = block.stream(x, histories[i])
    return x


class AcousticStream:
    """Cache conditioning/predictor history and separate refiner histories per step.

    Integration evaluations see different states at the same time position, so
    sharing one history across steps would change the learned vector field.
    """

    def __init__(self, model, steps=4, seed=0, ordinary=False):
        if not model.config.causal:
            raise ValueError("Streaming requires a causal acoustic package")
        if steps not in {0, 1, 2, 4, 8, 16, 32}:
            raise ValueError("Unsupported refinement budget")
        if steps and not bool(model.flow_trained):
            raise ValueError("Refinement requires a trained flow")
        if 0 < steps < 8 and not ordinary and not bool(model.shortcut_trained):
            raise ValueError("Few-step streaming requires shortcut training")
        self.model, self.steps, self.seed = model, steps, seed
        self.ordinary = ordinary or not bool(model.shortcut_trained)
        self.reset()

    def reset(self):
        c = self.model.config
        self.condition_history = [None] * len(self.model.condition_blocks)
        self.predictor_history = [None] * c.predictor_depth
        self.refiner_history = [[None] * c.refiner_depth for _ in range(self.steps)]
        self.frames = 0

    @torch.no_grad()
    def push(
        self,
        content,
        f0,
        voiced,
        energy,
        speaker,
        confidence=None,
        confidence_valid=None,
    ):
        m = self.model
        x = m.condition_input(
            content, f0, voiced, energy, speaker, confidence, confidence_valid
        )
        condition = cached_blocks(m.condition_blocks, x, self.condition_history)
        base = m.predictor_out(
            cached_blocks(
                m.predictor_blocks, m.predictor_in(condition), self.predictor_history
            )
        )
        if m.config.harmonic_detail:
            geometry = m.detail_pitch_features(f0, voiced)
            correction = m.spectral_detail(m.denormalize(base), geometry, condition)
            base = base + correction / m.mel_std
        if self.steps:
            if base.shape[0] != 1:
                raise ValueError("A streaming state belongs to one voice stream")
            noise = coordinate_noise(
                base.shape[-1], base.shape[1], self.frames, self.seed
            )
            z = torch.from_numpy(noise.T.copy()).to(base)[None]
            dt = 1 / self.steps
            for i in range(self.steps):
                t = z.new_full((1,), i * dt)
                d = z.new_full((1,), 0 if self.ordinary else dt)
                x = m.refiner_in(torch.cat([z, base], 1)) + m.refiner_condition(
                    condition
                )
                x = x + (m.time_embedding(t) + m.step_embedding(d))[:, :, None]
                x = cached_blocks(m.refiner_blocks, x, self.refiner_history[i])
                z = z + dt * m.refiner_out(x)
            base = base + m.residual_scale * z if m.config.prediction_centered else z
        self.frames += base.shape[-1]
        return m.denormalize(base)


class VocoderStream:
    """Emit finalized PCM while retaining synthesis halo and phase continuity.

    The right halo is withheld until enough future mel frames arrive. The retained
    left window covers causal spectral blocks plus transform/filter boundaries.
    Window recomputation reuses absolute noise and saved oscillator frame origins.
    """

    def __init__(self, model, seed=0):
        if not model.config.causal:
            raise ValueError("Streaming requires a causal vocoder package")
        self.model, self.seed = model, seed
        c = model.config
        halo_samples = 2 * model.stft.pad * c.streams + 2 * (c.filter_taps // 2)
        self.right = math.ceil(halo_samples / c.hop_length) + 1
        self.left = c.depth * (c.kernel_size - 1) + self.right
        self.reset()

    def reset(self):
        self.mel = self.f0 = self.voiced = None
        self.phase = None
        self.phases = []
        self.origin = self.received = self.emitted = 0
        self.closed = False

    @torch.no_grad()
    def push(self, mel, f0, voiced, final=False):
        if self.closed:
            raise ValueError("Stream is flushed; reset before another recording")
        c = self.model.config
        if (
            mel.shape[0] != 1
            or f0.shape != (1, mel.shape[-1])
            or voiced.shape != f0.shape
        ):
            raise ValueError("Stream requires one aligned mel/F0 sequence")
        if self.phase is None:
            self.phase = torch.zeros(1, dtype=torch.float64, device=mel.device)
        # Save a phase origin at every retained frame; windows never restart the oscillator.
        increments = f0[0].double() * (2 * math.pi * c.hop_length / c.sample_rate)
        starts = torch.cat([self.phase, self.phase + increments.cumsum(0)[:-1]])
        if len(increments):
            self.phases.extend(starts.remainder(2 * math.pi).unbind(0))
            self.phase = (self.phase + increments.sum()).remainder(2 * math.pi)
        self.mel = mel if self.mel is None else torch.cat([self.mel, mel], -1)
        self.f0 = f0 if self.f0 is None else torch.cat([self.f0, f0], -1)
        self.voiced = (
            voiced if self.voiced is None else torch.cat([self.voiced, voiced], -1)
        )
        self.received += mel.shape[-1]
        end = self.received if final else max(0, self.received - self.right)
        if end <= self.emitted:
            if final:
                self.closed = True
            return mel.new_empty(1, 0)
        excitation = coordinate_noise(
            self.f0.shape[-1] * c.hop_length,
            offset=self.origin * c.hop_length,
            seed=self.seed + 1,
        )
        noise = torch.from_numpy(excitation[:, 0].copy()).to(mel)[None]
        waveform = self.model(
            self.mel, self.f0, self.voiced, phase=self.phases[0].reshape(1), noise=noise
        )
        start_sample = (self.emitted - self.origin) * c.hop_length
        end_sample = (end - self.origin) * c.hop_length
        result = waveform[:, start_sample:end_sample].clone()
        self.emitted = end
        discard = max(0, self.emitted - self.left - self.origin)
        if discard:
            self.mel = self.mel[..., discard:].clone()
            self.f0 = self.f0[..., discard:].clone()
            self.voiced = self.voiced[..., discard:].clone()
            self.phases = self.phases[discard:]
            self.origin += discard
        self.closed = final
        return result


class FeatureStream:
    """Bounded PCM buffer; releases packets once their declared future is present."""

    def __init__(self, extractor):
        if extractor.config.profile != "bounded":
            raise ValueError(
                "Live conversion requires a bounded-profile trained package"
            )
        self.extractor = extractor
        self.reset()

    def reset(self):
        self.audio = np.empty(0, dtype=np.float32)
        self.origin = self.received = self.packet_index = 0
        self.closed = False

    def push(self, audio, final=False):
        if self.closed:
            raise ValueError("Frontend is flushed; reset before another recording")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Live audio must be finite mono PCM")
        self.audio = np.concatenate([self.audio, audio])
        self.received += len(audio)
        c, f = self.extractor.mel_config, self.extractor.config
        packets = []
        # One rational-rate grid interval covers the polyphase filter halo.
        halo = c.sample_rate // math.gcd(c.sample_rate, 16000)
        while self.packet_index * f.packet_seconds * c.sample_rate < self.received:
            end = (self.packet_index + 1) * f.packet_seconds + f.lookahead_seconds
            if not final and self.received < math.ceil(end * c.sample_rate) + halo:
                break
            packet = self.extractor.packet(
                self.audio, self.origin, self.received, self.packet_index, final
            )
            if packet is not None:
                packets.append(packet)
            self.packet_index += 1
            keep = max(
                0,
                int(
                    (self.packet_index * f.packet_seconds - f.context_seconds)
                    * c.sample_rate
                )
                - 2 * halo,
            )
            discard = max(0, keep - self.origin)
            self.audio = self.audio[discard:].copy()
            self.origin += discard
        self.closed = final
        return packets
