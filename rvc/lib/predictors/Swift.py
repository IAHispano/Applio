import os
import numpy as np
import onnxruntime
import torch

SAMPLE_RATE = 16000
HOP = 256
FRAME_PERIOD = HOP / SAMPLE_RATE
FMIN = 46.875
FMAX = 2093.75
BIN_RATIO = (FMAX / FMIN) ** (1 / 94)
LOOKAHEAD_FRAMES = 10
LEFT_FRAMES = 11
WINDOW_FRAMES = 1875
SILENCE_PEAK = 1e-3


def _to_mono(audio) -> np.ndarray:
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio)
    dtype = array.dtype
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        raise TypeError(f"audio must be a real integer or float array, got {dtype}")
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError("audio has no channels")
        channels = array.shape[1]
        array = (
            array[:, 0]
            if channels == 1
            else sum(array[:, c] for c in range(channels)) / channels
        )
    elif array.ndim > 2:
        array = array.squeeze()
        if array.ndim > 1:
            array = array.mean(axis=-1)
    if np.issubdtype(dtype, np.integer):
        full_scale = 2.0 ** (np.iinfo(dtype).bits - 1)
        offset = 0.0 if np.issubdtype(dtype, np.signedinteger) else full_scale
        array = (array - offset) / full_scale
    signal = np.array(array, dtype=np.float32)
    return signal


class SwiftPredictor:
    """
    SWIFT pitch predictor for fundamental frequency estimation.
    Uses a lightweight ONNX model (~14k parameters) designed for fast, accurate monophonic pitch tracking.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        threads: int = None,
        spin: bool = False,
    ):
        if model_path is None or not os.path.exists(model_path):
            candidates = [
                os.path.join("rvc", "models", "predictors", "swift.onnx"),
                os.path.join(os.path.dirname(__file__), "swift.onnx"),
            ]
            try:
                import swift_f0

                candidates.append(
                    os.path.join(os.path.dirname(swift_f0.__file__), "model.onnx")
                )
            except Exception:
                pass

            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if model_path is None or not os.path.exists(model_path):
            target_path = os.path.join("rvc", "models", "predictors", "swift.onnx")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            url = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/swift.onnx"
            try:
                import urllib.request

                print(f"Downloading SWIFT ONNX model from {url}...")
                urllib.request.urlretrieve(url, target_path)
                model_path = target_path
            except Exception as e:
                raise FileNotFoundError(
                    f"SWIFT ONNX model not found at '{target_path}' and download failed: {e}"
                )

        self.model_path = model_path
        self.device = str(device).lower()

        options = onnxruntime.SessionOptions()
        if threads is not None and threads > 0:
            options.intra_op_num_threads = int(threads)
        if not spin:
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        providers = ["CPUExecutionProvider"]
        if (
            "cuda" in self.device
            and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")

        self.session = onnxruntime.InferenceSession(
            self.model_path,
            options,
            providers=providers,
        )

    def _run(self, audio: np.ndarray, fmin: float, fmax: float):
        pitch, confidence = self.session.run(
            ["pitch", "confidence"],
            {
                "audio": audio[None, :],
                "fmin": np.asarray(fmin, dtype=np.float32),
                "fmax": np.asarray(fmax, dtype=np.float32),
            },
        )
        pitch = np.asarray(pitch[0], dtype=np.float64)
        confidence = np.asarray(confidence[0], dtype=np.float64)
        n = len(confidence)
        hops = audio[: n * HOP].reshape(n, HOP) if len(audio) >= HOP else audio[None, :]
        confidence[np.abs(hops).max(axis=1) < SILENCE_PEAK] = 0.0
        return pitch, confidence

    def detect(
        self,
        audio,
        sample_rate: int = SAMPLE_RATE,
        fmin: float = FMIN,
        fmax: float = FMAX,
    ):
        signal = _to_mono(audio)
        if signal.size == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )

        if sample_rate != SAMPLE_RATE:
            try:
                import soxr

                signal = soxr.resample(signal, sample_rate, SAMPLE_RATE)
            except ImportError:
                import librosa

                signal = librosa.resample(
                    signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                )
            if signal.size == 0:
                return (
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                )

        fmin = max(FMIN, float(fmin))
        fmax = min(FMAX, float(fmax))
        if fmax <= fmin:
            fmin, fmax = FMIN, FMAX

        n = max(1, len(signal) // HOP)
        parts = []
        for start in range(0, n, WINDOW_FRAMES):
            end = min(start + WINDOW_FRAMES, n)
            left = max(0, start - LEFT_FRAMES)
            window = (
                signal[left * HOP : (end + LOOKAHEAD_FRAMES) * HOP]
                if end < n
                else signal[left * HOP :]
            )
            parts.append(
                [
                    val[start - left : end - left]
                    for val in self._run(window, fmin, fmax)
                ]
            )

        pitch = np.concatenate([part[0] for part in parts])
        confidence = np.concatenate([part[1] for part in parts])
        timestamps = np.arange(len(pitch)) * FRAME_PERIOD
        return timestamps, pitch, confidence

    @staticmethod
    def _repair_subharmonics(pitch, confidence, frame_period=0.016):
        pitch = np.asarray(pitch, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        corrected = pitch.copy()
        repaired = np.zeros(len(pitch), dtype=bool)
        i = 1
        while i < len(pitch) - 1:
            if not np.isfinite(pitch[i - 1 : i + 1]).all() or np.any(
                pitch[i - 1 : i + 1] <= 0
            ):
                i += 1
                continue
            if confidence[i - 1] < 0.5 or not 0.3 <= confidence[i] < 0.95:
                i += 1
                continue
            ratio = pitch[i - 1] / pitch[i]
            factor = min((2, 3), key=lambda value: abs(np.log2(ratio / value)))
            if abs(1200 * np.log2(ratio / factor)) > 100:
                i += 1
                continue
            j = i
            while j < len(pitch) and (j - i) * frame_period < 1.0:
                if (
                    not np.isfinite(pitch[j])
                    or pitch[j] <= 0
                    or not 0.3 <= confidence[j] < 0.95
                ):
                    break
                previous = pitch[i - 1] if j == i else pitch[j - 1] * factor
                if abs(1200 * np.log2(pitch[j] * factor / previous)) > 100:
                    break
                j += 1
            if (
                j > i
                and j < len(pitch)
                and np.isfinite(pitch[j])
                and pitch[j] > 0
                and confidence[j] >= 0.5
            ):
                return_error = abs(1200 * np.log2(pitch[j] / (pitch[j - 1] * factor)))
                anchor_error = abs(1200 * np.log2(pitch[j] / pitch[i - 1]))
                if return_error <= 100 and anchor_error <= 150:
                    corrected[i:j] *= factor
                    repaired[i:j] = True
                    i = j + 1
                    continue
            i += 1
        return corrected, repaired

    def infer_from_audio(
        self,
        audio,
        sample_rate: int = SAMPLE_RATE,
        p_len: int = None,
        hop_size: int = 160,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        thred: float = 0.5,
    ) -> np.ndarray:
        """
        Infers F0 contour aligned to the target timeline (hop_size and p_len).
        """
        signal = _to_mono(audio)
        if p_len is None:
            p_len = len(signal) // hop_size

        if p_len <= 0:
            return np.zeros(0, dtype=np.float64)

        if signal.size == 0:
            return np.zeros(p_len, dtype=np.float64)

        timestamps, pitch, confidence = self.detect(
            signal,
            sample_rate=sample_rate,
            fmin=f0_min,
            fmax=f0_max,
        )

        if len(timestamps) == 0:
            return np.zeros(p_len, dtype=np.float64)

        target_timestamps = np.arange(p_len) * (hop_size / sample_rate)
        threshold = float(thred) if thred is not None else 0.5
        corrected, repaired = self._repair_subharmonics(pitch, confidence, FRAME_PERIOD)
        repaired &= (corrected >= f0_min) & (corrected <= f0_max)
        pitch = np.where(repaired, corrected, pitch)
        confidence = np.where(repaired, np.maximum(confidence, threshold), confidence)

        conf_interp = np.interp(
            target_timestamps,
            timestamps,
            confidence,
            left=0.0,
            right=0.0,
        )
        voiced = (confidence >= threshold) & np.isfinite(pitch) & (pitch > 0)

        if np.any(voiced):
            pitch_interp = np.exp2(
                np.interp(
                    target_timestamps,
                    timestamps[voiced],
                    np.log2(pitch[voiced]),
                )
            )
            pitch_interp[conf_interp < threshold] = 0.0
        else:
            pitch_interp = np.zeros(p_len, dtype=np.float64)

        return pitch_interp


SwiftF0Predictor = SwiftPredictor
