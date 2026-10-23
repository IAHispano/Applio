import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soxr
import time
import torch
import librosa
import logging
import numpy as np
import soundfile as sf
from functools import wraps
from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

import rvc.lib.zluda  # sets MIOPEN_FIND_MODE on AMD; must land before the first conv
from rvc.infer.pipeline import Pipeline as VC
from rvc.infer.flow_pipeline import FlowPipeline, FlowSynthesizer
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm.loudness import limit_peaks
from rvc.configs.config import inference_config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# High-pass on the input of a Rectified Flow model: a sub-bass pop would set
# the peak the input is normalised to and read as a loud frame of its energy
FLOW_INPUT_HIGHPASS_HZ = 40.0


def _toast(message, warning=False):
    """Status message helper (prints to stdout; the web UI streams engine logs)."""
    print(message)


def _retry_mps_on_cpu(convert):
    @wraps(convert)
    def wrapped(self, *args, **kwargs):
        try:
            return convert(self, *args, **kwargs)
        except (RuntimeError, NotImplementedError, TypeError) as error:
            message = str(error).lower()
            if self.config.device != "mps" or not any(
                marker in message
                for marker in (
                    "mps",
                    "metal",
                    "coreml",
                    "placeholder storage has not been allocated",
                )
            ):
                raise
            print(f"[!] MPS inference failed ({error}); retrying conversion on CPU.")
            self.config.device = "cpu"
            self.config.gpu_name = None
            for model in (self.hubert_model, self.net_g):
                if model is not None:
                    model.to("cpu")
            # Discard cached pitch predictors and device-bound pipeline state.
            self.setup_vc_instance()
            return convert(self, *args, **kwargs)

    return wrapped


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, config=None):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = config if config is not None else inference_config()
        if self.config.device == "mps":
            print(
                "Using Apple Silicon MPS for inference (FP32; CPU spectral frontends)."
            )
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None
        self.loaded_model_key = None
        self.model_path = None
        self.model_checksum = None
        self.onnx_decoder = False
        self._onnx_request = None
        self._onnx_precision = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output audio file.
            output_format (str): Desired audio format (e.g., "WAV", "MP3").
        """
        try:
            if output_format != "WAV":
                print(f"Saving audio as {output_format}...")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"An error occurred converting the audio format: {error}")

    @_retry_mps_on_cpu
    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1.0,
        protect: float = 0.5,
        hop_length: int = 128,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        export_format: str = "WAV",
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        inference_engine: str = None,
        onnx_precision: str = "auto",
        **kwargs,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            pitch (int): Key for F0 up-sampling.
            index_rate (float): Rate for index matching.
            volume_envelope (int): RMS mix rate.
            protect (float): Protection rate for certain audio segments.
            hop_length (int): Hop length for audio processing.
            f0_method (str): Method for F0 extraction.
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            f0_autotune (bool): Whether to use F0 autotune.
            export_format (str): Format for exporting the audio.
            f0_file (str): Path to the F0 file.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        if not model_path:
            print("No model path provided. Aborting conversion.")
            return

        self.get_vc(model_path, sid, inference_engine, onnx_precision)

        start_time = time.time()
        print(f"Converting audio '{audio_input_path}'...")

        # A Rectified Flow model measures the loudness, breathiness and tension
        # of the input at its own sampling rate, as its training does: the audio
        # is loaded at that rate and the 16 kHz copy is resampled from it
        audio_full = None
        flow_options = {}
        if isinstance(self.vc, FlowPipeline):
            self.net_g.set_vocoder(kwargs.get("flow_vocoder", ""))
            flow_options = {
                name: kwargs[f"flow_{name}"]
                for name in (
                    "steps",
                    "cfg_scale",
                    "content_guidance",
                    "sampler",
                    "guidance_rescale",
                    "guidance_until",
                    "formant_shift",
                    "tension_strength",
                )
                if f"flow_{name}" in kwargs
            }
            flow_sr = self.net_g.data["sample_rate"]
            audio_full = load_audio_infer(audio_input_path, flow_sr, **kwargs)
            highpass = signal.butter(
                4, FLOW_INPUT_HIGHPASS_HZ, "highpass", fs=flow_sr, output="sos"
            )
            audio_full = signal.sosfiltfilt(highpass, audio_full).astype(np.float32)
            audio = librosa.resample(
                audio_full, orig_sr=flow_sr, target_sr=16000, res_type="soxr_vhq"
            )
            flow_options["audio_full"] = audio_full
        else:
            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )
        audio_max = np.abs(audio).max() / 0.95

        # The loudness input of a Rectified Flow model is absolute, so a quiet
        # input is brought up to the peak the training data has as well. The
        # output stays at the level of the model, as that of an RVC model does
        if audio_max > 1 or (audio_full is not None and audio_max > 0):
            audio /= audio_max
            if audio_full is not None:
                audio_full /= audio_max

        if not self.hubert_model or embedder_model != self.last_embedder_model:
            self.load_hubert(embedder_model, embedder_model_custom)
            self.last_embedder_model = embedder_model

        file_index = (
            index_path.strip()
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip()
            .replace("trained", "added")
        )

        if self.tgt_sr != resample_sr >= 16000:
            self.tgt_sr = resample_sr

        audio_opt = self.vc.pipeline(
            model=self.hubert_model,
            net_g=self.net_g,
            sid=sid,
            audio=audio,
            pitch=pitch,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            pitch_guidance=self.use_f0,
            volume_envelope=volume_envelope,
            version=self.version,
            protect=protect,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            proposed_pitch=proposed_pitch,
            proposed_pitch_threshold=proposed_pitch_threshold,
            **flow_options,
        )

        if audio_full is not None:
            if kwargs.get("flow_peak_limiter", False):
                audio_opt = limit_peaks(audio_opt.astype(np.float32), self.tgt_sr)
            else:
                audio_opt = audio_opt / max(1.0, float(np.abs(audio_opt).max()) / 0.99)

        if (
            kwargs.get("cleanup_mode", "off") != "off"
            and kwargs.get("cleanup_stage", "input") == "output"
        ):
            from rvc.lib.tools.audio_cleanup import cleanup_audio

            audio_opt = cleanup_audio(
                audio_opt,
                self.tgt_sr,
                kwargs["cleanup_mode"],
                kwargs.get("cleanup_strength", 12),
            )
        sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
        output_path_format = (
            os.path.splitext(audio_output_path)[0] + f".{export_format.lower()}"
        )
        audio_output_path = self.convert_audio_format(
            audio_output_path, output_path_format, export_format
        )

        elapsed_time = time.time() - start_time
        print(
            f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
        )

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.

        Args:
            audio_input_paths (str): List of paths to the input audio files.
            audio_output_path (str): Path to the output audio file.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"), "w"
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            print(f"Converting audio batch '{audio_input_paths}'...")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.lower().endswith(
                    (
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "opus",
                        "m4a",
                        "mp4",
                        "aac",
                        "alac",
                        "wma",
                        "aiff",
                        "webm",
                        "ac3",
                    )
                )
            ]
            print(f"Detected {len(audio_files)} audio files for inference.")
            total = len(audio_files)
            converted = skipped = 0
            _next_milestone = 25
            _toast(f"Batch conversion started: {total} files")
            for a in audio_files:
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.splitext(a)[0] + "_output.wav"
                new_output = os.path.join(audio_output_path, new_output)
                if os.path.exists(new_output):
                    skipped += 1
                else:
                    self.convert_audio(
                        audio_input_path=new_input,
                        audio_output_path=new_output,
                        **kwargs,
                    )
                    converted += 1
                processed = converted + skipped
                # Milestone toast: fires at the first file past each threshold
                # (the label carries the threshold). processed < total
                # suppresses the 100% milestone (it would double-announce with
                # the terminal toast); total >= 8 keeps small batches
                # toast-free between start and terminal.
                if (
                    total >= 8
                    and processed < total
                    and processed * 100 // total >= _next_milestone
                ):
                    _toast(f"{processed}/{total} files converted ({_next_milestone}%)")
                    _next_milestone += 25
            print(f"Conversion completed at '{audio_input_paths}'.")
            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
            _toast(
                f"Batch conversion completed: {converted} converted, "
                f"{skipped} skipped in {elapsed_time:.0f}s"
            )
        except Exception as e:
            _toast(f"Batch conversion failed: {e}", warning=True)
            raise
        finally:
            pid_path = os.path.join(now_dir, "assets", "infer_pid.txt")
            if os.path.exists(pid_path):
                try:
                    os.remove(pid_path)
                except Exception:
                    pass

    def get_vc(self, weight_root, sid, inference_engine=None, onnx_precision="auto"):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int): Speaker ID.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not weight_root:
                return

        if inference_engine not in (None, "torch", "onnx"):
            raise ValueError("Inference engine must be torch or onnx")
        if onnx_precision not in ("auto", "float32", "float16"):
            raise ValueError("ONNX precision must be auto, float32 or float16")
        requested = (
            os.environ.get("APPLIO_EXPERIMENTAL_ONNX_INFERENCE") or None
            if inference_engine is None
            else "auto" if inference_engine == "onnx" else None
        )
        precision = (
            (
                os.environ.get("APPLIO_ONNX_PRECISION") or None
                if onnx_precision == "auto"
                else onnx_precision
            )
            if requested
            else None
        )
        checkpoint = os.path.normcase(os.path.abspath(weight_root))
        stat = os.stat(checkpoint)
        key = (
            checkpoint,
            stat.st_mtime_ns,
            stat.st_size,
            str(self.config.device),
            requested,
            precision,
            os.environ.get("APPLIO_ONNX_CACHE_DIR") if requested else None,
        )
        if self.loaded_model_key != key:
            # Prepare an entire replacement first. A failed export/model swap
            # must not leave a persistent worker with the wrong active voice.
            replacement = VoiceConverter(config=self.config)
            replacement._onnx_request, replacement._onnx_precision = (
                requested,
                precision,
            )
            replacement.load_model(checkpoint)
            replacement.setup_network()
            replacement.setup_vc_instance()
            for name in (
                "net_g",
                "cpt",
                "tgt_sr",
                "vc",
                "n_spk",
                "version",
                "use_f0",
                "vocoder",
                "text_enc_hidden_dim",
                "model_path",
                "model_checksum",
                "onnx_decoder",
            ):
                setattr(self, name, getattr(replacement, name))
            self.loaded_model, self.loaded_model_key = weight_root, key
            self._onnx_request, self._onnx_precision = requested, precision

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        self.hubert_model = self.net_g = self.n_spk = self.vc = None
        self.tgt_sr = self.cpt = None
        self.loaded_model = self.loaded_model_key = self.last_embedder_model = None
        self.onnx_decoder = False
        self.model_path = self.model_checksum = None
        self._onnx_request = self._onnx_precision = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.model_path = weight_root
        self.model_checksum = None
        if self._onnx_request:
            from rvc.realtime.onnx_decoder import checkpoint_hash

            self.model_checksum = checkpoint_hash(weight_root)
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=True)
            if os.path.isfile(weight_root)
            else None
        )
        if (
            self.model_checksum is not None
            and checkpoint_hash(weight_root) != self.model_checksum
        ):
            raise ValueError("Voice checkpoint changed while loading; retry")

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None and self.cpt.get("kind") == "rectified_flow":
            self.tgt_sr = self.cpt["config"]["data"]["sample_rate"]
            self.use_f0 = 1
            self.version = "v2"
            self.text_enc_hidden_dim = 768
            self.vocoder = "Rectified Flow"
            self.net_g = FlowSynthesizer(self.cpt)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
        elif self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            if self._onnx_request:
                # Match the realtime export's materialized CPU weights so the
                # same checkpoint/precision cache works in either workflow.
                for module in self.net_g.dec.modules():
                    if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
                        torch.nn.utils.remove_weight_norm(module)
                    if hasattr(module, "parametrizations"):
                        for name in list(module.parametrizations):
                            torch.nn.utils.parametrize.remove_parametrizations(
                                module, name, leave_parametrized=True
                            )
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            if self._onnx_request:
                from rvc.realtime.onnx_decoder import OnnxDecoder, resolve_decoder_path

                path = resolve_decoder_path(
                    self.net_g.dec,
                    self.model_path,
                    self._onnx_request,
                    self.config.device,
                    expected_checkpoint_hash=self.model_checksum,
                    precision=self._onnx_precision,
                )
                self.net_g.dec = OnnxDecoder(
                    self.net_g.dec,
                    self.model_path,
                    path,
                    self.config.device,
                    expected_checkpoint_hash=self.model_checksum,
                    capture=False,
                )
                if (
                    self._onnx_precision is not None
                    and self.net_g.dec.precision != self._onnx_precision
                ):
                    raise ValueError(
                        "ONNX export precision does not match the requested precision"
                    )
                self.onnx_decoder = True

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None and self.cpt.get("kind") == "rectified_flow":
            self.vc = FlowPipeline(self.tgt_sr, self.config)
            self.n_spk = self.cpt["speaker_count"]
        elif self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Applio Voice Converter Inference")
    # File / folder paths
    parser.add_argument(
        "--input-path", "--audio-input-path", dest="input_path", default=None
    )
    parser.add_argument(
        "--output-path", "--audio-output-path", dest="output_path", default=None
    )
    parser.add_argument(
        "--input-folder", "--audio-input-paths", dest="input_folder", default=None
    )
    parser.add_argument("--output-folder", dest="output_folder", default=None)
    parser.add_argument("--pth-path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--index-path", default="")
    parser.add_argument("--inference-engine", choices=("torch", "onnx"), default=None)
    parser.add_argument(
        "--onnx-precision", choices=("auto", "float32", "float16"), default="auto"
    )

    # Pitch and F0
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--volume-envelope", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--f0-autotune", action="store_true", default=False)
    parser.add_argument("--f0-autotune-strength", type=float, default=1.0)
    parser.add_argument("--proposed-pitch", action="store_true", default=False)
    parser.add_argument("--proposed-pitch-threshold", type=float, default=155.0)

    # Embedder
    parser.add_argument("--embedder-model", default="contentvec")
    parser.add_argument("--embedder-model-custom", default=None)
    parser.add_argument("--export-format", default="WAV")
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--sid", type=int, default=0)

    # Formant
    parser.add_argument("--formant-shifting", action="store_true", default=False)
    parser.add_argument("--formant-qfrency", type=float, default=1.0)
    parser.add_argument("--formant-timbre", type=float, default=1.0)

    parser.add_argument(
        "--cleanup-mode", choices=("off", "light", "speech"), default="off"
    )
    parser.add_argument("--cleanup-strength", type=float, default=12)
    parser.add_argument("--cleanup-stage", choices=("input", "output"), default="input")

    args = parser.parse_args()
    infer_kwargs = vars(args)
    input_folder = infer_kwargs.pop("input_folder")
    output_folder = infer_kwargs.pop("output_folder")
    input_path = infer_kwargs.pop("input_path")
    output_path = infer_kwargs.pop("output_path")

    if input_folder and output_folder:
        vc = VoiceConverter()
        vc.convert_audio_batch(
            audio_input_paths=input_folder,
            audio_output_path=output_folder,
            **infer_kwargs,
        )
        print(f"Files from {input_folder} inferred successfully.")
    elif input_path and output_path:
        vc = VoiceConverter()
        vc.convert_audio(
            audio_input_path=input_path,
            audio_output_path=output_path,
            **infer_kwargs,
        )
        print(f"File {input_path} inferred successfully.")
    else:
        print(
            "Error: Specify either --input-path and --output-path, or --input-folder and --output-folder."
        )
        sys.exit(1)
