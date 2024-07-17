import os
import sys
import time
import torch
import librosa
import logging
import numpy as np
import soundfile as sf
import noisereduce as nr

from scipy.io import wavfile
from audio_upscaler import upscale

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0

    def load_hubert(self, embedder_model, embedder_model_custom):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        models, _, _ = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = models[0].to(self.config.device)
        self.hubert_model = (
            self.hubert_model.half()
            if self.config.is_half
            else self.hubert_model.float()
        )
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(input_audio_path, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.

        Args:
            input_audio_path (str): Path to the input audio file.
            reduction_strength (float): Strength of the noise reduction. Default is 0.7.
        """
        try:
            rate, data = wavfile.read(input_audio_path)
            reduced_noise = nr.reduce_noise(
                y=data, sr=rate, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            print(f"Error cleaning audio: {error}")
            return None

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
                print(f"Converting audio to {output_format} format...")
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
                    audio, orig_sr=sample_rate, target_sr=target_sr
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"Failed to convert audio to {output_format} format: {error}")

    def convert_audio(
        self,
        audio_input_path,
        audio_output_path,
        model_path,
        index_path,
        sid=0,
        f0_up_key=None,
        f0_file=None,
        f0_method=None,
        index_rate=None,
        resample_sr=0,
        rms_mix_rate=None,
        protect=None,
        hop_length=None,
        split_audio=False,
        f0_autotune=False,
        filter_radius=None,
        embedder_model=None,
        embedder_model_custom=None,
        clean_audio=False,
        clean_strength=0.7,
        export_format="WAV",
        upscale_audio=False,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            sid (int, optional): Speaker ID. Default is 0.
            f0_up_key (str, optional): Key for F0 up-sampling. Default is None.
            f0_file (str, optional): Path to the F0 file. Default is None.
            f0_method (str, optional): Method for F0 extraction. Default is None.
            index_rate (float, optional): Rate for index matching. Default is None.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            rms_mix_rate (float, optional): RMS mix rate. Default is None.
            protect (float, optional): Protection rate for certain audio segments. Default is None.
            hop_length (int, optional): Hop length for audio processing. Default is None.
            split_audio (bool, optional): Whether to split the audio for processing. Default is False.
            f0_autotune (bool, optional): Whether to use F0 autotune. Default is False.
            filter_radius (int, optional): Radius for filtering. Default is None.
            embedder_model (str, optional): Path to the embedder model. Default is None.
            embedder_model_custom (str, optional): Path to the custom embedder model. Default is None.
            clean_audio (bool, optional): Whether to clean the audio. Default is False.
            clean_strength (float, optional): Strength of the audio cleaning. Default is 0.7.
            export_format (str, optional): Format for exporting the audio. Default is "WAV".
            upscale_audio (bool, optional): Whether to upscale the audio. Default is False.
        """
        self.get_vc(model_path, sid)

        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")

            if upscale_audio == "True":
                upscale(audio_input_path, audio_input_path)

            audio = load_audio(audio_input_path, 16000)
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model:
                self.load_hubert(embedder_model, embedder_model_custom)

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

            if split_audio == "True":
                result, new_dir_path = process_audio(audio_input_path)
                if result == "Error":
                    return "Error with Split Audio", None

                dir_path = (
                    new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                )
                if dir_path:
                    paths = [
                        os.path.join(root, name)
                        for root, _, files in os.walk(dir_path, topdown=False)
                        for name in files
                        if name.endswith(".wav") and root == dir_path
                    ]
                try:
                    for path in paths:
                        self.convert_audio(
                            path,
                            path,
                            model_path,
                            index_path,
                            sid,
                            f0_up_key,
                            None,
                            f0_method,
                            index_rate,
                            resample_sr,
                            rms_mix_rate,
                            protect,
                            hop_length,
                            False,
                            f0_autotune,
                            filter_radius,
                            embedder_model,
                            embedder_model_custom,
                            clean_audio,
                            clean_strength,
                            export_format,
                            upscale_audio,
                        )
                except Exception as error:
                    print(error)
                    return f"Error {error}"
                print("Finished processing segmented audio, now merging audio...")
                merge_timestamps_file = os.path.join(
                    os.path.dirname(new_dir_path),
                    f"{os.path.basename(audio_input_path).split('.')[0]}_timestamps.txt",
                )
                self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                os.remove(merge_timestamps_file)
            else:
                audio_opt = self.vc.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    audio_input_path,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.use_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    hop_length,
                    f0_autotune,
                    f0_file=f0_file,
                )

            if audio_output_path:
                sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")

            if clean_audio == "True":
                cleaned_audio = self.remove_audio_noise(
                    audio_output_path, clean_strength
                )
                if cleaned_audio is not None:
                    sf.write(
                        audio_output_path, cleaned_audio, self.tgt_sr, format="WAV"
                    )

            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )

        except Exception as error:
            print(f"Voice conversion failed: {error}")

    def get_vc(self, weight_root, sid):
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

        self.load_model(weight_root)

        if self.cpt is not None:
            self.setup_network()
            self.setup_vc_instance()

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu")
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                is_half=self.config.is_half,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = (
                self.net_g.half() if self.config.is_half else self.net_g.float()
            )

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
