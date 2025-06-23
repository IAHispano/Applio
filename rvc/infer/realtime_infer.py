import os
import sys
import time
import torch
import numpy as np
import librosa
import faiss
from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config import Config
from rvc.lib.utils import load_embedding
from rvc.infer.pipeline import Pipeline as VC, bh, ah # Import filter coefficients
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import FCPEF0Predictor

# Suppress faiss logging if possible, or set level if it has a logger
try:
    faiss.logger.setLevel(logging.WARNING)
except AttributeError: # Older faiss might not have this
    pass


class RealtimeVoiceConverter:
    def __init__(self):
        self.config = Config()
        self.hubert_model = None
        self.net_g = None
        self.vc_pipeline = None # This will be an instance of the original Pipeline (VC)

        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None # bool
        self.tgt_sr = None # Target sample rate of the RVC model output
        self.sid = None
        self.sid_tensor = None

        self.index = None
        self.big_npy = None

        self.current_model_path = None
        self.current_index_path = None
        self.current_embedder_model = None

        # F0 predictors (initialized when model is loaded based on tgt_sr)
        self.model_rmvpe = None
        # self.model_fcpe = None # FCPE is initialized on-demand in original pipeline

        print("RealtimeVoiceConverter initialized.")

    def _load_hubert(self, embedder_model_name: str, embedder_model_custom_path: str = None):
        # Adapted from VoiceConverter.load_hubert
        if self.hubert_model is not None and self.current_embedder_model == embedder_model_name:
            print("Hubert model already loaded and matches current selection.")
            return True

        print(f"Loading Hubert model: {embedder_model_name}")
        try:
            self.hubert_model = load_embedding(embedder_model_name, embedder_model_custom_path)
            self.hubert_model = self.hubert_model.to(self.config.device).float()
            self.hubert_model.eval()
            self.current_embedder_model = embedder_model_name
            print("Hubert model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading Hubert model: {e}")
            self.hubert_model = None
            return False

    def _load_rvc_model(self, model_path: str, sid: int):
        # Adapted from VoiceConverter.get_vc and its sub-methods
        if self.net_g is not None and self.current_model_path == model_path and self.sid == sid:
            print("RVC model already loaded and SID matches.")
            return True

        print(f"Loading RVC model from: {model_path} for SID: {sid}")
        try:
            self.cpt = torch.load(model_path, map_location="cpu", weights_only=False) # weights_only=False needed for config

            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0] # n_spk
            self.use_f0 = self.cpt.get("f0", 1) == 1 # Ensure boolean
            self.version = self.cpt.get("version", "v1")

            text_enc_hidden_dim = 768 if self.version == "v2" else 256
            vocoder = self.cpt.get("vocoder", "HiFi-GAN")

            from rvc.lib.algorithm.synthesizers import Synthesizer # Local import to avoid circular deps if any
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=text_enc_hidden_dim,
                vocoder=vocoder,
            )
            del self.net_g.enc_q # Remove unnecessary part
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g = self.net_g.to(self.config.device) # .float() is done by Synthesizer based on is_half
            self.net_g.eval()

            self.n_spk = self.cpt["config"][-3]
            self.current_model_path = model_path
            self.sid = sid
            self.sid_tensor = torch.tensor(self.sid, device=self.config.device).unsqueeze(0).long()

            # Initialize the VC pipeline with the new target SR
            self.vc_pipeline = VC(self.tgt_sr, self.config)
            # Re-initialize F0 predictors as they might depend on device/config from VC pipeline
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                device=self.config.device, # Use device from config
            )
            # FCPE is loaded on demand by original pipeline's get_f0 if chosen

            print(f"RVC model loaded. Target SR: {self.tgt_sr}, Use F0: {self.use_f0}, Version: {self.version}")
            return True
        except Exception as e:
            print(f"Error loading RVC model: {e}")
            import traceback
            traceback.print_exc()
            self.net_g = None
            self.cpt = None
            self.vc_pipeline = None
            return False

    def _load_index(self, index_path: str):
        if not index_path or not os.path.exists(index_path):
            print("Index path is empty or file does not exist. Clearing index.")
            self.index = None
            self.big_npy = None
            self.current_index_path = None
            return True # Not an error if no index is desired

        if self.index is not None and self.current_index_path == index_path:
            print("Index file already loaded.")
            return True

        print(f"Loading index file from: {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            self.current_index_path = index_path
            print("Index file loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading index file: {e}")
            self.index = None
            self.big_npy = None
            return False

    def load_resources(self, model_path, index_path, sid, embedder_model_name="contentvec", embedder_model_custom_path=None):
        """Loads all necessary model resources."""
        if not model_path:
            print("Error: RVC Model path is required.")
            return False, "RVC Model path is required."

        hubert_loaded = self._load_hubert(embedder_model_name, embedder_model_custom_path)
        if not hubert_loaded:
            return False, "Failed to load Hubert model."

        rvc_loaded = self._load_rvc_model(model_path, sid)
        if not rvc_loaded:
            return False, f"Failed to load RVC model from {model_path}."

        index_loaded = self._load_index(index_path)
        if not index_loaded: # This is only a problem if index_path was actually provided
             if index_path and index_path.strip() != "":
                return False, f"Failed to load Index file from {index_path}."

        print(f"All resources loaded. Target SR: {self.tgt_sr}, Hubert: {self.current_embedder_model}, RVC Model: {self.current_model_path}, Index: {self.current_index_path}")
        return True, "Models loaded successfully."

    def is_ready(self):
        return self.hubert_model is not None and self.net_g is not None and self.vc_pipeline is not None

    def process_chunk(self, audio_chunk_16khz_float32, pitch_change=0, f0_method="rmvpe",
                        index_rate=0.75, protect_val=0.33,
                        f0_autotune=False, f0_autotune_strength=0.8,
                        output_volume_envelope_mix=1.0):
        """
        Processes a single audio chunk for real-time voice conversion.
        Assumes audio_chunk_16khz_float32 is mono, at 16000 Hz.
        Output will be at self.tgt_sr.
        """
        if not self.is_ready():
            # print("RealtimeVoiceConverter not ready. Call load_resources first.")
            return np.zeros_like(audio_chunk_16khz_float32, dtype=np.float32) # Return silence of same input length (approx)

        if audio_chunk_16khz_float32.ndim > 1 and audio_chunk_16khz_float32.shape[1] == 1:
            audio_chunk_16khz_float32 = audio_chunk_16khz_float32.flatten()

        # 0. High-pass filter (copied from Pipeline)
        audio_filtered = signal.filtfilt(bh, ah, audio_chunk_16khz_float32)

        # 1. Pad the input chunk (mimicking Pipeline's audio_pad for short audio)
        # self.vc_pipeline.t_pad is calculated based on 16kHz
        # self.vc_pipeline.t_pad_tgt is calculated based on self.tgt_sr
        # This padding is essential for the model to have context.
        audio_padded = np.pad(audio_filtered, (self.vc_pipeline.t_pad, self.vc_pipeline.t_pad), mode='reflect')

        p_len = audio_padded.shape[0] // self.vc_pipeline.window

        # 2. F0 Extraction
        f0_course = None
        f0_fine = None
        if self.use_f0:
            # Use the get_f0 method from the vc_pipeline instance
            # Need to ensure correct f0 predictor is used if not rmvpe/fcpe
            # The 'input_audio_path' argument to get_f0 in original pipeline is only for hybrid method's caching.
            # For realtime, we pass a dummy constant string, and avoid hybrid for now.
            if f0_method == "rmvpe" and self.model_rmvpe:
                 # RMVPE needs audio normalized differently than crepe/fcpe inside get_f0
                f0_fine = self.model_rmvpe.infer_from_audio(audio_padded, thred=0.03) # Raw F0
            elif f0_method == "fcpe":
                # FCPE needs to be initialized here if not already handled by vc_pipeline setup
                model_fcpe = FCPEF0Predictor(
                    os.path.join("rvc", "models", "predictors", "fcpe.pt"),
                    f0_min=int(self.vc_pipeline.f0_min), f0_max=int(self.vc_pipeline.f0_max),
                    dtype=torch.float32, device=self.config.device,
                    sample_rate=self.vc_pipeline.sample_rate, threshold=0.03,
                )
                f0_fine = model_fcpe.compute_f0(audio_padded, p_len=p_len)
                del model_fcpe # clean up
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else: # Fallback or other methods like crepe (might be slow)
                # This path is simplified; original get_f0 has more logic for crepe.
                # For real-time, stick to rmvpe or fcpe if possible.
                print(f"Warning: Using simplified F0 path for {f0_method}. Consider rmvpe/fcpe.")
                # A dummy F0 if other methods are too complex for direct integration here
                f0_fine = np.zeros(p_len, dtype=np.float32)


            if f0_autotune:
                # vc_pipeline.autotune is an Autotune instance.
                f0_fine = self.vc_pipeline.autotune.autotune_f0(f0_fine, f0_autotune_strength)

            f0_fine *= pow(2, pitch_change / 12)

            # Convert to F0 coarse (similar to get_f0 logic)
            f0_mel = 1127 * np.log(1 + f0_fine / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.vc_pipeline.f0_mel_min) * 254 / (self.vc_pipeline.f0_mel_max - self.vc_pipeline.f0_mel_min) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            f0_course = np.rint(f0_mel).astype(int)

            f0_course_tensor = torch.tensor(f0_course, device=self.config.device).unsqueeze(0).long()
            f0_fine_tensor = torch.tensor(f0_fine, device=self.config.device).unsqueeze(0).float()
        else: # No F0 usage
            f0_course_tensor = None
            f0_fine_tensor = None

        # 3. Voice Conversion (mimicking Pipeline.voice_conversion)
        with torch.no_grad():
            # 3.1 Hubert features
            feats = torch.from_numpy(audio_padded).float().to(self.config.device)
            if feats.ndim == 1: feats = feats.unsqueeze(0) # Batch dim
            if feats.ndim == 2 and feats.shape[0] > 1: # If stereo, take mean (should be mono already)
                feats = torch.mean(feats, dim=0, keepdim=True)

            # Original pipeline does feats = model(feats) where model is hubert
            # The dict output is new in some Hubert versions.
            hubert_out = self.hubert_model(feats)
            if isinstance(hubert_out, dict) and "last_hidden_state" in hubert_out:
                 feats = hubert_out["last_hidden_state"]
            else: # Older direct tensor output
                 feats = hubert_out

            feats = self.hubert_model.final_proj(feats[0]).unsqueeze(0) if self.version == "v1" else feats

            # 3.2 Indexing
            feats_for_protection = feats.clone() if self.use_f0 and protect_val < 0.5 else None

            if self.index is not None and self.big_npy is not None and index_rate > 0:
                feats = self.vc_pipeline._retrieve_speaker_embeddings(feats, self.index, self.big_npy, index_rate)

            # 3.3 Feature Upsampling
            feats = torch.nn.functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            # 3.4 Length adjustment for pitch and generator
            # The p_len for generator should match the feature length after interpolation
            # And also the length of pitch features.
            # Max length for pitch is p_len (audio_padded_len / window_size)
            # Max length for feats is also related to p_len * 2 (due to interpolate)

            gen_p_len = min(feats.shape[1], p_len) # This might need adjustment based on how pitch was calculated
                                                   # If pitch was calculated for 'p_len' frames, and features are upsampled,
                                                   # then pitch needs to be used up to 'gen_p_len'

            if f0_course_tensor is not None:
                f0_course_tensor = f0_course_tensor[:, :gen_p_len]
                f0_fine_tensor = f0_fine_tensor[:, :gen_p_len]

            # 3.5 Pitch Protection (if applicable)
            if feats_for_protection is not None and f0_fine_tensor is not None:
                 feats_for_protection = torch.nn.functional.interpolate(feats_for_protection.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                 # Ensure pitchf (f0_fine_tensor) has a value for every feature frame after interpolation
                 pitchff = f0_fine_tensor.clone() # This was pitchf in original
                 pitchff[f0_fine_tensor > 0] = 1
                 pitchff[f0_fine_tensor <= 0] = protect_val # Original logic was protect for <1
                 # Align protect mask with feature length
                 current_protect_mask = pitchff[:, :feats.shape[1]].unsqueeze(-1)
                 feats = feats * current_protect_mask + feats_for_protection[:, :feats.shape[1]] * (1 - current_protect_mask)
                 feats = feats.to(feats_for_protection.dtype)


            p_len_tensor = torch.tensor([gen_p_len], device=self.config.device).long()

            # 3.6 Generator Inference
            audio_out_tensor = self.net_g.infer(
                feats.float(), p_len_tensor,
                f0_course_tensor, f0_fine_tensor,
                self.sid_tensor
            )[0][0, 0]
            audio_out_numpy = audio_out_tensor.data.cpu().float().numpy()

        # 4. Trim the output (mimicking Pipeline's trim for short audio)
        # Output length of net_g.infer is related to input feature length * hop_size of vocoder
        # The original pipeline trims by self.t_pad_tgt.
        # Expected output length from net_g.infer for input `audio_padded` should be `len(audio_padded) * (self.tgt_sr / self.vc_pipeline.sample_rate)`
        # after vocoder's upsampling.
        # Then it's trimmed by `self.vc_pipeline.t_pad_tgt` on both sides.

        # If tgt_sr == 16000 (input SR), then t_pad_tgt == t_pad.
        # Output of infer is len(audio_padded_feats / 2) * vocoder_hop_length
        # This part is tricky. The original pipeline's trimming `[self.vc_pipeline.t_pad_tgt:-self.vc_pipeline.t_pad_tgt]`
        # is applied *after* the full audio (or a large segment) is processed.
        # For a small chunk, this implies the output of `net_g.infer` should be related to `len(audio_padded)`.
        # Let's assume the output of `net_g.infer` corresponds to `audio_padded` length but at `tgt_sr`.

        # If `self.tgt_sr` is different from `self.vc_pipeline.sample_rate` (16kHz),
        # the length of `audio_out_numpy` will be different from `len(audio_padded)`.
        # Vocoder upsamples by `self.tgt_sr / self.vc_pipeline.sample_rate` factor approximately.
        # `t_pad_tgt` is `self.tgt_sr * self.vc_pipeline.x_pad`
        # `t_pad` is `self.vc_pipeline.sample_rate * self.vc_pipeline.x_pad`

        # Let's assume audio_out_numpy corresponds to audio_padded but at tgt_sr.
        # Its length would be approx. len(audio_padded) * (self.tgt_sr / 16000)
        # The trim amount self.vc_pipeline.t_pad_tgt is correct for this tgt_sr audio.

        start_trim = self.vc_pipeline.t_pad_tgt
        end_trim = -self.vc_pipeline.t_pad_tgt

        if len(audio_out_numpy) > 2 * start_trim :
            audio_trimmed = audio_out_numpy[start_trim : end_trim]
        else: # Not enough audio to trim, maybe return empty or the small bit available
            print(f"Warning: Output audio ({len(audio_out_numpy)}) too short to trim by {start_trim}. Using untrimmed or silence.")
            # This case might indicate an issue with expected output length from net_g
            # For now, let's return what we have, or silence if it's critically short.
            # If len(audio_out_numpy) is very small, it might be better to return silence of expected length.
            # Expected length is len(audio_chunk_16khz_float32) * (self.tgt_sr / 16000)
            expected_out_len = int(len(audio_chunk_16khz_float32) * (self.tgt_sr / self.vc_pipeline.sample_rate))
            # audio_trimmed = audio_out_numpy # Use whatever came out if too short to trim
            if len(audio_out_numpy) < expected_out_len / 2 : # Heuristic: if less than half expected, something is wrong
                 audio_trimmed = np.zeros(expected_out_len, dtype=np.float32)
            else: # Otherwise, use the (possibly too short) output
                 audio_trimmed = audio_out_numpy


        # 5. Volume Envelope Matching (Optional)
        if output_volume_envelope_mix < 1.0 and output_volume_envelope_mix >= 0.0:
            # Resample original chunk to target SR for RMS comparison if needed
            if self.tgt_sr != self.vc_pipeline.sample_rate:
                original_chunk_resampled = librosa.resample(
                    audio_chunk_16khz_float32,
                    orig_sr=self.vc_pipeline.sample_rate,
                    target_sr=self.tgt_sr
                )
            else:
                original_chunk_resampled = audio_chunk_16khz_float32

            # Ensure audio_trimmed has same length as original_chunk_resampled for change_rms
            # This is tricky if lengths don't align perfectly.
            # For now, let's assume change_rms can handle it or we match lengths.
            target_len = len(original_chunk_resampled)
            if len(audio_trimmed) != target_len:
                # Pad or truncate audio_trimmed to match. Simple padding for now.
                if len(audio_trimmed) < target_len:
                    audio_trimmed = np.pad(audio_trimmed, (0, target_len - len(audio_trimmed)))
                else:
                    audio_trimmed = audio_trimmed[:target_len]

            audio_trimmed = self.vc_pipeline.AudioProcessor.change_rms(
                source_audio=original_chunk_resampled,
                source_rate=self.tgt_sr, # both are at tgt_sr now
                target_audio=audio_trimmed,
                target_rate=self.tgt_sr,
                rate=output_volume_envelope_mix
            )

        # 6. Final normalization (copied from Pipeline)
        audio_max = np.abs(audio_trimmed).max() / 0.99
        if audio_max > 1:
            audio_trimmed /= audio_max

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_trimmed.astype(np.float32)

    def __del__(self):
        if self.hubert_model: del self.hubert_model
        if self.net_g: del self.net_g
        if self.cpt: del self.cpt
        if self.index: del self.index
        if self.vc_pipeline: del self.vc_pipeline # vc_pipeline might also need a __del__
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("RealtimeVoiceConverter resources released.")

if __name__ == "__main__":
    # This is a placeholder for testing.
    # To test, you would need valid model paths, index paths, etc.
    # And then feed it audio chunks.
    print("RealtimeVoiceConverter module loaded. Standalone test requires model paths.")
    # rvc = RealtimeVoiceConverter()
    # model_p = "path/to/your/model.pth"
    # index_p = "path/to/your/index.index" # Optional
    # sid_val = 0
    # loaded, msg = rvc.load_resources(model_p, index_p, sid_val)
    # if loaded:
    #     print("Models loaded for test.")
    #     # dummy_chunk = np.random.randn(1024 * 16).astype(np.float32) # 1 sec at 16kHz
    #     # dummy_chunk = np.clip(dummy_chunk, -1.0, 1.0)
    #     # processed = rvc.process_chunk(dummy_chunk)
    #     # print(f"Processed chunk shape: {processed.shape}, dtype: {processed.dtype}")
    # else:
    #     print(f"Failed to load models for test: {msg}")

print("realtime_infer.py loaded")
