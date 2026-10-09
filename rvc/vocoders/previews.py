"""Epoch previews for the rectified trainers, written the way the RVC trainer
writes them: the three-panel mel figure and both recordings in TensorBoard
(``eval/media``, newest 10 kept) and as files under ``validation_samples``."""

import os

from rvc.vocoders.mel import LogMel
from rvc.vocoders.messages import TENSORBOARD_VALIDATION_AUDIO_TAG
from rvc.vocoders.media import MediaLog, limit_audio_peak, log_validation_preview


class RectifiedPreviews:
    def __init__(self, run_dir: str, config: dict, resume_step: int, device):
        self.run_dir = run_dir
        self.data = config["data"]
        self.media = MediaLog(os.path.join(run_dir, "eval", "media"), resume_step=resume_step)
        self.mel = LogMel.from_config(self.data).to(device)

    def log(self, epoch, step, source, generated_audio=None, reference_audio=None,
            generated_mel=None, reference_mel=None, extra_audio=None):
        """Log one preview. Mels are taken from the audio when not given; a
        figure without audio is written when only mels are given.

        ``extra_audio`` maps a name to more [1, samples] clips for the same sample.
        """
        if generated_mel is None:
            generated_mel = self.mel(generated_audio.float().reshape(1, -1))
        if reference_mel is None:
            reference_mel = self.mel(reference_audio.float().reshape(1, -1))
        with self.media.writer(step) as writer:
            log_validation_preview(
                writer=writer,
                experiment_dir=self.run_dir,
                epoch=epoch,
                sample_index=0,
                global_step=step,
                sample_rate=self.data["sample_rate"],
                predicted_mel=generated_mel[0],
                target_mel=reference_mel[0],
                predicted_wave=None if generated_audio is None else generated_audio.reshape(-1),
                target_wave=None if reference_audio is None else reference_audio.reshape(-1),
                source=source,
                hop_length=self.data["hop_length"],
                mel_fmin=self.data["mel_fmin"],
                mel_fmax=self.data["mel_fmax"],
            )
            for name, audio in (extra_audio or {}).items():
                writer.add_audio(
                    TENSORBOARD_VALIDATION_AUDIO_TAG.format(sample="sample_00", kind=name),
                    limit_audio_peak(audio).unsqueeze(0),
                    step,
                    sample_rate=self.data["sample_rate"],
                )
