"""Validation previews: the mel comparison figure and the audio, written to
TensorBoard and as loose files under the run's log directory."""

import os
from contextlib import contextmanager

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rvc.vocoders.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_AUDIO_TAG,
    TENSORBOARD_VALIDATION_AXIS_X,
    TENSORBOARD_VALIDATION_AXIS_Y,
    TENSORBOARD_VALIDATION_DB_LABEL,
    TENSORBOARD_VALIDATION_DIFFERENCE_LABEL,
    TENSORBOARD_VALIDATION_FOOTER,
    TENSORBOARD_VALIDATION_MEL_TAG,
    TENSORBOARD_VALIDATION_MEL_TITLES,
    TENSORBOARD_VALIDATION_PREVIEW_DIR,
    TENSORBOARD_VALIDATION_SOURCE_TAG,
)

MATPLOTLIB_FLAG = False
#: Defaults for the validation preview figure.  Overridable per model through
#: ``validation_preview_dpi`` / ``_width`` / ``_height`` in the config JSON.
#:
#: The two knobs are not interchangeable.  ``figsize`` is in inches and dpi
#: converts to pixels, so raising **dpi** scales the whole figure -- panels and
#: text together -- and is what "make it sharper" means.  Raising **figsize**
#: at a fixed dpi gives the panels more room while the text stays the same
#: physical size, so labels shrink relative to the plot.  67 x 24in lands on
#: 1608px wide, which is the historical output.
VALIDATION_PREVIEW_DPI = 67
VALIDATION_PREVIEW_FIGSIZE = (24.0, 5.8)

#: Candidate ticks for the frequency axis, in Hz.  A mel axis is close to
#: logarithmic, so the evenly spaced ticks a linear axis wants would crowd the
#: bottom octaves into a few pixels and label almost nothing where the voice
#: actually lives.  Whatever falls outside the filterbank's range is dropped.
VALIDATION_PREVIEW_FREQUENCY_TICKS = (
    0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
)


#: Previews kept for TensorBoard, which by default shows 10 images and 10
#: clips per tag; anything older only made the event file bigger.
TENSORBOARD_MEDIA_KEEP = 10


class MediaLog:
    """TensorBoard run for the validation media: one event file per preview,
    pruned to the newest ``keep``.

    Event files are append-only, so previews written beside the scalars kept
    every one of them (~1 MB each) while TensorBoard displays only 10.
    """

    def __init__(self, log_dir, resume_step=0, keep=TENSORBOARD_MEDIA_KEEP):
        self.log_dir = log_dir
        self.keep = keep
        os.makedirs(log_dir, exist_ok=True)
        # Written by a previous run past the checkpoint this one resumes from.
        for path, step in self._files():
            if step > resume_step:
                os.remove(path)

    def _files(self):
        """``(path, step)`` of every preview file, oldest step first."""
        found = []
        for name in os.listdir(self.log_dir):
            stem, _, step = name.rpartition(".step")
            if "tfevents" in stem and step.isdigit():
                found.append((os.path.join(self.log_dir, name), int(step)))
        return sorted(found, key=lambda item: item[1])

    @contextmanager
    def writer(self, global_step):
        """A writer for one preview; the file is closed and pruned on exit."""
        writer = SummaryWriter(
            self.log_dir, filename_suffix=f".step{int(global_step)}"
        )
        try:
            yield writer
        finally:
            writer.close()
            for path, _ in self._files()[: -self.keep]:
                os.remove(path)


def limit_audio_peak(audio, max_peak=0.98):
    """Return a finite mono preview whose peak stays inside TensorBoard's range."""
    if torch.is_tensor(audio):
        preview = audio.detach().float().cpu()
        preview = torch.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        preview = preview.reshape(-1)
        if preview.numel() == 0:
            return preview
        peak = preview.abs().amax()
        if torch.isfinite(peak) and peak.item() > max_peak:
            preview = preview * (max_peak / peak)
        return preview.clamp(-max_peak, max_peak)

    preview = np.asarray(audio, dtype=np.float32)
    preview = np.nan_to_num(
        preview,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=True,
    ).reshape(-1)
    if preview.size == 0:
        return preview
    peak = float(np.max(np.abs(preview)))
    if np.isfinite(peak) and peak > max_peak:
        preview *= max_peak / peak
    return np.clip(preview, -max_peak, max_peak).astype(np.float32, copy=False)


def _mel_to_numpy(mel):
    if torch.is_tensor(mel):
        mel = mel.detach().float().cpu().numpy()
    mel = np.nan_to_num(np.asarray(mel, dtype=np.float32), copy=True)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError("Validation mel spectrograms must be two-dimensional.")
    return mel


def _mel_axis_positions(frequencies, mel_bins, mel_fmin, mel_fmax):
    """Where ``frequencies`` land on a mel filterbank's bin-index axis.

    The filterbank's ``mel_bins + 2`` band edges are evenly spaced *in mel*, so
    a frequency's position in bin-index space is linear in ``hz_to_mel``.  Bin 0
    is centred on edge 1 and the last bin on edge ``mel_bins``, which is where
    the ``mel_bins + 1`` scaling and the -1 offset come from.

    ``htk=False`` is not a preference: it is librosa's default, and so it is
    what ``librosa.filters.mel`` used to build the basis in ``mel_processing``.
    Reading the axis off a different mel scale than the data was binned with
    would swap one wrong answer for another.
    """
    low = librosa.hz_to_mel(float(mel_fmin), htk=False)
    high = librosa.hz_to_mel(float(mel_fmax), htk=False)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("The mel axis needs a positive frequency range.")
    mels = librosa.hz_to_mel(np.asarray(frequencies, dtype=float), htk=False)
    return (mels - low) / (high - low) * (mel_bins + 1) - 1.0


def _format_frequency(frequency):
    return f"{frequency:g}" if frequency < 1000.0 else f"{frequency / 1000:g}k"


def plot_validation_preview_to_figure(
    predicted_mel,
    target_mel,
    epoch,
    global_step,
    sample_index=0,
    sample_rate=22050,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
):
    """Create a dark TensorBoard-style three-panel validation report.

    ``hop_length`` sets the time axis and must match the config the mels were
    produced with -- the default is a fallback, not a good guess.  The same goes
    for ``mel_fmin`` / ``mel_fmax``, which set the frequency axis; ``mel_fmax``
    of ``None`` means Nyquist, matching the mel extraction.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    target_mel = _mel_to_numpy(target_mel)
    predicted_mel = _mel_to_numpy(predicted_mel)
    mel_bins = min(target_mel.shape[0], predicted_mel.shape[0])
    frames = min(target_mel.shape[1], predicted_mel.shape[1])
    if mel_bins <= 0 or frames <= 0:
        raise ValueError("Validation mel spectrograms must not be empty.")

    target_mel = target_mel[:mel_bins, :frames]
    predicted_mel = predicted_mel[:mel_bins, :frames]
    difference_mel = predicted_mel - target_mel

    shared_values = np.concatenate((target_mel.ravel(), predicted_mel.ravel()))
    shared_low = float(np.quantile(shared_values, 0.01))
    shared_high = float(np.quantile(shared_values, 0.99))
    if not np.isfinite(shared_low) or not np.isfinite(shared_high):
        shared_low, shared_high = 0.0, 1.0
    if shared_low >= shared_high:
        shared_high = shared_low + 1.0

    difference_limit = max(
        1e-5,
        float(np.quantile(np.abs(difference_mel), 0.99)),
    )
    time_axis = np.arange(frames, dtype=np.float32) * float(hop_length) / float(sample_rate)
    # The mel filterbank spans DC to Nyquist, not DC to the sample rate. Using
    # the sample rate as the extent labelled every frequency tick at twice its
    # true value.
    top_frequency = float(sample_rate) / 2.0 if mel_fmax is None else float(mel_fmax)
    # The vertical axis is bin index, and mel bins are not evenly spaced in Hz.
    # Stretching them onto a linear 0..Nyquist ruler -- which is what an extent
    # of ``(0, top_frequency)`` does -- put every label except the two endpoints
    # in the wrong place, and not by a little: at 128 bins over 32 kHz the
    # tick reading "4.0k" sits on bin 32, whose real centre is 0.94 kHz. A
    # deficit under 1 kHz reads as a defect at 4 kHz, which is a diagnosis this
    # axis has no business making. Keep the image in bin space and move the
    # ticks instead, so what the eye measures is the filterbank's own spacing.
    tick_positions = _mel_axis_positions(
        VALIDATION_PREVIEW_FREQUENCY_TICKS, mel_bins, mel_fmin, top_frequency
    )
    visible = (tick_positions >= -0.5) & (tick_positions <= mel_bins - 0.5)
    tick_positions = tick_positions[visible]
    tick_labels = [
        _format_frequency(frequency)
        for frequency, keep in zip(VALIDATION_PREVIEW_FREQUENCY_TICKS, visible)
        if keep
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=tuple(figsize) if figsize else VALIDATION_PREVIEW_FIGSIZE,
        dpi=float(dpi) if dpi else VALIDATION_PREVIEW_DPI,
        facecolor="#10161f",
        gridspec_kw={
            "left": 0.045,
            "right": 0.955,
            "wspace": 0.16,
            "bottom": 0.25,
            "top": 0.86,
        },
    )
    figure.patch.set_facecolor("#10161f")
    panels = (predicted_mel, target_mel, difference_mel)
    panel_cmaps = ("inferno", "inferno", "turbo")
    panel_norms = (
        (shared_low, shared_high),
        (shared_low, shared_high),
        (-difference_limit, difference_limit),
    )
    for axis, mel, title, cmap, (vmin, vmax) in zip(
        axes,
        panels,
        TENSORBOARD_VALIDATION_MEL_TITLES,
        panel_cmaps,
        panel_norms,
    ):
        axis.set_facecolor("#10161f")
        rendered = axis.imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # Half a bin past the first and last centres, so a data coordinate
            # of ``i`` lands exactly on the centre of bin ``i``.
            extent=(time_axis[0], time_axis[-1], -0.5, mel_bins - 0.5),
        )
        axis.set_title(title, color="#f4f4f4", fontsize=17, fontweight="bold", pad=12)
        axis.set_xlabel(TENSORBOARD_VALIDATION_AXIS_X, color="#f4f4f4", fontsize=11)
        axis.set_ylabel(TENSORBOARD_VALIDATION_AXIS_Y, color="#f4f4f4", fontsize=11)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)
        axis.tick_params(colors="#f4f4f4", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#c9cdd3")
        colorbar = figure.colorbar(
            rendered,
            ax=axis,
            orientation="horizontal",
            fraction=0.055,
            pad=0.12,
        )
        colorbar.set_label(
            TENSORBOARD_VALIDATION_DIFFERENCE_LABEL
            if title == TENSORBOARD_VALIDATION_MEL_TITLES[-1]
            else TENSORBOARD_VALIDATION_DB_LABEL,
            color="#d7dbe1",
            fontsize=8,
        )
        colorbar.ax.tick_params(colors="#d7dbe1", labelsize=8)
        colorbar.outline.set_edgecolor("#c9cdd3")

    figure.text(
        0.5,
        0.055,
        TENSORBOARD_VALIDATION_FOOTER.format(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            n_mels=int(mel_bins),
            epoch=int(epoch),
            step=int(global_step),
        ),
        color="#bfc5ce",
        fontsize=10,
        ha="center",
    )
    return figure


def _write_atomically(path, write):
    """Write ``path`` through a temporary file and a rename.

    A reader -- the GUI polls these files -- or an interrupted run then sees
    the old file or the new one, never a WAV whose header was not finalised.
    """
    temporary = f"{path}.tmp"
    write(temporary)
    try:
        os.replace(temporary, path)
    except OSError:
        # Windows refuses to replace a file another process has open, such as
        # the GUI playing the previous preview; fall back to writing in place.
        os.remove(temporary)
        write(path)


def log_validation_preview(
    writer,
    experiment_dir,
    epoch,
    sample_index,
    global_step,
    sample_rate,
    predicted_mel,
    target_mel,
    predicted_wave=None,
    target_wave=None,
    source=None,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
):
    """Save and log one organized validation preview with mel difference."""
    sample_stem = f"sample_{int(sample_index):02d}"
    preview_dir = os.path.join(
        str(experiment_dir),
        TENSORBOARD_VALIDATION_PREVIEW_DIR,
        f"epoch_{int(epoch):04d}",
    )
    mel_dir = os.path.join(preview_dir, "mel")
    audio_dir = os.path.join(preview_dir, "audio")
    os.makedirs(mel_dir, exist_ok=True)

    figure = plot_validation_preview_to_figure(
        predicted_mel=predicted_mel,
        target_mel=target_mel,
        epoch=epoch,
        global_step=global_step,
        sample_index=sample_index,
        sample_rate=sample_rate,
        hop_length=hop_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        dpi=dpi,
        figsize=figsize,
    )
    image_path = os.path.join(mel_dir, f"{sample_stem}.png")
    try:
        figure.canvas.draw()
        composite = np.asarray(
            figure.canvas.buffer_rgba(),
            dtype=np.uint8,
        )[..., :3].copy()
        # ``figure.dpi`` rather than the module default: the figure may have
        # been built at an overridden dpi, and saving at a different one would
        # write a PNG that does not match what TensorBoard received.
        _write_atomically(
            image_path,
            lambda target: figure.savefig(target, dpi=figure.dpi, format="png"),
        )

        if writer is not None:
            writer.add_image(
                TENSORBOARD_VALIDATION_MEL_TAG.format(sample=sample_stem),
                torch.from_numpy(composite).permute(2, 0, 1),
                global_step,
                dataformats="CHW",
            )
            if source:
                writer.add_text(
                    TENSORBOARD_VALIDATION_SOURCE_TAG.format(sample=sample_stem),
                    str(source),
                    global_step=global_step,
                )
    finally:
        plt.close(figure)

    if predicted_wave is None or target_wave is None:
        return image_path

    os.makedirs(audio_dir, exist_ok=True)
    predicted_wave = limit_audio_peak(predicted_wave)
    target_wave = limit_audio_peak(target_wave)
    audio_length = min(predicted_wave.numel(), target_wave.numel())
    if audio_length <= 0:
        return image_path
    predicted_wave = predicted_wave[:audio_length]
    target_wave = target_wave[:audio_length]

    generated_path = os.path.join(audio_dir, f"{sample_stem}_generated.wav")
    original_path = os.path.join(audio_dir, f"{sample_stem}_original.wav")
    for path, wave in ((generated_path, predicted_wave), (original_path, target_wave)):
        _write_atomically(
            path,
            lambda target, wave=wave: sf.write(
                target, wave.numpy(), int(sample_rate), subtype="PCM_16", format="WAV"
            ),
        )

    if writer is not None:
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"],
            ),
            predicted_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["original"],
            ),
            target_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )

    return image_path
