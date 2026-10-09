DISCRIMINATOR_COMPILE_NOT_SUPPORTED = (
    "Discriminator compilation ignored: the selected vocoder does not "
    "support it."
)
DISCRIMINATOR_COMPILE_NO_CUDA = (
    "Discriminator compilation ignored: CUDA is unavailable."
)
DISCRIMINATOR_COMPILE_ENABLED = (
    "Discriminator compilation enabled with mode '{mode}'. "
    "The first training batch builds the graph."
)
DISCRIMINATOR_COMPILE_ENABLE_FAILED = (
    "Discriminator compilation could not be enabled:"
)
DISCRIMINATOR_COMPILE_RUNTIME_FAILED = (
    "Discriminator compilation failed; continuing in eager mode:"
)

TENSORBOARD_VALIDATION_PREVIEW_DIR = "validation_samples"
TENSORBOARD_VALIDATION_MEL_TITLES = (
    "Mel Generated",
    "Mel Original",
    "Difference",
)
TENSORBOARD_VALIDATION_FOOTER = (
    "Sample rate: {sample_rate} Hz   |   Hop length: {hop_length}   |   n_mels: {n_mels}"
    "   |   Epoch: {epoch}   |   Global Step: {step}"
)
TENSORBOARD_VALIDATION_DIFFERENCE_LABEL = "(Mel Generated − Mel Original) in dB"
TENSORBOARD_VALIDATION_AXIS_X = "Time (s)"
TENSORBOARD_VALIDATION_AXIS_Y = "Frequency (Hz)"
TENSORBOARD_VALIDATION_DB_LABEL = "dB"
TENSORBOARD_VALIDATION_MEL_TAG = "validation_previews/mel/{sample}"
TENSORBOARD_VALIDATION_AUDIO_TAG = "validation_previews/audio/{sample}/{kind}"
TENSORBOARD_VALIDATION_SOURCE_TAG = "validation_previews/source/{sample}"
TENSORBOARD_VALIDATION_AUDIO_NAMES = {
    "generated": "generated",
    "original": "original",
}
