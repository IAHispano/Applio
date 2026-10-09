# Vocoder training

Command-line trainer for two mel vocoders, each by its own recipe's spectral
losses and against the v3 discriminator (period and spectrogram branches,
UnivHD, SAN):

- `nsf-bigvgan`: BigVGAN's anti-aliased SnakeBeta blocks over HiFi-GAN's
  transposed upsamplers, with the NSF sine excitation at every stage.
- `nsf-hifigan`: OpenVPI SingingVocoders' NSF-HiFiGAN.

Both render a log mel plus f0 to 44.1 kHz audio.

## Requirements

Applio's environment (`requirements.txt`) has everything the trainer imports
but Triton, which is optional: with it, NSF-BigVGAN's activations run as fused
kernels on CUDA; without it, or on the CPU, they run as plain PyTorch, slower
and heavier on memory. NSF-HiFiGAN does not use it.

- Linux: nothing to install, the CUDA build of PyTorch brings `triton`.
- Windows: `pip install triton-windows`, in the release that matches the
  installed PyTorch.

## Data

A filelist with one clip per row:

```
path/to/clip.wav|path/to/clip.f0.npy
```

- audio at the config's sample rate (44.1 kHz), any format `soundfile` reads;
- f0 in Hz as a `.npy` at 100 frames per second, 0 where unvoiced.

An RVC `filelist.txt` (`audio|features|f0|f0_voiced|speaker`) works as it is:
the fourth field is read as the pitch. Relative paths start at `--data-root`.

## Train

```
python rvc/vocoders/train.py --vocoder nsf-bigvgan --name my_vocoder --filelist data/filelist.txt --epochs 100 --batch-size 8 --precision bf16
```

Everything goes to `logs/my_vocoder/vocoder/`:

- `config.json`, copied from `rvc/vocoders/configs/<vocoder>.json` on the
  first run; edit it there to change the model or the losses of that run;
- `G_<step>.pth` and `D_<step>.pth`, the checkpoints a rerun resumes from;
- `my_vocoder_vocoder_<epoch>e_<step>s.pth`, the export (EMA weights, weight
  norm folded, with the model's settings and its mel; `kind: rectified_vocoder`);
- `eval/`, TensorBoard scalars, and previews of one clip rendered from its
  real mel: `tensorboard --logdir logs/my_vocoder/vocoder/eval`.

`python rvc/vocoders/train.py --help` lists the remaining options (several
GPUs, starting from a pretrain).
