# Applio UVR — self-contained local stem separation

Distilled from the UVR / audio-separator / MSST ecosystem. No external package
imports from those libraries; only standard scientific Python dependencies
(see `requirements.txt`: torch, onnx stack, julius, rotary/beartype).

## Layout

```
uvr/
  separate.py      CLI entry (what the backend spawns per job).
  separator.py     Orchestrator: download, arch dispatch, stem naming.
  models.py        Catalog (VR/MDX/MDXC/Demucs/MSST-Roformer + custom_models.json).
  base.py          Shared mix prep, soundfile writing, downloads, devices.
  spectral.py      Spectral DSP helpers.
  stft.py device.py tfc_tdf_v3.py  Shared low-level helpers.
  vr.py            VR-architecture separator (.pth).
  mdx.py           MDX separator (.onnx via onnx2torch / ONNX Runtime).
  mdxc.py          MDXC separator (.ckpt + yaml via TFC-TDF).
  demucs.py        Demucs v4 separator (+ vendored `demucs_lib/` stack).
  roformer.py      MSST Roformer separator (.ckpt + yaml, see `roformer_lib/`).
  roformer_lib/
    *.py           Vendored model classes (mel/band, BS + derived variants).
    registry.py    Variant key -> class (+ `custom/` user architectures).
  demucs_lib/      Vendored Demucs v4 stack (torch-only).
  vr_net/          VR network definitions + band parameter JSONs.
  vr_model_data.json / mdx_model_data.json
                   Vendored UVR parameter tables (hash-keyed) + current-file entries.
```

## Notes

- Weights download on first use into `rvc/models/uvr/<model-key>/` and are
  reused afterwards. The catalog (`--list-models`) never imports torch.
- MDX parameters resolve through the vendored hash tables, extended with
  entries for the current release files (verified from each ONNX graph).
  Unknown files fail with a clear error instead of degrading silently.
- Roformer variants with their own architecture file (e.g. HyperACE v2)
  are first-class: add the module under `roformer_lib/custom/` (see its README)
  and reference it from a catalog entry.
