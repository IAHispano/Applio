# Third-party notices (uvr/)

## audio-separator engine (vendored under `uvr/audio_separator/`)

Source: https://github.com/nomadkaraoke/python-audio-separator

```
MIT License

Copyright (c) 2023 karaokenerds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Bundled architecture code (inside the vendored tree)

- `separator/uvr_lib_v5/demucs/` — Facebook Research Demucs (MIT).
- `separator/uvr_lib_v5/vr_network/`, `mdxnet.py`, `tfc_tdf_v3.py` — from UVR
  (Anjok0109 / Ultimate Vocal Remover, GPL-3.0 as released upstream). These files
  ship as part of the audio-separator package; keep their headers intact.

## Model weights (downloaded at runtime, NOT vendored)

- Community UVR models: https://github.com/TRvlvr/model_repo (various authors/licenses).
- "VIP" models are paywalled releases by Anjok07 — the engine warns on use; only
  separate with weights you are entitled to.
- Demucs weights: https://github.com/facebookresearch/demucs (MIT).
