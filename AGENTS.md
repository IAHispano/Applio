# AGENTS.md

Notes for AI coding agents (and humans automating Applio) based on hands-on
verification on an actual NVIDIA Tesla T4 (16GB, sm_75/Turing).

## Headless / CLI usage

- `core.py infer` is the CLI entry point (also used by the official
  [No-UI Colab notebook](https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio_NoUI.ipynb)).
  It works fully headlessly with no code changes and is the reliable path if
  the Gradio GUI (`run-applio.sh` / `python app.py`) fails to start in your
  environment (see the PortAudio note in the README).
- `core.py infer` does **not** expose `--device`, `--dtype`, `--fp16`,
  `--half`, or `--precision` flags — none of these exist in `infer_parser`.
  GPU selection is automatic (`torch.cuda.is_available()`); there is no
  silent CPU fallback and no way to force CPU/GPU or a dtype via the CLI.

## Precision (fp16/bf16) — fp32 only, by design of the current code

- The inference pipeline (HuBERT embedder, `net_g` / HiFi-GAN vocoder, RMVPE,
  FCPE) is hardcoded to fp32 via explicit `.float()` calls, e.g.
  `rvc/infer/infer.py:72` and `:479`, and `rvc/infer/pipeline.py:326` and
  `:367`.
- This is not merely "fp16 unsupported" — it actively breaks if you try to
  force it. Casting an already-loaded model to fp16 externally (e.g.
  `vc.hubert_model.half()`, no repo code changes) still crashes, because
  `pipeline.py` casts the input tensor back to fp32 right before the model
  call. The crash reproduces deterministically at HuBERT's first conv1d
  layer:
  ```
  RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
  ```
- The *inference* pipeline has no bf16 code path either — it's all fp32 as
  described above. (Training is a separate story: `tabs/settings/sections/precision.py`,
  `rvc/train/train.py`, and `rvc/train/anyprecision_optimizer.py` do have real
  bf16 support gated on `torch.cuda.is_bf16_supported()`, but that's unrelated
  to the inference path this note covers.) So for inference specifically,
  bf16-capable GPUs (e.g. this T4, `is_bf16_supported()=True`) gain nothing
  and also hit no bf16-specific bugs — the only currently working inference
  precision is fp32.
- Net effect for T4/Turing and similar GPUs: no fp16/bf16 tensor-core
  acceleration is available through the *inference* codepath today; treat
  fp32 as the only supported inference precision until the hardcoded casts
  are refactored (tracked
  as a code-logic issue, out of scope for this doc-only note).

## Positive findings on T4

- `torch` is version- and index-pinned in `requirements.txt`
  (`torch==2.7.1+cu128` via `--extra-index-url .../whl/cu128`), so a fresh
  install matches the CUDA wheel's minimum driver requirement instead of
  silently requiring a newer driver than what's installed.
- GPU is auto-selected with no silent CPU fallback.
- HuBERT's attention implementation defaults to `sdpa` (no forced
  `flash_attention_2`), which runs correctly on T4 (sm_75).
- Model footprint is small (<200MB) and peak VRAM during inference measured
  at ~1.1GB on a 16GB T4 — no OOM risk in practice.
- int8 quantization (bitsandbytes), CUDA Graphs, and `torch.compile` are not
  wired into the inference path (0 matches in the repo); given the small
  model size the expected benefit is limited anyway.
