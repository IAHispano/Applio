# Shared Applio backend

Applio and Applio-App use the same Python engine: `core.py`, `rvc/`, and
`uvr/`. The desktop API and Gradio tabs are adapters around that engine.
`backend-sync.json` records the files and settings that must stay aligned.
Model weights, recordings, user settings, and UI assets are not synchronized.

This synchronization combines Applio main at `2586dddc` with Applio-App at
`c47771f0`. It carries Swift subharmonic repair, GPU pitch prediction,
split-audio timing, independent output/monitor queues, and worker readiness
from Applio; and realtime warmup, bounded queues, optional supervised CUDA
graphs, inference workers, ROCm handling, automatic pretrained downloads,
configurable storage, training fixes, and UVR separation from Applio-App.

## Checking changes

Run from either checkout, using any Python 3.11+ interpreter:

```sh
python scripts/check-backend-sync.py /path/to/the/other/checkout
```

The command compares Python syntax trees (allowing independent formatting),
other source files, engine dependencies in `requirements.txt`, and backend
defaults. Gradio is the only interface-specific dependency excluded. New
tracked engine modules must be added to the manifest. Existing `test_`
modules are not part of the production engine.

For future changes, fetch both repositories, merge engine changes using
their common Git ancestor, resolve overlapping behavior, and apply the
result to both. Run this check and exercise the affected entry points in
both repositories before committing. Copying one entire repository over
the other would discard independent UI work.

## Interface integration

The desktop API supplies `APPLIO_ROOT`, `APPLIO_CODE_ROOT`,
`APPLIO_CONFIG_DIR`, and `APPLIO_LOGS_DIR` to its engine processes. Standalone
engine commands use per-user settings by default. Gradio supplies
`APPLIO_CONFIG_FILE` at startup so its existing settings controls and child
processes use the same repository settings file. UI theme, version, and
layout remain specific to each interface.

CUDA graphs remain opt-in through `APPLIO_ENABLE_CUDA_GRAPHS=1`, for a
supervisor that can replace a worker process after a capture failure. The
native audio worker uses eager inference by default and signals readiness
after warmup. Output and monitor consume independent, bounded queues.

Python subprocesses and model functions retain their existing Gradio call
signatures. Engine imports stay lazy, so importing `core` and requesting
CLI help do not load torch or download models. Optional separation models
are downloaded only when that backend is invoked.
