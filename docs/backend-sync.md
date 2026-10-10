# Shared RVC engine

Applio is the source of RVC used by Gradio and Applio-App. The app embeds
this repository as a pinned Git submodule, loads its engine directly, and
bundles that version into releases. Backend fixes belong here. The app
adopts them by updating its reference rather than copying source.

`requirements-engine.txt` contains RVC dependencies. `requirements.txt`
includes it and adds Gradio dependencies. UVR code and dependencies stay
in Applio-App.

Source/config templates resolve relative to engine code. Weights and
outputs live in the working/data directory. `APPLIO_ROOT`, `APPLIO_CODE_ROOT`,
and `APPLIO_LOGS_DIR` select data, interface resources, and training storage.
Gradio sets `APPLIO_CONFIG_FILE` to connect engine settings to its controls.

CUDA graphs are opt-in for a supervisor able to replace a failed worker.
Native audio signals readiness after warmup and uses independent bounded
output/monitor queues.
