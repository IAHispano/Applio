# Applio — Express.js + Next.js

The Python RVC engine (`rvc/`, `core.py`, torch/librosa/pedalboard stack) is kept as-is;
the UI layer is Next.js + Express:

```
Browser / Electron window
        │  http://localhost:3000 (Next.js App Router)
        ▼
Next.js (app/web) ──same-origin /api rewrite──▶ Express (app/api :8000)
                                                      │ spawn `python core.py <cmd> …`
                                                      ▼
                                              Python RVC engine (rvc/ + core.py)
```

- `app/api` is a thin orchestrator: zod validation → `core.py` CLI subprocess → job record
  (queued/running/done/error) → output file served under `/outputs`. No ML logic is
  reimplemented; flags map 1:1 to `core.py` click options.
- `app/web` is the UI: App Router pages for inference, training, TTS, voice blender,
  realtime, plugins, download, bug reports, extra tools, settings, TensorBoard.
  Shared `components/JobPanel.tsx` polls jobs, streams logs, renders audio/image/download outputs.
- `app/desktop` shells both `node` processes for macOS/Windows/Linux installers.
- `docker/` is the Colab/Kaggle path (same code, no Electron, GPU via a CUDA base image).

## Repo map

| Path                          | What                                                                                                                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/api/src/index.ts`        | Express app + WS proxy wiring                                                                                                                                                           |
| `app/api/src/schemas.ts`      | zod mirror of `core.py` inference options                                                                                                                                               |
| `app/api/src/routes/`         | one router per feature (inference, batch, presets, tts, blender, download, extra, train, settings, tensorboard, report, plugins, realtime, models, jobs)                                |
| `app/api/src/cli.ts`          | shared `core.py` job runner + PID-tracked stop                                                                                                                                          |
| `app/api/src/setup.ts`        | first-run setup engine (dependency checks + installer)                                                                                                                                  |
| `app/api/src/routes/setup.ts` | `GET /api/setup/status`, `POST /api/setup/install                                                                                                                                       | /prerequisites` |
| `app/api/src/python.ts`       | repo-root/python-bin resolution, traversal guards, spawn helper                                                                                                                         |
| `app/web/app/<page>/page.tsx` | UI pages                                                                                                                                                                                |
| `app/web/components/`         | `JobPanel`, `InferenceForm`, `BatchForm`, `PresetsPanel`                                                                                                                                |
| `app/web/lib/api.ts`          | typed API client                                                                                                                                                                        |
| `app/desktop/src/main.ts`     | Electron shell (dev: loads `/`; prod: spawns bundled API+web)                                                                                                                           |
| `core.py`                     | Python CLI (`infer`, `batch-infer`, `tts`, `preprocess`, `extract`, `train`, `index`, `model-blender`, `model-information`, `audio-analyzer`, `f0-curve`, `download`, `tensorboard`, …) |
| `rvc/`                        | Python RVC engine (inference, training, realtime)                                                                                                                                       |
| `docker/`                     | `Dockerfile` + `docker-compose.yml`                                                                                                                                                     |
| `tests/`                      | smoke suite (`npm test`)                                                                                                                                                                |
| `plugins/`                    | user-installed plugin packages                                                                                                                                                          |

## Run it

```bash
npm install
cp app/api/.env.example app/api/.env        # optional: pin PYTHON_BIN to an existing env
npm run dev                                  # api :8000 + web :3000
# open http://localhost:3000/  (setup screen checks everything, then continues)
```

## API (all endpoints)

| Method          | Route                                                   | Notes                                                                                           |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| GET             | `/api/health`, `/api/diagnostics`                       | liveness, python/torch versions                                                                 |
| GET             | `/api/models`                                           | `{models[], indexes[], audios[]}` repo-relative                                                 |
| GET             | `/api/setup/status`, `POST /api/setup/install           | /prerequisites`                                                                                 | first-run checks + installer jobs            |
| POST            | `/api/inference`                                        | multipart `audio` + fields in `schemas.ts`, or `inputPath`; → `202 {jobId}`                     |
| POST            | `/api/inference/batch`                                  | folder→folder batch convert                                                                     |
| GET/POST/DELETE | `/api/presets`, `GET /api/presets/formant`              | inference presets JSON CRUD                                                                     |
| GET             | `/api/jobs`, `/api/jobs/:id`, `POST /api/jobs/:id/stop` | poll every 2s; generic stop                                                                     |
| POST            | `/api/train/preprocess                                  | /extract                                                                                        | /train                                       | /index`                                      | full pipeline (auto-index after train) |
| GET             | `/api/train/datasets                                    | /pretraineds                                                                                    | /embedders                                   | /gpus                                        | /exports`                              | disk/GPU discovery       |
| POST            | `/api/train/upload-dataset                              | /upload-pretrained                                                                              | /upload-embedder`, `/api/train/stop`         | uploads + stop (job or legacy pid-file)      |
| GET/POST        | `/api/tts/voices`, `/api/tts`                           | EdgeTTS voices; text/.txt → TTS+RVC job                                                         |
| POST            | `/api/voice-blender`                                    | fuse two models → `logs/<name>.pth`                                                             |
| POST            | `/api/download`, `/api/download/drop`                   | link pipeline + .pth/.index drop → `logs/<model>/`                                              |
| GET/POST        | `/api/download/pretraineds`                             | pretrains.json cache + streamed G/D with % logs                                                 |
| POST            | `/api/extra/analyze                                     | /model-info                                                                                     | /f0`                                         | per-job plot paths (no fixed-path collision) |
| GET/PUT         | `/api/settings`, `/settings/languages                   | /version-check`                                                                                 | `assets/config.json` sections                |
| POST/GET        | `/api/tensorboard/start                                 | /stop                                                                                           | /status`                                     | `tensorboard --logdir logs`, iframe in UI    |
| GET/POST        | `/api/report/info                                       | /upload`                                                                                        | system info + screen-recording clip upload   |
| GET/POST        | `/api/plugins`, `/plugins/install                       | /toggle`                                                                                        | zip install + enable list (restart to apply) |
| POST/GET        | `/api/realtime/start                                    | /stop                                                                                           | /status                                      | /config                                      | /record`                               | uvicorn engine lifecycle |
| WS              | `/api/realtime/ws-audio`, `/api/realtime/change-config` | binary-safe proxy to engine (attached to HTTP server, since Next rewrites don't proxy upgrades) |

Deliberate server-side limits: OS auto-shutdown after training is not exposed;
output paths for TTS/analyze/F0 are server-assigned per job (no traversal risk);
realtime is single-session (engine-global voice instance).

## Colab / Kaggle

`assets/Applio.ipynb` / `assets/Applio_Kaggle.ipynb` install, build, start API+web in the
background and tunnel port 3000. Long training jobs: poll `GET /api/jobs/:id` rather than
blocking a cell; the server survives kernel reconnects when run with `nohup`.

## Desktop packaging

```bash
npm run build
npm run dist:win   # .exe lands in app/desktop/dist-installers/ (also dist:mac/dist:linux)
```

The installer is per-user (no admin rights): first launch opens the setup screen, which
provisions the Python env + engine packages into the install folder. torch/CUDA is
downloaded at setup time, not bundled.

## Tests

```bash
npm test   # builds the API, boots it on a test port, runs tests/smoke.ts (node:test, no deps)
```

Covers: health, setup status, models shape, presets CRUD, TTS voices, train discovery, settings
get/put round-trip, validation 400s, unknown-job 404, plus `core.py --help`.
