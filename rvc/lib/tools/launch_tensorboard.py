"""Run-scoped TensorBoard servers; unrelated experiments never enter the view."""

import threading
from pathlib import Path

from tensorboard import program

log_path = "logs"
_tb_url = None
_servers = {}
_lock = threading.Lock()


def available_runs():
    root = Path(log_path)
    runs = []
    for project in root.iterdir() if root.exists() else []:
        if project.is_dir():
            events = list(project.rglob("events.out.tfevents.*"))
            if events:
                progress = list(project.glob("checkpoints/*/metrics.jsonl"))
                if (project / "train.log").exists():
                    progress.append(project / "train.log")
                # Dashboard-only audio imports must not change the default run.
                runs.append(
                    (
                        bool(progress),
                        max(p.stat().st_mtime for p in progress or events),
                        project.name,
                    )
                )
    return [name for _, _, name in sorted(runs, reverse=True)]


def project_path(run):
    root = Path(log_path).resolve()
    project = (root / run).resolve()
    if project.parent != root or not project.is_dir():
        raise ValueError("Select an existing training run inside logs.")
    return project


def dashboard_sources(project):
    """Readable stage labels, including earlier externally bridged history."""
    sources = {}
    staged = (project / "checkpoints").is_dir()
    for stage in ("predictor", "vocoder", "flow", "shortcut", "adapt"):
        native = project / "checkpoints" / stage / "tensorboard"
        history = project / "tensorboard" / stage
        if staged or native.exists():
            sources[stage.capitalize()] = native
        if history.exists():
            label = stage.capitalize() + (
                " (earlier updates)" if staged or native.exists() else ""
            )
            sources[label] = history
    listening = project / "tensorboard" / "listening"
    if staged or any(listening.glob("events.out.tfevents.*")):
        sources["Listening comparisons"] = listening
    return sources


def get_tb_url():
    return _tb_url


def _follow_audio(project):
    from rvc.train.process.v3_tensorboard import sync_evaluation_audio

    while True:
        # A partially written evaluation is retried after its report is complete.
        try:
            sync_evaluation_audio(project)
        except (OSError, ValueError):
            pass
        threading.Event().wait(15)


def launch_tensorboard(run=None):
    global _tb_url
    with _lock:
        try:
            runs = available_runs() if run is None else [run]
            if not runs:
                return "Error: No training events yet."
            project = project_path(runs[0])
            if project not in _servers:
                from rvc.train.process.v3_tensorboard import sync_evaluation_audio

                sync_evaluation_audio(project)
                tb = program.TensorBoard()
                sources = dashboard_sources(project)
                arguments = (
                    [
                        "--logdir_spec",
                        ",".join(f"{label}:{path}" for label, path in sources.items()),
                    ]
                    if sources
                    else ["--logdir", str(project)]
                )
                tb.configure(
                    argv=[
                        None,
                        *arguments,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--path_prefix",
                        "/tensorboard",
                        "--reload_interval",
                        "5",
                    ]
                )
                _servers[project] = (tb, tb.launch())
                threading.Thread(
                    target=_follow_audio, args=(project,), daemon=True
                ).start()
            _tb_url = _servers[project][1]
            return _tb_url
        except (OSError, ValueError, RuntimeError) as error:
            return f"Error: {error}"


def launch_tensorboard_pipeline():
    url = launch_tensorboard()
    print(f"TensorBoard running at: {url}#scalars")
    threading.Event().wait()
