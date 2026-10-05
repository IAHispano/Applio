import logging
import threading
from tensorboard import program

log_path = "logs"
_tb_url = None
_tb_thread = None
_tb_ready = threading.Event()


def get_tb_url():
    return _tb_url


def launch_tensorboard_pipeline():
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_path, "--path_prefix", "/tensorboard"])
    url = tb.launch()
    print(f"TensorBoard running at: {url}#scalars")

    while True:
        import time

        time.sleep(600)


def launch_tensorboard():
    global _tb_url, _tb_thread, _tb_ready
    if _tb_thread is not None and _tb_thread.is_alive():
        _tb_ready.wait(timeout=10)
        return _tb_url
    _tb_ready.clear()
    _tb_thread = threading.Thread(target=_start_tb, daemon=True)
    _tb_thread.start()
    _tb_ready.wait(timeout=15)
    return _tb_url


def _start_tb():
    global _tb_url, _tb_ready
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_path, "--path_prefix", "/tensorboard"])
    try:
        _tb_url = tb.launch()
    except Exception as e:
        _tb_url = f"Error: {e}"
    finally:
        _tb_ready.set()
    if not _tb_url or _tb_url.startswith("Error"):
        return
    print(f"TensorBoard running at: {_tb_url}#scalars")
    while True:
        import time

        time.sleep(600)
