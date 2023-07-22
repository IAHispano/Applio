import threading
import time
from tensorboard import program
import webbrowser

log_path = "logs"

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    webbrowser.open_new(url)
    print(f'Tensorboard can be accessed at: {url}')

    while True:
        time.sleep(600)  # Keep the main thread running