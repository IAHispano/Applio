import time
import logging
from tensorboard import program

log_path = "logs"


def launch_tensorboard_pipeline():
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_path])
    url = tb.launch()

    print(
        f"Access the tensorboard using the following link:\n{url}?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fd%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Fkl%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Fmel%22%7D%5D"
    )

    while True:
        time.sleep(600)
