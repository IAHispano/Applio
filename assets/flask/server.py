import os
import socket
import subprocess
import time
import requests
import json

# Constants
NOW_DIR = os.getcwd()
CONFIG_FILE = os.path.join(NOW_DIR, "assets", "config.json")
ENV_PATH = os.path.join(NOW_DIR, "env", "python.exe")
FLASK_SCRIPT_PATH = os.path.join(NOW_DIR, "assets", "flask", "routes.py")
HOST = "localhost"
PORT = 8000
TIMEOUT = 2


# Functions
def start_flask():
    """
    Starts the Flask server if it's not already running.
    """
    try:
        # Check if Flask server is already running
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(TIMEOUT)
            sock.connect((HOST, PORT))
            print("Flask server is already running. Trying to restart it.")
            requests.post("http://localhost:8000/shutdown")
            time.sleep(3)

    except socket.timeout:
        # Start the Flask server
        try:
            subprocess.Popen(
                [ENV_PATH, FLASK_SCRIPT_PATH],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except Exception as error:
            print(f"An error occurred starting the Flask server: {error}")


def load_config_flask():
    """
    Loads the Flask server configuration from the config.json file.
    """
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
        return config["flask_server"]


def save_config(value):
    """
    Saves the Flask server configuration to the config.json file.
    """
    with open(CONFIG_FILE, "r", encoding="utf8") as file:
        config = json.load(file)
        config["flask_server"] = value
    with open(CONFIG_FILE, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)
