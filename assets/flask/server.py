import os
import socket
import subprocess
import time
import requests
import sys
import json

now_dir = os.getcwd()
sys.path.append(now_dir)
config_file = os.path.join(now_dir, "assets", "config.json")
env_path = os.path.join(now_dir, "env", "python.exe")

host = "localhost"
port = 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)


def start_flask():
    try:
        sock.connect((host, port))
        print(
            f"Something is listening on port {port}; Probably the Flask server is already running."
        )
        print("Trying to start it anyway")
        sock.close()
        requests.post("http://localhost:8000/shutdown")
        time.sleep(3)
        script_path = os.path.join(now_dir, "assets", "flask", "routes.py")
        try:
            subprocess.Popen(
                [env_path, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        except Exception as e:
            print(f"Failed to start the Flask server")
            print(e)
    except Exception as e:
        sock.close()
        script_path = os.path.join(now_dir, "assets", "flask", "routes.py")
        try:
            subprocess.Popen(
                [env_path, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        except Exception as e:
            print("Failed to start the Flask server")
            print(e)


def load_config_flask():
    with open(config_file, "r") as file:
        config = json.load(file)
        return config["flask_server"]


def save_config(value):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        config["flask_server"] = value
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)
