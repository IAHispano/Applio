import os
import sys
import json
import requests

now_dir = os.getcwd()
sys.path.append(now_dir)

config_file = os.path.join(now_dir, "assets", "config.json")


def load_local_version():
    try:
        with open(config_file, "r", encoding="utf8") as file:
            config = json.load(file)
            return config["version"]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"Error loading local version: {error}")
        return None


def obtain_tag_name():
    url = "https://api.github.com/repos/IAHispano/Applio/releases/latest"
    session = requests.Session()

    try:
        response = session.get(url)
        response.raise_for_status()

        data = response.json()
        return data.get("tag_name")

    except requests.exceptions.RequestException as error:
        print(f"Error obtaining online version: {error}")
        return None


def compare_version():
    local_version = load_local_version()
    if not local_version:
        return "Local version could not be determined."

    online_version = obtain_tag_name()
    if not online_version:
        return "Online version could not be determined. Make sure you have an internet connection."

    elements_online_version = list(map(int, online_version.split(".")))
    elements_local_version = list(map(int, local_version.split(".")))

    for online, local in zip(elements_online_version, elements_local_version):
        if local < online:
            return f"Your local version {local_version} is older than the latest version {online_version}."

    if len(elements_online_version) > len(elements_local_version):
        return f"Your local version {local_version} is older than the latest version {online_version}."

    return f"Your local version {local_version} is the latest version."
