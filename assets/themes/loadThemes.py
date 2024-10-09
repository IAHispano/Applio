import json
import os
import sys
import importlib
import gradio as gr


# Setting up the paths
base_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
themes_folder = os.path.join(base_folder, "assets", "themes")
config_file_path = os.path.join(os.getcwd(), "assets", "config.json")

sys.path.append(themes_folder)


def get_class(filename):
    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if line.startswith("class "):
                return line.split("class ")[1].split(":")[0].split("(")[0].strip()
    return None


def get_list():
    themes_from_files = [
        os.path.splitext(name)[0]
        for root, _, files in os.walk(themes_folder, topdown=False)
        for name in files
        if name.endswith(".py") and root == themes_folder
    ]

    json_file_path = os.path.join(themes_folder, "theme_list.json")

    try:
        with open(json_file_path, "r", encoding="utf8") as json_file:
            themes_from_url = [item["id"] for item in json.load(json_file)]
    except FileNotFoundError:
        themes_from_url = []

    return list(set(themes_from_files + themes_from_url))


def select_theme(name):
    selected_file = f"{name}.py"
    full_path = os.path.join(themes_folder, selected_file)

    try:
        with open(config_file_path, "r", encoding="utf8") as json_file:
            config_data = json.load(json_file)

        if not os.path.exists(full_path):
            config_data["theme"]["file"] = None
            config_data["theme"]["class"] = name
        else:
            class_name = get_class(full_path)
            if class_name:
                config_data["theme"]["file"] = selected_file
                config_data["theme"]["class"] = class_name
            else:
                print(f"Theme {name} was not found.")
                return

        with open(config_file_path, "w", encoding="utf8") as json_file:
            json.dump(config_data, json_file, indent=2)

        print(f"Theme {name} successfully selected, restart Applio.")
        gr.Info(f"Theme {name} successfully selected, restart Applio.")
    except Exception as error:
        print(f"An error occurred: {error}")


def read_json():
    try:
        with open(config_file_path, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
            return data["theme"].get("class", "ParityError/Interstellar")
    except Exception as error:
        print(f"An error occurred loading the theme: {error}")
        return "ParityError/Interstellar"


def load_json():
    try:
        with open(config_file_path, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
            selected_file = data["theme"].get("file")
            class_name = data["theme"].get("class")

            if selected_file and class_name:
                module = importlib.import_module(selected_file[:-3])
                obtained_class = getattr(module, class_name)
                instance = obtained_class()
                print(f"Theme {class_name} successfully loaded.")
                return instance
            elif class_name:
                return class_name
            else:
                print("The theme is incorrect.")
                return None
    except Exception as error:
        print(f"An error occurred loading the theme: {error}")
        return None
