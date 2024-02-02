import ast
import json
import os
import importlib
import requests


folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.dirname(folder)
folder = os.path.dirname(folder)
folder = os.path.join(folder, "assets", "themes")

import sys
sys.path.append(folder)
def get_class(filename):
    with open(filename, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            if 'class ' in line:
                found = line.split('class ')[1].split(':')[0].split('(')[0].strip()
                return found
                break
    return None

def get_list():

    themes_from_files = [
        os.path.splitext(name)[0]
        for root, _, files in os.walk(folder, topdown=False)
        for name in files
        if name.endswith(".py") and root == folder
    ]

    json_file_path = os.path.join(folder, "theme_list.json")

    try:
        with open(json_file_path, 'r') as json_file:
            themes_from_url = [item["id"] for item in json.load(json_file)]
    except FileNotFoundError:
        themes_from_url = []

    combined_themes = set(themes_from_files + themes_from_url)

    return list(combined_themes)

def select_theme(name):
    selected_file = name + ".py"
    full_path = os.path.join(folder, selected_file)
    if not os.path.exists(full_path):
        with open(os.path.join(folder, 'theme.json'), 'w') as json_file:
            json.dump({"file": None, "class": name}, json_file)
        print(f"Theme {name} successfully selected, restart applio.")
        return
    class_found = get_class(full_path)
    if class_found:
        with open(os.path.join(folder, 'theme.json'), 'w') as json_file:
            json.dump({"file": selected_file, "class": class_found}, json_file)
        print(f"Theme {name} successfully selected, restart applio.")
    else:
        print(f"Theme {name} was not found.")

def read_json():
    json_file_name = os.path.join(folder, 'theme.json')
    try:
        with open(json_file_name, 'r') as json_file:
            data = json.load(json_file)
            selected_file = data.get("file")
            class_name = data.get("class")
            if not selected_file == None and class_name:
                return class_name
            elif selected_file == None and class_name:
                return class_name
            else:
                return "ParityError/Interstellar"
    except:
        return "ParityError/Interstellar"

def load_json():
    json_file_name = os.path.join(folder, 'theme.json')
    try:
        with open(json_file_name, 'r') as json_file:
            data = json.load(json_file)
            selected_file = data.get("file")
            class_name = data.get("class")
            if not selected_file == None and class_name:
                module = importlib.import_module(selected_file[:-3])
                obtained_class = getattr(module, class_name)
                instance = obtained_class()
                print(f"Theme Loaded: {class_name}")
                return instance
            elif selected_file == None and class_name:
                return class_name
            else:
                print("The theme is incorrect.")
                return None
    except Exception as e:
        print(f"Error Loading: {str(e)}")
        return None