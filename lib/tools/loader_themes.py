import ast
import json
import os
import importlib
import logging
logger = logging.getLogger(__name__)

folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.dirname(folder)
folder = os.path.dirname(folder)
folder = os.path.join(folder, "assets", "themes")

import sys
sys.path.append(folder)

def get_class(file_name, class_name):
    with open(file_name, 'r') as file:
        content = file.read()
        syntax_tree = ast.parse(content)
        
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node

    return None

def get_list():
    themes_list = [
        os.path.splitext(name)[0]
        for root, _, files in os.walk(folder, topdown=False)
        for name in files
        if name.endswith(".py") and root == folder
    ]
    return themes_list

def select_theme(name):
    selected_file = name + ".py"
    class_name = name
    full_path = os.path.join(folder, selected_file)
    class_found = get_class(full_path, class_name)
    if class_found:
        with open(os.path.join(folder, 'theme.json'), 'w') as json_file:
            json.dump({"file": selected_file, "class": class_name}, json_file)
        logger.info(f"Theme {class_name} successfully selected, restart applio.")
    else:
        logger.warn(f"Theme {class_name} was not found.")

def read_json():
    json_file_name = os.path.join(folder, 'theme.json')
    try:
        with open(json_file_name, 'r') as json_file:
            data = json.load(json_file)
            selected_file = data.get("file")
            class_name = data.get("class")
            if selected_file and class_name:
                return class_name
            else:
                return ""
    except:
        return "applio"

def load_json():
    json_file_name = os.path.join(folder, 'theme.json')
    try:
        with open(json_file_name, 'r') as json_file:
            data = json.load(json_file)
            selected_file = data.get("file")
            class_name = data.get("class")
            if selected_file and class_name:
                module = importlib.import_module(selected_file[:-3])
                obtained_class = getattr(module, class_name)
                instance = obtained_class()
                logger.info(f"Theme Loaded: {class_name}")
                return instance
            else:
                logger.warn("The theme is incorrect.")
                return None
    except Exception as e:
        logger.warning(f"Error Loading: {str(e)}")
        return None
