import ast
import json
from pathlib import Path
from collections import OrderedDict


def extract_i18n_strings(node):
    i18n_strings = []

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            if isinstance(arg, ast.Str):
                i18n_strings.append(arg.s)

    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings


def process_file(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        code = file.read()
        if "I18nAuto" in code:
            tree = ast.parse(code)
            i18n_strings = extract_i18n_strings(tree)
            print(file_path, len(i18n_strings))
            return i18n_strings
    return []


# Use pathlib for file handling
py_files = Path(".").rglob("*.py")

# Use a set to store unique strings
code_keys = set()

for py_file in py_files:
    strings = process_file(py_file)
    code_keys.update(strings)

print()
print("Total unique:", len(code_keys))

standard_file = "languages/en_US.json"
with open(standard_file, "r", encoding="utf-8") as file:
    standard_data = json.load(file, object_pairs_hook=OrderedDict)
standard_keys = set(standard_data.keys())

# Combine unused and missing keys sections
unused_keys = standard_keys - code_keys
missing_keys = code_keys - standard_keys

print("Unused keys:", len(unused_keys))
for unused_key in unused_keys:
    print("\t", unused_key)

print("Missing keys:", len(missing_keys))
for missing_key in missing_keys:
    print("\t", missing_key)

code_keys_dict = OrderedDict((s, s) for s in code_keys)

# Use context manager for writing back to the file
with open(standard_file, "w", encoding="utf-8") as file:
    json.dump(code_keys_dict, file, ensure_ascii=False, indent=4, sort_keys=True)
    file.write("\n")
