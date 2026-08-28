import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES_DIR = Path(__file__).parent / "languages"
EXCLUDE = {".venv","env"}


def extract_i18n_strings(node):
    strings = []
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "i18n":
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                strings.append(arg.value)
    for child in ast.iter_child_nodes(node):
        strings.extend(extract_i18n_strings(child))
    return strings


def process_file(path: Path):
    try:
        code = path.read_text(encoding="utf8", errors="ignore")
    except Exception:
        return []
    if "i18n" not in code:
        return []
    try:
        tree = ast.parse(code, filename=str(path))
    except SyntaxError:
        return []
    found = extract_i18n_strings(tree)
    if found:
        print(path.relative_to(ROOT), len(found))
    return found


py_files = ROOT.rglob("*.py")
code_keys = set()

for py_file in py_files:
    if any(part in EXCLUDE for part in py_file.parts):
        continue
    code_keys.update(process_file(py_file))

print()
print("Total unique:", len(code_keys))

standard_file = LANGUAGES_DIR / "en_US.json"

with open(standard_file, "r", encoding="utf-8") as f:
    standard_data = json.load(f)

standard_keys = set(standard_data.keys())
unused_keys = standard_keys - code_keys
missing_keys = code_keys - standard_keys

print("Unused keys:", len(unused_keys))
for k in sorted(unused_keys):
    print("\t", k)

print("Missing keys:", len(missing_keys))
for k in sorted(missing_keys):
    print("\t", k)

if code_keys:
    new_data = {k: k for k in sorted(code_keys)}
    with open(standard_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
else:
    print("No keys found, skipping write to avoid wiping en_US.json")
