#!/bin/sh
printf "\033]0;Applio\007"

if ! dpkg -s python3.10-venv > /dev/null 2>&1; then
  echo "Installing python3.10-venv..."
  sudo apt-get update
  sudo apt-get install -y python3.10-venv
fi

if ! command -v python > /dev/null 2>&1 && ! command -v python3 > /dev/null 2>&1; then
  echo "Error: Python or Python 3 not found. Please install one of them."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment not found. Please run the installer first."
  exit 1
fi

venv_dir=".venv" 

if command -v python3.10 > /dev/null 2>&1; then
  python_exe=$(which python3.10)
  echo "Using python3.10"
elif command -v python3 > /dev/null 2>&1; then
  python_exe=$(which python3)
  echo "Using python3"
elif command -v python > /dev/null 2>&1; then
  python_exe=$(which python)
  echo "Using python"
else
  echo "Error: Unable to find a suitable Python version."
  exit 1
fi

python_home=$(dirname "$python_exe")
python_prefix=$(dirname "$python_home")
python_exec_prefix="$python_prefix"

sed -i 's/\r$//' "$venv_dir/pyvenv.cfg" 
sed -i "s|^home =.*|home = $python_home|" "$venv_dir/pyvenv.cfg"
sed -i "s|^base-prefix =.*|base-prefix = $python_prefix|" "$venv_dir/pyvenv.cfg"
sed -i "s|^base-exec-prefix =.*|base-exec-prefix = $python_exec_prefix|" "$venv_dir/pyvenv.cfg"
sed -i "s|^base-executable =.*|base-executable = $python_exe|" "$venv_dir/pyvenv.cfg"

current_dir=$(pwd)

find "$venv_dir" -type f -exec sed -i -e 's/\r$//' -e "s|/home/runner/work/Applio/Applio/|$current_dir/|g" -e "s|/.venv/bin/python|/.venv/bin/$(basename $python_exe)|g" {} +

echo "Virtual environment paths fixed." 