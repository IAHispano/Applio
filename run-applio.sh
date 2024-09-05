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

echo "Checking if python exists"
if command -v python3.10 > /dev/null 2>&1; then
  PYTHON_EXECUTABLE=$(which python3.10)
  echo "Using python3.10"
elif command -v python3 > /dev/null 2>&1; then
  PYTHON_EXECUTABLE=$(which python3)
  echo "Using python3"
elif command -v python > /dev/null 2>&1; then
  PYTHON_EXECUTABLE=$(which python)
  echo "Using python"
else
  echo "Error: Unable to find a suitable Python version."
  exit 1
fi

PYTHON_HOME=$(dirname "$PYTHON_EXECUTABLE")

CURRENT_HOME=$(grep "^home =" .venv/pyvenv.cfg | cut -d "=" -f 2 | xargs)

if [ "$CURRENT_HOME" != "$PYTHON_HOME" ]; then
  sed -i "s|home =.*|home = $PYTHON_HOME|" .venv/pyvenv.cfg
  VENV_PATH=$(realpath .venv)
  find "$VENV_PATH/bin/" -type f -exec sed -i "0,/^VIRTUAL_ENV=/s|VIRTUAL_ENV=.*|VIRTUAL_ENV='$VENV_PATH'|" {} \;
else
  echo "Home path in .venv/pyvenv.cfg is already up-to-date"
fi

. .venv/bin/activate

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

clear

"$PYTHON_EXECUTABLE" app.py --open
