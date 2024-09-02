#!/bin/sh
printf "\033]0;Applio\007"

if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment not found. Please run the installer first."
  exit 1
fi

echo "Checking if python exists"
if command -v python3.10 > /dev/null 2>&1; then
  py=$(which python3.10)
  echo "Using python3.10"
else
  if python --version | grep -qE "3\.(7|8|9|10)\."; then
    py=$(which python)
    echo "Using python"
  else
    echo "Please install Python3 or 3.10 manually."
    exit 1
  fi
fi

PYTHON_EXECUTABLE="$py"
PYTHON_HOME=$(dirname "$PYTHON_EXECUTABLE")

CURRENT_HOME=$(grep "^home =" .venv/pyvenv.cfg | cut -d "=" -f 2 | xargs)
if [ "$CURRENT_HOME" != "$PYTHON_HOME" ]; then
  sed -i "s/home =.*/home = $PYTHON_HOME/" .venv/pyvenv.cfg
fi

. .venv/bin/activate

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

clear

python app.py --open