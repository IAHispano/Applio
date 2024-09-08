#!/bin/sh

printf "\033]0;Applio\007"

echo "Checking for a suitable Python interpreter..."
if command -v python3.10 > /dev/null 2>&1; then
  py=$(which python3.10)
  echo "Using python3.10"
elif command -v python3 > /dev/null 2>&1; then
  py=$(which python3)
  echo "Using python3 (Note: Python 3.10 is recommended)"
elif command -v python > /dev/null 2>&1; then
  py=$(which python)
  echo "Using python (Note: Python 3.10 is recommended)"
else
  echo "Error: No suitable Python interpreter found. Please install Python 3.10 or a compatible version."
  exit 1
fi

echo "Recreating virtual environment..."
$py -m venv .venv

current_dir=$(pwd)
find ".venv" -type f -exec sed -i -e 's/\r$//' -e "s|/home/runner/work/Applio/Applio/|$current_dir/|g" -e "s|/.venv/bin/python|/.venv/bin/$(basename $py)|g" {} +
echo "Virtual environment paths fixed."

echo "Done."