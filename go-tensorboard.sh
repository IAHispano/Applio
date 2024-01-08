#!/bin/sh
printf "\033]0;Applio-RVC-Fork - Tensorboard\007"
. .venv/bin/activate
clear

echo "Starting the Tensorboard with Conda..."
clear
echo "Starting Tensorboard..."
pip install tensorboard
python lib/tools/tensorLaunch.py

printf "Press Enter to exit..." >&2
read -r ""
exit 0
