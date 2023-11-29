#!/bin/bash
echo -e "\033]0;Applio-RVC-Fork - Tensorboard\007"
source .venv/bin/activate
clear

echo "Starting the Tensorboard with Conda..."
clear
echo "Starting Tensorboard..."
pip install tensorboard
python lib/fixes/tensor-launch.py

read -p "Press Enter to exit..."
exit 0
