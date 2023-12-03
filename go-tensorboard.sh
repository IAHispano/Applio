#!/bin/bash
CONDA_ENV=".venv"
echo -e "\033]0;Applio-RVC-Fork - Tensorboard\007"
     
 if [ -d "$CONDA_ENV" ]; then
    source "$CONDA_ENV/bin/activate"
    echo "Activated Conda environment."
    else
    echo "Conda environment not found. Please ensure it's set up properly."
    exit 1
    fi


    clear
   
      
    if ! command -v tensorboard &> /dev/null; then
    echo "Installing Tensorboard..."
    pip install tensorboard || { echo "Tensorboard installation failed."; exit 1; }
    fi


 echo "Starting Tensorboard..."
 python lib/fixes/tensor-launch.py || { echo "Failed to start Tensorboard."; exit 1; }


 read -p "Press Enter to exit..."

 exit 0
