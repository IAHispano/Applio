#!/bin/bash

printf "\033]0;Applio Installer\007"
clear

echo "Welcome to the Applio Installer!"
echo

principal=$(pwd)
CONDA_ROOT_PREFIX="$HOME/Miniconda3"
INSTALL_ENV_DIR="$principal/env"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh"
CONDA_EXECUTABLE="$CONDA_ROOT_PREFIX/bin/conda"

echo "Cleaning up unnecessary files..."
rm -f Makefile Dockerfile docker-compose.yaml *.bat
echo "Cleanup complete."
echo

if [ ! -f "$CONDA_EXECUTABLE" ]; then
    echo "Miniconda not found. Starting download and installation..."
    echo "Downloading Miniconda..."
    wget -q -O miniconda.sh "$MINICONDA_DOWNLOAD_URL"
    if [ ! -f "miniconda.sh" ]; then
        echo "Download failed. Please check your internet connection and try again."
        exit 1
    fi

    echo "Installing Miniconda..."
    bash miniconda.sh -b -p "$CONDA_ROOT_PREFIX"
    if [ $? -ne 0 ]; then
        echo "Miniconda installation failed."
        exit 1
    fi
    rm miniconda.sh
    echo "Miniconda installation complete."
else
    echo "Miniconda already installed. Skipping installation."
fi
echo

echo "Creating Conda environment..."
"$CONDA_ROOT_PREFIX/bin/conda" create --no-shortcuts -y -k --prefix "$INSTALL_ENV_DIR" python=3.9
if [ $? -ne 0 ]; then exit 1; fi
echo "Conda environment created successfully."
echo

if [ -f "$INSTALL_ENV_DIR/bin/python" ]; then
    echo "Installing specific pip version..."
    "$INSTALL_ENV_DIR/bin/python" -m pip install "pip<24.1"
    if [ $? -ne 0 ]; then exit 1; fi
    echo "Pip installation complete."
    echo
fi

echo "Installing dependencies..."

source "$INSTALL_ENV_DIR/bin/activate" 
pip install --upgrade setuptools || exit 1
pip install -r "$principal/requirements.txt" || exit 1
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 || exit 1
conda deactivate
echo "Dependencies installation complete."
echo

echo "Applio has been installed successfully!"
echo "To start Applio, please run './run-applio.sh'."
echo

exit 0