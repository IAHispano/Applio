#!/bin/bash

# Define common paths for Homebrew
BREW_PATHS=(
  "/usr/local/bin"
  "/opt/homebrew/bin"
)

if [[ "$(uname)" == "Darwin" ]]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [[ "$(uname)" != "Linux" ]]; then
  echo "Unsupported operating system."
  exit 1
fi

requirements_file="requirements.txt"

# Function to add a path to PATH
add_to_path() {
  echo "Homebrew found in $1, which is not in your PATH."
  read -p "Do you want to add this path to your PATH? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Adding $1 to PATH..."

    # Detect the shell and choose the right profile file
    local shell_profile
    if [[ $SHELL == *"/bash"* ]]; then
      shell_profile="$HOME/.bashrc"
      [[ ! -f "$shell_profile" ]] && shell_profile="$HOME/.bash_profile"
    elif [[ $SHELL == *"/zsh"* ]]; then
      shell_profile="$HOME/.zshrc"
    else
      echo "Unsupported shell. Please add the following line to your shell profile file manually:"
      echo "export PATH=\"$PATH:$1\""
      return
    fi

    # Add the export line to the shell profile file
    echo "export PATH=\"$PATH:$1\"" >> "$shell_profile"

    # Source the shell profile file
    source "$shell_profile"

    # Verify that the new PATH includes Homebrew
    if ! command -v brew &> /dev/null; then
      echo "Failed to add Homebrew to the PATH."
    fi
  fi
}

# Check if Homebrew is in PATH
if command -v brew &> /dev/null; then
  echo "Homebrew is already in your PATH."
else
  # If not, check common paths for Homebrew
  echo "Homebrew not found in PATH. Checking common paths..."
  for path in "${BREW_PATHS[@]}"; do
    if [[ -x "$path/brew" ]]; then
      add_to_path "$path"
      break
    fi
  done
fi

# Check again if Homebrew is in PATH
if ! command -v brew &> /dev/null; then
  echo "Homebrew still not found. Attempting to install..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Check again if Homebrew is in PATH
  if ! command -v brew &> /dev/null; then
    echo "Homebrew not found in PATH even after installation. Checking common paths again..."
    for path in "${BREW_PATHS[@]}"; do
      if [[ -x "$path/brew" ]]; then
        echo "Found post-install homebrew, adding to PATH...."
        add_to_path "$path"
        break
      fi
    done
  fi
fi

# Verifying if Homebrew has been installed successfully
if command -v brew &> /dev/null; then
  echo "Homebrew installed successfully."
else
  echo "Homebrew installation failed."
  exit 1
fi

# Installing ffmpeg with Homebrew
if [[ "$(uname)" == "Darwin" ]]; then
  echo "Installing ffmpeg..."
  brew install ffmpeg
fi

# Check if Python 3.8 is installed
if ! command -v python3.8 &> /dev/null; then
  echo "Python 3.8 not found. Attempting to install..."
  if [[ "$(uname)" == "Darwin" ]] && command -v brew &> /dev/null; then
    brew install python@3.8
  elif [[ "$(uname)" == "Linux" ]] && command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install python3.8
  else
    echo "Please install Python 3.8 manually."
    exit 1
  fi
fi

# Check if required packages are installed and install them if not
if [ -f "${requirements_file}" ]; then
  installed_packages=$(python3.8 -m pip list --format=freeze)
  while IFS= read -r package; do
    [[ "${package}" =~ ^#.* ]] && continue
    package_name=$(echo "${package}" | sed 's/[<>=!].*//')
    if ! echo "${installed_packages}" | grep -q "${package_name}"; then
      echo "${package_name} not found. Attempting to install..."
      python3.8 -m pip install --upgrade "${package}"
    fi
  done < "${requirements_file}"
else
  echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
  exit 1
fi

# Install onnxruntime package
echo "Installing onnxruntime..."
python3.8 -m pip install onnxruntime

download_if_not_exists() {
  local filename=$1
  local url=$2
  if [ ! -f "$filename" ]; then
    echo "$filename does not exist, downloading..."
    curl -# -L -o "$filename" "$url"
    echo "Download finished."
  else
    echo "$filename already exists."
  fi
}

# Check and download hubert_base.pt
download_if_not_exists "hubert_base.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

# Check and download rmvpe.pt
download_if_not_exists "rmvpe.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

# Run the main script
python3.8 infer-web.py --pycmd python3.8
