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
  fi
}

# Check if Homebrew is in PATH
if ! command -v brew &> /dev/null; then
  # If not, check common paths for Homebrew
  echo "Homebrew not found in PATH. Checking common paths..."
  for path in "${BREW_PATHS[@]}"; do
    if [[ -x "$path/brew" ]]; then
      add_to_path "$path"
    fi
  done
fi

# Check again if Homebrew is in PATH
if ! command -v brew &> /dev/null; then
  echo "Homebrew still not found. Attempting to install..."
  INSTALL_OUTPUT=$(/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)")
fi

# Verifying if Homebrew has been installed successfully
if command -v brew &> /dev/null; then
  echo "Homebrew installed successfully."
else
  echo "Homebrew installation failed."
  exit 1
fi

# Extracting the commands to add Homebrew to the PATH
PATH_COMMANDS=$(echo "$INSTALL_OUTPUT" | awk '/Next steps:/,/Further documentation:/' | grep 'eval')

echo "Extracted commands to add Homebrew to the PATH:"

IFS=$'\n' # Set the Internal Field Separator to a new line
for cmd in $PATH_COMMANDS
do
    echo "$cmd"
done

# Asking the user if they want to execute them
echo "Do you want to automatically add Homebrew to your PATH by executing the commands above? (y/n)"
read -p "Are you sure you want to run these commands? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  IFS=$'\n' # Set the Internal Field Separator to a new line
  for cmd in $PATH_COMMANDS
  do
    eval "$cmd"
  done
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
  installed_packages=$(python3.8 -m pip freeze)
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

# Run the main script
python3.8 infer-web.py --pycmd python3.8
