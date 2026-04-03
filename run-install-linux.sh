
#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

printf "\033]0;Installer\007"
clear
rm -f *.bat  

# Function to log messages with timestamps
log_message() {
    local msg="$1"
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - $msg${NC}"
}
log_error() {
    echo -e "${RED}[ERROR]$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}


# Helper function for yes/no user prompt
confirm() {
    local prompt="${1:-Are you sure?}"
    read -p "$prompt [y/N]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Function to install FFmpeg based on the distribution
install_ffmpeg() {
    if ! command -v ffmpeg > /dev/null; then
        log_message "Attempting to install FFmpeg..."
        if command -v apt > /dev/null; then
            log_message "Installing FFmpeg using apt..."
            sudo apt update && sudo apt install -y ffmpeg
        elif command -v pacman > /dev/null; then
            log_message "Installing FFmpeg using pacman..."
            sudo pacman -Sy --noconfirm ffmpeg
        elif command -v dnf > /dev/null; then
            log_message "Installing FFmpeg using dnf..."
            sudo dnf install -y ffmpeg --allowerasing
        else
            # try using flatpack
            if command -v flatpak > /dev/null; then
                log_message "Installing FFmpeg using Flatpak..."
                flatpak install --user -y flathub org.freedesktop.Platform.ffmpeg
            else
                log_error "Unsupported distribution for FFmpeg installation. Install FFmpeg and try again."
            fi
        fi
    fi
}

# Function to install build tools based on the distribution (reguired for wheel)
install_build_tools() {
    log_message "Attempting to install build tools..."
    if command -v apt > /dev/null; then
        log_message "Installing build-essential using apt..."
        sudo apt update && sudo apt install -y build-essential
    elif command -v pacman > /dev/null; then
        log_message "Installing base-devel using pacman..."
        sudo pacman -Sy --noconfirm base-devel
    elif command -v dnf > /dev/null; then
        log_message "Installing Development Tools using dnf..."
        sudo dnf groupinstall -y "Development Tools" --allowerasing
    else
        log_error "Unsupported distribution for build tools installation. Install build tools equivelant for your distribution and try again."
    fi
}

# Function to create or activate a virtual environment
prepare_install() {
    if [ -d ".venv" ]; then
        log_message "Virtual environment found. This implies Applio has been already installed or this is a broken install."
        if confirm "Do you want to execute the install script again?"; then
            log_message "Continuing with the installation."
            rm -rf .venv
            create_venv
        else
            chmod +x run-applio.sh
            ./run-applio.sh && exit 0
        fi
    else
        create_venv
    fi
}

# Function to create the virtual environment and install dependencies
create_venv() {
    log_message "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_message "Creating virtual environment..."
    uv venv .venv --python 3.11

    log_message "Activating virtual environment..."
    source .venv/bin/activate

    install_build_tools
    install_ffmpeg
    log_message "Installing python-ffmpeg..."
    uv pip install python-ffmpeg

    log_message "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        export UV_HTTP_TIMEOUT=300 # for slow internet
        uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
    else
        log_message "requirements.txt not found. Please ensure it exists."
        exit 1
    fi

    finish
}

# Function to finish installation
finish() {
    # clear
    log_message "##########"
    log_message "Applio has been successfully installed. Run the file run-applio.sh to start the web interface!"
    log_message "##########"
    exit 0
}

# Main script execution
if [ "$(uname)" = "Darwin" ]; then
    log_message "Unsupported operating system. Are you using macOS?"
    log_message "If yes, use the 'run-install.sh' file instead of this one!"
elif [ "$(uname)" != "Linux" ]; then
    log_message "Unsupported operating system. Are you using Windows?"
    log_message "If yes, use the batch (.bat) file instead of this one!"
    exit 1
fi

prepare_install
