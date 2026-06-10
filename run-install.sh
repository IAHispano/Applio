#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

printf "\033]0;Installer\007"
clear

# Delete Windows bat files (.bat)
find . -type f -iname "*.bat" -delete

# Function to log messages with timestamps
log_message() {
    local msg="$1"
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - $msg${NC}"
}

log_error() {
    echo -e "${RED}[ERROR]$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_warn() {
    echo -e "${RED}[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

# Helper function for yes/no user prompt
confirm() {
    local prompt="${1:-Are you sure?}"
    read -p "$prompt [y/N]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Detect the best package manager for this system.
# Uses distro identification first, then falls back to available package managers.
# Outputs the package manager name; returns 1 if none found.
detect_package_manager() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|linuxmint|pop|elementary|zorin|kali|parrot|tails|deepin|devuan|mx|antix|bodhi|raspbian)
                command -v apt >/dev/null && { echo "apt"; return 0; }
                ;;
            arch|manjaro|endeavouros|garuda|artix|arco|archcraft|cachyos|chimeraos)
                command -v pacman >/dev/null && { echo "pacman"; return 0; }
                ;;
            fedora|rhel|centos|rocky|alma|ol|nobara|ultramarine|sangoma)
                command -v dnf >/dev/null && { echo "dnf"; return 0; }
                command -v yum >/dev/null && { echo "yum"; return 0; }
                ;;
            opensuse*|suse|sled|sles|gecko)
                command -v zypper >/dev/null && { echo "zypper"; return 0; }
                ;;
            gentoo|funtoo)
                command -v emerge >/dev/null && { echo "emerge"; return 0; }
                ;;
            alpine)
                command -v apk >/dev/null && { echo "apk"; return 0; }
                ;;
            void)
                command -v xbps-install >/dev/null && { echo "xbps"; return 0; }
                ;;
            nixos)
                command -v nix-env >/dev/null && { echo "nix"; return 0; }
                ;;
            solus|serpent)
                command -v eopkg >/dev/null && { echo "eopkg"; return 0; }
                ;;
            clear-linux*)
                command -v swupd >/dev/null && { echo "swupd"; return 0; }
                ;;
        esac
    fi

    # Fallback: try common package managers (order matters for systems where
    # multiple are present, e.g. an Arch user who also has apt installed)
    if command -v apt >/dev/null; then echo "apt"; return 0; fi
    if command -v dnf >/dev/null; then echo "dnf"; return 0; fi
    if command -v pacman >/dev/null; then echo "pacman"; return 0; fi
    if command -v zypper >/dev/null; then echo "zypper"; return 0; fi
    if command -v brew >/dev/null; then echo "brew"; return 0; fi
    if command -v emerge >/dev/null; then echo "emerge"; return 0; fi
    if command -v apk >/dev/null; then echo "apk"; return 0; fi
    if command -v xbps-install >/dev/null; then echo "xbps"; return 0; fi

    echo "unknown"
    return 1
}

# Check if C build tools are already available
has_build_tools() {
    command -v gcc >/dev/null 2>&1 && command -v make >/dev/null 2>&1
}

# Function to install build tools based on the distribution (required for wheels)
install_build_tools() {
    if has_build_tools; then
        log_message "Build tools (gcc, make) are already available."
        return 0
    fi

    log_message "Installing build tools..."
    local pm
    pm=$(detect_package_manager) || true

    case "$pm" in
        apt)
            log_message "Using apt..."
            sudo apt update && sudo apt install -y build-essential
            ;;
        pacman)
            log_message "Using pacman..."
            sudo pacman -Sy --noconfirm base-devel
            ;;
        dnf|yum)
            log_message "Using $pm..."
            sudo $pm group install -y development-tools --allowerasing 2>/dev/null || \
                sudo $pm install -y gcc gcc-c++ make
            ;;
        zypper)
            log_message "Using zypper..."
            sudo zypper install -y -t pattern devel_basis 2>/dev/null || \
                sudo zypper install -y gcc gcc-c++ make
            ;;
        apk)
            log_message "Using apk..."
            sudo apk add build-base
            ;;
        xbps)
            log_message "Using xbps..."
            sudo xbps-install -Sy base-devel
            ;;
        emerge)
            log_message "Gentoo detected — build tools should already be present."
            ;;
        brew)
            log_message "Using Homebrew..."
            brew install gcc make
            ;;
        eopkg)
            log_message "Using eopkg..."
            sudo eopkg install -y -c system.devel
            ;;
        swupd)
            log_message "Using swupd..."
            sudo swupd bundle-add c-basic
            ;;
        *)
            log_warn "Could not detect a supported package manager."
            log_warn "If pip fails to build wheels, install gcc and make manually and re-run."
            ;;
    esac
}

# Function to install FFmpeg based on the distribution
install_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        log_message "FFmpeg is already installed ($(ffmpeg -version 2>&1 | head -1))."
        return 0
    fi

    log_message "Installing FFmpeg..."
    local pm
    pm=$(detect_package_manager) || true

    case "$pm" in
        apt)
            sudo apt update && sudo apt install -y ffmpeg
            ;;
        pacman)
            sudo pacman -Syu --noconfirm ffmpeg
            ;;
        dnf|yum)
            sudo $pm install -y ffmpeg --allowerasing 2>/dev/null || \
                sudo $pm install -y ffmpeg-free --allowerasing 2>/dev/null || \
                install_ffmpeg_flatpak
            ;;
        zypper)
            sudo zypper install -y ffmpeg 2>/dev/null || \
                install_ffmpeg_flatpak
            ;;
        apk)
            sudo apk add ffmpeg
            ;;
        xbps)
            sudo xbps-install -y ffmpeg
            ;;
        emerge)
            sudo emerge -q media-video/ffmpeg
            ;;
        brew)
            log_message "Using Homebrew..."
            brew install ffmpeg
            ;;
        eopkg)
            sudo eopkg install -y ffmpeg
            ;;
        swupd)
            sudo swupd bundle-add ffmpeg
            ;;
        *)
            log_message "No native package manager for FFmpeg. Trying Flatpak..."
            install_ffmpeg_flatpak
            ;;
    esac
}

# Function to install FFmpeg using Flatpak
install_ffmpeg_flatpak() {
    if command -v flatpak >/dev/null; then
        log_message "Installing FFmpeg using Flatpak..."
        flatpak install --user -y flathub org.freedesktop.Platform.ffmpeg
    else
        log_message "Flatpak is not installed. Installing Flatpak..."
        local pm
        pm=$(detect_package_manager) || true
        case "$pm" in
            apt)        sudo apt update && sudo apt install -y flatpak ;;
            pacman)     sudo pacman -Syu --noconfirm flatpak ;;
            dnf|yum)    sudo $pm install -y flatpak ;;
            zypper)     sudo zypper install -y flatpak ;;
            apk)        sudo apk add flatpak ;;
            xbps)       sudo xbps-install -y flatpak ;;
            emerge)     sudo emerge -q sys-apps/flatpak ;;
            brew)       brew install flatpak ;;
            eopkg)      sudo eopkg install -y flatpak ;;
            *)
                log_error "Unable to install Flatpak automatically. Please install Flatpak and try again."
                exit 1
                ;;
        esac
        flatpak install --user -y flathub org.freedesktop.Platform.ffmpeg
        if ! command -v ffmpeg >/dev/null 2>&1; then
            log_warn "FFmpeg is still not available on PATH after Flatpak install. Please install FFmpeg using your system package manager."
        fi
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
    install_build_tools

    if ! command -v uv >/dev/null 2>&1; then
        log_message "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        hash -r 2>/dev/null || true
    fi

    log_message "Creating virtual environment..."
    uv venv .venv --python 3.12
    log_message "Activating virtual environment..."
    source .venv/bin/activate

    install_ffmpeg
    log_message "Installing python-ffmpeg..."
    uv pip install python-ffmpeg

    log_message "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        export UV_HTTP_TIMEOUT=300 # for slow internet
        if [ "$(uname)" = "Darwin" ]; then
            uv pip install -r requirements.txt
        else
            uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
        fi
    else
        log_message "requirements.txt not found. Please ensure it exists."
        exit 1
    fi

    finish
}

# Function to finish installation
finish() {
    log_message "##########"
    log_message "Applio has been successfully installed. Run the file run-applio.sh to start the web interface!"
    log_message "##########"
    exit 0
}

# Main script execution
if [ "$(uname)" = "Darwin" ]; then
    log_message "Detected macOS..."

    if ! command -v brew >/dev/null 2>&1; then
        log_message "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export PATH="$(brew --prefix)/bin:$PATH"
    brew install faiss
elif [ "$(uname)" != "Linux" ]; then
    log_message "Unsupported operating system. Are you using Windows?"
    log_message "If yes, use the batch (.bat) file instead of this one!"
    exit 1
fi

prepare_install
