#!/bin/bash
echo -e "\033]0;Applio-RVC-Fork - Installer\007"
clear

cat << "EOF"
 :::                       _ _
 :::     /\               | (_)
 :::    /  \   _ __  _ __ | |_  ___
 :::   / /\ \ | '_ \| '_ \| | |/ _ \
 :::  / ____ \| |_) | |_) | | | (_) |
 ::: /_/    \_\ .__/| .__/|_|_|\___/
 :::          | |   | |
 :::          |_|   |_|
 :::
EOF

# Function to create or activate a virtual environment
prepare_install() {
        #Clone repo
        if [[ -n $(cat .git/config | grep Applio-RVC-Fork) ]]; then
        echo "Clonned repo detected!"
        else
        git clone https://github.com/IAHispano/Applio-RVC-Fork
        cd Applio-RVC-Fork
        fi      

    if [ -d ".venv" ]; then
        echo "Venv found. this implies Applio has been already installed or this is a broken install"
        read -p "Do you want to execute go-applio.sh? (Y/N): " r
        r=$(echo "$r" | tr '[:upper:]' '[:lower:]')
    if [[ "$r" == "y" ]]; then
        ./go-applio.sh && exit 1
    else
        echo "Ok! the installation will continue, good luck!"  
    fi
        source .venv/bin/activate
    else
        echo "Creating venv..."
        requirements_file="assets/requirements/requirements.txt"
        echo "Checking if python exists"
        if command -v python3 > /dev/null 2>&1; then
            py=$(which python3)
            echo "Using python3"
        else
        if [[ -n $(python --version | grep 3.) ]]; then
            py=$(which python)
            echo "Using python"
        else
                echo "Please install Python3 or 3.11 manually."
                exit 1
            fi
        fi
  
        $py -m venv .venv
        source .venv/bin/activate
        chmod +x stftpitchshift *.sh ./lib/modules/infer/stftpitchshift
        python -m ensurepip
      # Update pip within the virtual environment
        pip3 install --upgrade pip
    fi
}


main_menu() {
    while true; do
        clear
        echo
        echo "Only recommended for experienced users:"
        echo "[1] Nvidia graphics cards"
        echo "These options below are probably broken. use with precaution"
        echo "[2] AMD graphics cards" 
        echo "[3] Intel ARC graphics cards"
        echo

        read -p "Select the option according to your GPU: " choice

        case $choice in
            1)
                echo
                python -m pip install -r assets/requirements/requirements.txt
                python -m pip uninstall torch torchvision torchaudio -y
                python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
                echo
                finish
                ;;
            2)
                echo
                echo "Before installing this, check https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md#rocm-support-for-amd-graphic-cards-linux-only"
                read -p "Press Enter to continue"
                python -m pip install -r assets/requirements/requirements-dml.txt
                python -m pip uninstall torch torchvision torchaudio -y
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
                echo
                finish
                ;;
            3)
                echo
                python -m pip install -r assets/requirements/requirements-ipex.txt
                python -m pip install scikit-learn-intelex
                finish
                ;;
            *)
                echo "Invalid option. Please enter a number from 1 to 3."
                echo ""
                read -p "Press Enter to access the main menu..."
                ;;
        esac
    done
}

# Function to finish installation (this should install missing dependencies)
finish() {
    # Check if required packages are installed and install them if not
    if [ -f "${requirements_file}" ]; then
        installed_packages=$(python -m pip freeze)
        while IFS= read -r package; do
            [[ "${package}" =~ ^#.* ]] && continue
            package_name=$(echo "${package}" | sed 's/[<>=!].*//')
            if ! echo "${installed_packages}" | grep -q "${package_name}"; then
                echo "${package_name} not found. Attempting to install..."
                python -m pip install --upgrade "${package}"
            fi
        done < "${requirements_file}"
    else
        echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
        exit 1
    fi
    clear
    echo "Applio has been successfully downloaded, run the file go-applio.sh to run the web interface!"
    mv install_Applio.sh reinstall_applio.sh
    exit 0
}

# Loop to the main menu
if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
    brew install python
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    fi    
elif [[ "$(uname)" != "Linux" ]]; then
    echo "Unsupported operating system. Are you using Windows...?"
    echo "If yes, use the batch (.bat) file instead of this one!"
    exit 1
fi

prepare_install
main_menu

