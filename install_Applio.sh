#!/bin/bash
echo -e "\033]0;Applio - Installer\007"
clear
echo " :::"
echo " :::                       _ _ "
echo " :::     /\               | (_) "
echo " :::    /  \   _ __  _ __ | |_  ___ "
echo " :::   / /\ \ | '_ \| '_ \| | |/ _ \ "
echo " :::  / ____ \| |_) | |_) | | | (_) | "
echo " ::: /_/    \_\ .__/| .__/|_|_|\___/ "
echo " :::          | |   | | "
echo " :::          |_|   |_| "
echo " ::: "
echo " ::: "

if [[ "$(uname)" == "Darwin" ]]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [[ "$(uname)" != "Linux" ]]; then
  echo "Unsupported operating system. Are you using Windows...?"
  echo "If yes use the batch (.bat) file insted this one!"
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Activate venv..."
  source .venv/bin/activate
else
  echo "Creating venv..."
  requirements_file="assets/requirements/requirements-applio.txt"
  # Check if Python is installed
  if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Attempting to install..."
    if [[ "$(uname)" == "Darwin" ]] && command -v brew &> /dev/null; then
      brew install python
    elif [[ "$(uname)" == "Linux" ]] && command -v apt-get &> /dev/null; then
      sudo apt-get update
      sudo apt-get install python
    else
      echo "Please install Python manually."
      exit 1
    fi
  fi
 

# Clone the repo for make this script usable with echo 1 | curl blabla https://script.sh
git clone https://github.com/IAHispano/Applio-RVC-Fork
cd Applio-RVC-Fork
python -m venv .venv
source .venv/bin/activate
chmod +x stftpitchshift
chmod +x *.sh
# maybe is needed idk
chmod +x ./lib/infer/infer_libs/stftpitchshift
python -m ensurepip

  
  # Check if required packages are installed and install them if notia
  # I will change this to make a  requirements with the applio changes 
  # And add a custom one for nvidia, ipx, amd support on linux and directml for the batch script
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
fi

clear
menu() {
  while true; do
  clear
echo
echo "Only recommended for experienced users:"
echo "[1] Nvidia graphics cards"
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
        echo "Before install this check https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md#rocm-support-for-amd-graphic-cards-linux-only"
        read -p "Press enter to continue"
        python -m pip install -r assets/requirements/requirements-amd.txt
        python -m pip uninstall torch torchvision torchaudio -y
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
        echo
        finish
        ;;
    3)
        echo 
        python -m pip install -r assets/requirements/requirements-ipex.txt
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

# Finish installation
finish() {
  python -m pip uninstall fairseq -y
  if [[ "$OSTYPE" == "darwin"* ]]; then
  python -m pip install https://github.com/soudabot/fairseq-build-whl/releases/download/3.11/fairseq-0.12.3-cp311-cp311-macosx_10_9_universal2.whl
  else
  python -m pip install https://github.com/soudabot/fairseq-build-whl/releases/download/3.11/fairseq-0.12.3-cp311-cp311-linux_x86_64.whl
  fi
  clear
  echo "Applio has been successfully downloaded, run the file go-applio.sh to run the web interface!"
  exit 0
}
# Loop to the main menu
menu
