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
  echo "Unsupported operating system. Are you using windows or something?"
  echo "If yes use the batch (.bat) file insted this one"
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Activate venv..."
  source .venv/bin/activate
else
  echo "Create venv..."
  requirements_file="assets/requirements/requirements.txt"
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
  python3 -m venv .venv
  source .venv/bin/activate

# Clone the repo for make this script usable with echo 1 | curl blabla https://script.sh
git clone https://github.com/IAHispano/Applio-RVC-Fork
cd Applio-RVC-Forko
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
echo
read -p "Select the option according to your GPU: " choice

case $choice in
    1)
        echo "Disabled until split requirements.txt in applio-requirements.txt and requirements-nvidia.txt"
        finish
        python -m pip install -r assets/requirements/requirements-nvidia.txt
        echo
        python -m pip uninstall torch torchvision torchaudio -y
        echo
        python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
        echo
        ;;
    2)
        echo "Disabled until merge amd support"
        
        echo
        finish
        ;;
    3)
        echo "Disabled until merge ipx support"

        echo
        finish
        ;;
    *)
        echo "Invalid option. Please enter a number from 1 to 2."
        echo ""
        read -p "Press Enter to access the main menu..."
        ;;
esac
done
}

# Finish installation
finish() {
  clear
  echo "Applio has been successfully downloaded, run the file go-applio.sh to run the web interface!"
  exit 0
}
# Loop to the main menu
menu
