#!/bin/bash
echo -e "\033]0;Applio-RVC-Fork\007"
source .venv/bin/activate
menu() {
  while true; do
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
 [1] Start Applio (Nvidia/AMD Support)
 [2] Start Applio (Intel GPU/CPU) Probably broken
 [3] Exit
EOF
    read -p "Select an option: " choice
    case $choice in
    1)
      clear
      python infer-web.py --pycmd python --port 7897 --theme dark
      read -p "Press Enter to continue..."
      ;;
   2)
      clear
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.venv/lib
      python -m sklearnex infer-web.py --pycmd python --port 7897 --theme dark
      read -p "Press Enter to continue..."
      ;;
    3)
      finish
      ;;
    *)
      clear
      echo "Invalid option. Please enter a number from 1 to 3."
      echo ""
      read -n 1 -s -r -p "Press 'Enter' to access the main menu..."
      ;;
    esac
  done
}

finish() {
  clear
  echo "Exiting Applio..."
  exit 0
}

menu
