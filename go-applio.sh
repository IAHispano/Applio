#!/bin/sh
printf "\033]0;Applio-RVC-Fork\007"
. .venv/bin/activate
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
    printf "Select an option: " >&2
    read -r choice
    case $choice in
    1)
      clear
      python infer-web.py --pycmd python --port 7897 --theme dark
      printf "Press Enter to continue..." >&2
      read -r ""
      ;;
   2)
      clear
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.venv/lib
      python -m sklearnex infer-web.py --pycmd python --port 7897 --theme dark
      printf "Press Enter to continue..." >&2
      read -r ""
      ;;
    3)
      finish
      ;;
    *)
      clear
      echo "Invalid option. Please enter a number from 1 to 3."
      echo ""
      printf "Press 'Enter' to access the main menu..." >&2
      read -r ""
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