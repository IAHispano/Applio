#!/bin/bash
echo -e "\033]0;Applio - Start\007"

menu() {
  while true; do
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
    echo ""
    echo "[1] Start Applio (Nvidia Support)"
    echo "[2] Start Applio (AMD Support)"
    echo ""
    echo "[3] Exit"
    echo ""

    read -p "Select an option: " choice
    case $choice in
    1)
      clear
      python3.9 infer-web.py --pycmd python3.9 --port 7897 --theme dark
      read -p "Press Enter to continue..."
      ;;
    2)
      clear
      python3.9 infer-web.py --pycmd python3.9 --port 7897 --dml --theme dark
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
