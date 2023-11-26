#!/bin/bash
echo -e "\033]0;Applio - Installer\007"
source .venv/bin/activate
clear
menu1() {
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
echo
echo "[1] Uninstall Applio"
echo "[2] Update Applio"
echo "[3] Update Applio + Dependencies"
echo "[4] Fix Tensorboard"
echo
read -p "Select an option:  " choice1

case $choice1 in
    1)
        pip uninstall -r assets/requirements/requirements-dml* -y
        pip uninstall -r assets/requirements/requirements-ipex* -y
        pip uninstall -r https://raw.githubusercontent.com/WorXeN/Retrieval-based-Voice-Conversion-WebUI/main/requirements-amd.txt -y
        pip uninstall -r assets/requirements/requirements-realtime-vc.txt -y
        cd .. && rm -rf *Applio*
        finish1
        ;;
    2)
        git pull
        finish1
        ;;
    3)
        git pull
        ./install_Applio.sh
        finish1
        ;;
    4)
        python3.9 -m pip uninstall tb-nightly tensorboardX tensorboard
        python3.9 -m pip install tensorboard
        cls
        echo Tensorboard re-installed correctly!
        read -p "Press Enter to access the main menu..."
        finish1
        ;;

    *)
        echo "Invalid option. Please enter a number from 1 to 4."
        echo ""
        read -p "Press Enter to access the main menu..."
        ;;
esac
done
}

# Finish this thing
finish1() {
  clear
  echo "Goodbye!"
}
# Loop to the main menu
menu1
