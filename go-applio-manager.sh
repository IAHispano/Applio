#!/bin/bash
echo -e "\033]0;Applio-RVC-Fork - Manager\007"
source .venv/bin/activate
clear
menu1() {
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
 [1] Update Applio
 [2] Update Applio + Dependencies
 [3] Fix Tensorboard
 [4] Exit
EOF
read -p "Select an option:  " choice1

case $choice1 in
    1)
        git pull
        read -p "Applio has been updated or its already in the latest version"
        finish1
        ;;
    2)
        git pull
        ./reinstall_applio.sh
        read -p "Applio has been updated or its already in the latest version"
        finish1
        ;;
    3)
        python3.9 -m pip uninstall tb-nightly tensorboardX tensorboard
        python3.9 -m pip install tensorboard
        cls
        echo Tensorboard re-installed correctly!
        read -p "Press Enter to access the main menu..."
        finish1
        ;;
     4) 
       echo "Exiting Applio..." 
       exit 0
       ;;

    *)
        echo "Invalid option. Please enter a number from 1 to 4."
        echo ""
        read -p "Press Enter to access the main menu..."
        ;;
esac
done
}

finish1() {
  chmod +x *.sh
  clear
  echo "Goodbye!"
}
menu1

