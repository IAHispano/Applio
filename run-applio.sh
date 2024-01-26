#!/bin/sh
printf "\033]0;Applio\007"
. .venv/bin/activate

clear
python app.py --open