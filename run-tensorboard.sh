#!/bin/sh
printf "\033]0;Tensorboard\007"
. .venv/bin/activate

clear
python core.py tensorboard