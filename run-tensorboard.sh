#!/bin/sh
printf "\033]0;Tensorboard\007"
. .venv/bin/activate
export PYTHONNOUSERSITE=1

clear
python core.py tensorboard