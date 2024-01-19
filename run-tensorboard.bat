@echo off
setlocal
title Tensorboard

env\python.exe core.py tensorboard
pause