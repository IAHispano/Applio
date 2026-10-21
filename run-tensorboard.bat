@echo off
setlocal
title Tensorboard

set "python=env\python.exe"
if not exist env if exist .venv\Scripts\python.exe set "python=.venv\Scripts\python.exe"
if not exist "%python%" (
    echo Please run 'run-install.bat' ^(or 'uv sync' for a development checkout^) first to set up the environment.
    pause
    exit /b 1
)

"%python%" core.py tensorboard
echo.
pause
