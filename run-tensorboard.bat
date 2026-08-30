@echo off
setlocal
title Tensorboard

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

:: keep another project's per-user packages from shadowing this env
set "PYTHONNOUSERSITE=1"

env\python.exe core.py tensorboard
echo.
pause
