@echo off
setlocal
title Applio

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

set HIP_VISIBLE_DEVICES="0"
zluda\zluda.exe -- env\python.exe app.py --open
echo.
pause