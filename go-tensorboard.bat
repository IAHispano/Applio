@echo off
title Applio-RVC-Fork - Tensorboard

rem Conda env variables
set CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3
set INSTALL_ENV_DIR=%cd%\env

echo Activating Conda environment...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"

if errorlevel 1 (
    echo Failed to activate Conda environment.
    pause
    exit /b 1
)

echo Starting the Tensorboard with Conda...
cls
echo Starting Tensorboard...
python lib\tools\tensorLaunch.py
pause
exit /b 0
