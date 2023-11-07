@echo off
title Applio - Tensorboard

rem Conda env variables
set CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3
set INSTALL_ENV_DIR=%cd%\env

echo Starting the Tensorboard...
echo.
echo Recommended for Conda (Nvdia) users: 
echo [1] Start with Conda (customized python environment designed for the installation of required dependencies)
echo.
echo SOMETIMES NOT WORKING: Recommended for Runtime (Nvdia/AMD/Intel) users: 
echo [2] Start with Runtime (pre-installed dependencies)
echo.
echo Only recommended for experienced users:
echo [3] Local python
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls
echo Staring with conda...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
python lib/fixes/tensor-launch.py
pause
exit
)

if "%choice%"=="2" (
cls
echo Staring with runtime...
runtime\python.exe lib/fixes/tensor-launch.py
pause
exit
)


if "%choice%"=="3" (
cd %~dp0
cls
echo Staring with local python...
python lib/fixes/tensor-launch.py
pause
exit
)
