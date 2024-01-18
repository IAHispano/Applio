@echo off
color 0A
setlocal
title Applio-RVC-Fork
cd %~dp0

set CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3
set INSTALL_ENV_DIR=%cd%\env

:::
:::                       _ _
:::     /\               | (_)
:::    /  \   _ __  _ __ | |_  ___
:::   / /\ \ | '_ \| '_ \| | |/ _ \
:::  / ____ \| |_) | |_) | | | (_) |
::: /_/    \_\ .__/| .__/|_|_|\___/
:::          | |   | |
:::          |_|   |_|
:::
:::


for /f "usebackq delims=" %%i in ("%cd%\assets\configs\version.txt") do (
    set "localVersion=%%i"
)
for /f %%i in ('powershell -command "(Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IAHispano/Applio-RVC-Fork/main/assets/configs/version.txt').Content"') do set "onlineVersion=%%i"

for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
powershell -command "if ('%localVersion%' -lt '%onlineVersion%') { exit 1 } else { exit 0 }"
if %errorlevel% equ 1 (
    echo You are currently using an outdated version %localVersion%
    echo.
    echo We're excited to announce that version %onlineVersion% is now available for download on https://github.com/IAHispano/Applio-RVC-Fork. 
    echo Upgrade now to access the latest features and improvements!
    echo.
    goto continue
) else (
    goto continue
)

:continue
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
echo Starting Applio-RVC-Fork with Conda default compatible with all platforms...
python infer-web.py --pycmd python --port 7897 --theme dark
pause
