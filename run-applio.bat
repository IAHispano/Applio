@echo off

net session >nul 2>&1
if %errorlevel% == 0 (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo If you want to disable User Account Control (UAC) temporarily, follow these steps:
    echo   1. Open the Control Panel.
    echo   2. Go to "Security and Maintenance" > "Security" > "Change User Account Control settings".
    echo   3. Move the slider down to "Never notify (disable UAC)".
    echo   4. Click "OK" and restart if prompted.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"

title %folder_name%

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

env\python.exe app.py --open
echo.
pause
