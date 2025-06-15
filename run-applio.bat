@echo off

net session >nul 2>&1
if %errorlevel% == 0 (
    color 0C
    echo Please run this script as a regular user. Applio does not require administrator permissions.
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