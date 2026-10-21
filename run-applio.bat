@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"

title %folder_name%

set "python=env\python.exe"
if not exist env if exist .venv\Scripts\python.exe set "python=.venv\Scripts\python.exe"
if not exist "%python%" (
    echo Please run 'run-install.bat' ^(or 'uv sync' for a development checkout^) first to set up the environment.
    pause
    exit /b 1
)

"%python%" app.py --open
echo.
pause
