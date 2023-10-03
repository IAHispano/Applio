@echo off
setlocal
title Applio - Start
cd %~dp0

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

:menu
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
echo Runtime: Recommended for regular users
echo [1] Start Applio - Runtime ^(Nvidia Support^)
echo [2] Start Applio - Runtime ^(Intel Support. Requires Nvidia runtime^)
echo [3] Start Applio - Runtime ^(AMD Support^)
echo.
echo Dependencies: Only recommended for experienced users
echo [4] Start Applio ^(Nvidia Support^)
echo [5] Start Applio ^(AMD Support^)
echo.
echo [6] Exit
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="6" (
    goto finish
) else if "%choice%"=="5" (
    cls
    echo Starting Applio with AMD support...
    python infer-web.py --pycmd python --port 7897 --dml --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="4" (
    cls
    echo Starting Applio with Nvidia support...
    python infer-web.py --pycmd python --port 7897 --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="3" (
    cls
    echo Starting Applio with runtime for AMD support ^(you must have it installed^)...
    runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897 --dml --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="2" (
    runtime\python.exe -m pip install scikit-learn-intelex
    cls
    echo Starting Applio with runtime for Intel CPU support ^(you must have Nvidia support installed^)...
    runtime\python.exe -m sklearnex infer-web.py --pycmd runtime/python.exe --port 7897 --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="1" (
    cls
    echo Starting Applio with runtime for Nvidia support ^(you must have it installed^)...
    runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897 --theme dark
    pause
    cls
    goto menu
)

cls
echo Invalid option. Please enter a number from 1 to 5.
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
:finish
