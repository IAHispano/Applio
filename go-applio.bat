@echo off
setlocal
title Applio - Start

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

:menu
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A

echo [1] Start Applio : If you have installed the dependencies
echo [2] Start Applio (DML)
echo.
echo [3] Start Applio (Runtime) : If you have used runtime
echo [4] Start Applio (Runtime DML)
echo.
echo If you don't know which one to use, try 1 or 3 and the one that doesn't give you errors is the correct one.
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="1" (
    cls
    echo WARNING: At this point, it's recommended to disable antivirus or firewall.
    echo.
    python infer-web.py --pycmd python --port 7897
    pause
    cls
    goto menu
)

if "%choice%"=="2" (
    cls
    echo WARNING: At this point, it's recommended to disable antivirus or firewall.
    echo.
    python infer-web.py --pycmd python --port 7897 --dml
    pause
    cls
    goto menu
)

if "%choice%"=="3" (
    cls
    echo WARNING: At this point, it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
    echo.
    runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897
    pause
    cls
    goto menu
)

if "%choice%"=="4" (
    cls
    echo WARNING: At this point, it's recommended to disable antivirus or firewall.
    echo.
    runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897 --dml
    pause
    cls
    goto menu
)

cls
echo Invalid option. Please enter a number from 1 to 4.
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
