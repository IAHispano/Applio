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

echo Recommended for regular users:
echo [1] Start Applio with Runtime (Nvidia Support)
echo [2] Start Applio with Runtime (AMD Support)
echo.
echo Only recommended for experienced users:
echo [3] Start Applio (Nvidia Support)
echo [4] Start Applio (AMD Support)
echo.
echo [5] Exit
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="5" (
    goto finish
) else if "%choice%"=="4" (
    cls
    python infer-web.py --pycmd python --port 7897 --dml --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="3" (
    cls
    python infer-web.py --pycmd python --port 7897 --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="2" (
    cls
    runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897 --dml --theme dark
    pause
    cls
    goto menu
) else if "%choice%"=="1" (
    cls
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
