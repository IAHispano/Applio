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

echo [1] Start Applio (Nvidia Support)
echo [2] Start Applio (AMD Support)
if exist "runtime\python.exe" (
    echo.
    echo ^[3^] Start Applio with Runtime ^(Nvidia Support^)
    echo ^[4^] Start Applio with Runtime ^(AMD Support^)
)
echo.
echo [5] Exit
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="5" (
    goto finish
) else if "%choice%"=="1" (
    cls
    python infer-web.py --pycmd python --port 7897
    pause
    cls
    goto menu
) else if "%choice%"=="2" (
    cls
    python infer-web.py --pycmd python --port 7897 --dml
    pause
    cls
    goto menu
) else if exist "runtime/python.exe" (
    if "%choice%"=="3" (
        cls
        runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897
        pause
        cls
        goto menu
    ) else if "%choice%"=="4" (
        cls
        runtime\python.exe infer-web.py --pycmd runtime/python.exe --port 7897 --dml
        pause
        cls
        goto menu
    )
)

cls
echo Invalid option. Please enter a number from 1 to 4.
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
:finish
