@echo off
setlocal
title Applio Updater
cd /d "%~dp0"
if not exist "env\python.exe" (
    echo Bundled Python was not found. Run this updater from your extracted Applio folder.
    pause
    exit /b 1
)
"%~dp0env\python.exe" -u "%~dp0assets\updater.py" %*
set "update_result=%errorlevel%"
echo.
pause
exit /b %update_result%
