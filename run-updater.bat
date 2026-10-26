@echo off
setlocal DisableDelayedExpansion
title Applio Updater
cd /d "%~dp0"
if not exist "env\python.exe" (
    echo Bundled Python was not found. Run this updater from your extracted Applio folder.
    pause
    exit /b 1
)
rem Read the whole tail before Python replaces this running batch file.
rem CALL expands the exit code at execution time, after Python finishes.
(
    "%~dp0env\python.exe" -u "%~dp0assets\updater.py" %*
    call set "update_result=%%errorlevel%%"
    echo.
    pause
    call exit /b %%update_result%%
)
