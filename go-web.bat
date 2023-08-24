@ECHO OFF
SETLOCAL

:: Set the Python command.
SET PYCMD="runtime\python.exe"

:: Set the port number.
SET PORT="7897"

:: Set the theme of Gradio.
SET THEME="JohnSmith9982/small_and_pretty"

:: Echo the current settings.
ECHO Current Settings:
ECHO.
ECHO Python command: %PYCMD%
ECHO Port number: %PORT%
ECHO Theme: %THEME%
ECHO.

:: Execute the Python script with the current settings.
%PYCMD% infer-web.py --pycmd %PYCMD% --port %PORT% --theme %THEME%

:: Pause the script at the end.
pause