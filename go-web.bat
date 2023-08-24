@ECHO OFF
SETLOCAL

:: Set the Python command.
SET PYCMD="runtime\python.exe"

:: Set the port number.
SET PORT="7897"

:: Set the theme of Gradio.
:: You can get more themes at https://huggingface.co/spaces/gradio/theme-gallery
:: For example if you want this one: https://huggingface.co/spaces/bethecloud/storj_theme
:: You will have to look at a line that starts like "To use this theme, set"
:: On the same line look for [" theme='[AUTHOR]/[THEME]' "]. e.g. [" theme='bethecloud/storj_theme' "]
:: Copy just the part in apostrophes: ''. e.g. bethecloud/storj_theme
:: Now modify the line below and paste that part with replacement in quotation mark. e.g. "bethecloud/storj_theme"
:: In the end you should have `SET THEME="bethecloud/storj_theme"`
SET THEME="gradio/soft"

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