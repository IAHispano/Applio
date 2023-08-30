@echo off
Title Applio Updater
setlocal

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
::: Version 1.0.0 - Developed by Aitron
:::

set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork/archive/refs/heads/main.zip"
set "principal=%cd%"
set "fixesFolder=Fixes"
set "localFixesPy=local_fixes.py"
set "subdir=temp_update"
set "colabmdx=colab_for_mdx.py"
echo.
cls

for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

echo Downloading the ZIP file...
powershell -command "& { Invoke-WebRequest -Uri '%repoUrl%' -OutFile '%principal%\repo.zip' }"
echo.

echo Extracting ZIP file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%principal%\repo.zip', '%principal%') }"
echo.

echo Copying folder and file structure from subdirectory to main directory...
robocopy "%principal%\Applio-RVC-Fork-main" "%principal%" /E
echo.

echo Deleting contents of the subdirectory (files and folders)...
rmdir "%principal%\Applio-RVC-Fork-main" /S /Q
echo.

echo Cleaning up...
del "%principal%\repo.zip"
echo.
cls

echo Verifying if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    python "%fixesFolder%\%localFixesPy%"
) else (
    echo The file "%localFixesBat%" was not found in the "Fixes" folder.
)

echo Applio has been updated!
endlocal
echo.
pause
exit
