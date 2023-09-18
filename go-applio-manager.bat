@echo off
title Applio Manager

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

setlocal
set "branch=main"
set "runtime=runtime"
set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork.git"
set "fixesFolder=fixes"
set "localFixesPy=local_fixes.py"
set "principal=%cd%"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"

:menu
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A

echo [1] Reinstall Applio
echo [2] Update Applio
echo [3] Update Applio + Dependencies
echo.

set /p choice=Select an option: 
set choice=%choice: =%

if "%choice%"=="1" (
    cls
    echo Starting Applio Reinstaller...
    echo.
    goto reinstaller
    pause
    cls
    goto menu

)

if "%choice%"=="2" (
    cls
    echo Starting Applio Updater...
    echo.
    goto updater
    pause
    cls
    goto menu
)

if "%choice%"=="3" (
    cls
    echo Updating Applio + Dependencies...
    echo.
    goto updaterDependencies
    pause
    cls
    goto menu

)

cls
echo Invalid option. Please enter a number from 1 to 3.
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu

:reinstaller

echo WARNING: Remember to install Microsoft C++ Build Tools, Redistributable, Python, and Git before continuing.
echo.
echo Step-by-step guide: https://rentry.org/appliolocal
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
echo Python: https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe
echo.
pause
cls

echo %py_version% | findstr /C:"3.9.8" >nul
if %errorlevel% equ 0 (
    echo Python 3.9.8 is installed, continuing...
) else (
    echo Python 3.9.8 is not installed or not added to the path, exiting.
    echo Press Enter to exit
    pause
    exit
)

echo Cloning the repository...
git pull
cd %repoFolder%
echo.

echo Proceeding to download the models...
echo.

echo WARNING: At this point, it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
pause
cls

echo Downloading the "pretrained" folder...
cd "pretrained"
curl -LJO "%URL_BASE%/pretrained/D32k.pth"
curl -LJO "%URL_BASE%/pretrained/D40k.pth"
curl -LJO "%URL_BASE%/pretrained/D48k.pth"
curl -LJO "%URL_BASE%/pretrained/G32k.pth"
curl -LJO "%URL_BASE%/pretrained/G40k.pth"
curl -LJO "%URL_BASE%/pretrained/G48k.pth"
curl -LJO "%URL_BASE%/pretrained/f0D32k.pth"
curl -LJO "%URL_BASE%/pretrained/f0D40k.pth"
curl -LJO "%URL_BASE%/pretrained/f0D48k.pth"
curl -LJO "%URL_BASE%/pretrained/f0G32k.pth"
curl -LJO "%URL_BASE%/pretrained/f0G40k.pth"
curl -LJO "%URL_BASE%/pretrained/f0G48k.pth"
cd ".."
echo.
cls

echo Downloading the "pretrained_v2" folder...
cd "pretrained_v2"
curl -LJO "%URL_BASE%/pretrained_v2/D32k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/D40k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/D48k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/G32k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/G40k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/G48k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0D32k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0D40k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0D48k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0G32k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0G40k.pth"
curl -LJO "%URL_BASE%/pretrained_v2/f0G48k.pth"
cd ".."
echo.
cls

echo Downloading the "uvr5_weights" folder...
cd "uvr5_weights"
curl -LJO "%URL_BASE%/uvr5_weights/HP2_all_vocals.pth"
curl -LJO "%URL_BASE%/uvr5_weights/HP3_all_vocals.pth"
curl -LJO "%URL_BASE%/uvr5_weights/HP5_only_main_vocal.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoAggressive.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoDeReverb.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoNormal.pth"
cd ".."
echo.
cls

echo Downloading the rmvpe.pt file...
curl -LJO "%URL_BASE%/rmvpe.pt"
echo.
cls

echo Downloading the hubert_base.pt file...
curl -LJO "%URL_BASE%/hubert_base.pt"
echo.
cls

echo Downloading the ffmpeg.exe file...
curl -LJO "%URL_BASE%/ffmpeg.exe"
echo.
cls

echo Downloading the ffprobe.exe file...
curl -LJO "%URL_BASE%/ffprobe.exe"
echo.
cls

echo Downloading the runtime.zip file...
curl -LJO "%URL_EXTRA%/%runtime%.zip"
echo.
cls
echo Extracting the runtime.zip file, this might take a while...
powershell -Command "Expand-Archive -Path '%runtime%.zip' -DestinationPath '.'"
del %runtime%.zip
echo.
cls

echo Downloads completed!
echo.

echo Checking if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    runtime\python.exe "%fixesFolder%\%localFixesPy%"
) else (
    echo The "%localFixesPy%" file was not found in the "Fixes" folder.
)
echo.

echo Fixes Applied!
echo.

echo Applio has been reinstalled!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu


:updater

echo Updating the repo...
git pull
echo.

echo Verifying if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    runtime\python.exe "%fixesFolder%\%localFixesPy%"
) else (
    echo The file "%localFixesPy%" was not found in the "Fixes" folder.
)
echo.

echo Applio has been updated!
echo.
echo Press 'Enter' to access the main menu... 
pause>nul
cls
goto menu


:updaterDependencies

echo Updating the repo...
git pull
echo.

echo Installing dependencies...
pip install -r requirements.txt
echo.
pip uninstall torch torchvision torchaudio -y
echo.
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
echo.
pip install git+https://github.com/suno-ai/bark.git
echo.
cls
echo Dependencies installed!
echo.

echo Verifying if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    runtime\python.exe "%fixesFolder%\%localFixesPy%"
) else (
    echo The file "%localFixesPy%" was not found in the "Fixes" folder.
)
echo.

echo Applio has been updated!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
