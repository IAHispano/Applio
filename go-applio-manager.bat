@echo off
title Applio - Manager

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
set "branch=applio-recode"
set "runtime=runtime-recode"
set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork/archive/refs/heads/%branch%.zip"
set "fixesFolder=lib/fixes"
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
    echo.
    goto reinstaller
    pause
    cls
    goto menu

)

if "%choice%"=="2" (
    cls
    echo.
    goto updater
    pause
    cls
    goto menu
)

if "%choice%"=="3" (
    cls
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
echo Python: Add this route to the windows enviroment variables the user path variable: %principal%\runtime\Scripts
echo.
pause
cls

echo Updating the repository...
git pull

echo Proceeding to download the models...
echo.

echo WARNING: At this point, it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
pause
cls

echo Downloading models in the assets folder...
cd "assets"
echo.
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

echo Downloading the hubert_base.pt file...
cd "hubert"
curl -LJO "%URL_BASE%/hubert_base.pt"
cd ".."
echo.
cls


echo Downloading the rmvpe.pt file...
cd "rmvpe"
curl -LJO "%URL_BASE%/rmvpe.pt"
echo.
cls

echo Downloading the rmvpe.onnx file...
curl -LJO "%URL_BASE%/rmvpe.onnx"
cd ".."
cd ".."
echo.
cls

echo Downloading the rest of the large files
cd "assets"
echo Downloading the "uvr5_weights" folder...
cd "uvr5_weights"
curl -LJO "%URL_BASE%/uvr5_weights/HP2_all_vocals.pth"
curl -LJO "%URL_BASE%/uvr5_weights/HP3_all_vocals.pth"
curl -LJO "%URL_BASE%/uvr5_weights/HP5_only_main_vocal.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoAggressive.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoDeReverb.pth"
curl -LJO "%URL_BASE%/uvr5_weights/VR-DeEchoNormal.pth"
cd ".."
cd ".."
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

echo Downloading torchcrepe
mkdir temp_torchcrepe
echo.

echo Clone the GitHub repository to the temporary directory
git clone --depth 1 https://github.com/maxrmorrison/torchcrepe.git temp_torchcrepe

echo Copy the "torchcrepe" folder and its contents to the current directory
robocopy "temp_torchcrepe\torchcrepe" ".\torchcrepe" /E
echo.

echo Remove the temporary directory
rmdir /s /q temp_torchcrepe
echo.

echo Downloads completed!
echo.

echo Applio has been reinstalled!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu


:updater

echo Updating the repository...
git pull

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

echo Updating the repository...
git pull

echo Installing dependencies...

echo [1] Nvidia graphics cards
echo [2] AMD / Intel graphics cards
echo [3] I have already installed the dependencies
echo.

set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls
pip uninstall tb-nightly tensorboardX tensorboard
echo.
pip install -r assets/requirements/requirements.txt
echo.
pip uninstall torch torchvision torchaudio -y
echo.
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
echo.
echo.
cls
echo Dependencies installed!
echo.
goto dependenciesFinished
)

if "%choice%"=="2" (
cls
pip uninstall tb-nightly tensorboardX tensorboard
echo.
pip install -r assets/requirements/requirements-dml.txt
echo.
echo.
cls
echo Dependencies installed!
echo.
goto dependenciesFinished
)

if "%choice%"=="3" (
echo Dependencies installed!
echo.
goto dependenciesFinished
)

:dependenciesFinished
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
