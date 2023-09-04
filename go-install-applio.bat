@echo off
Title Applio Installer
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
set "repoFolder=Applio-RVC-Fork"
set "fixesFolder=Fixes"
set "localFixesPy=local_fixes.py"
set "colabmdx=colab_for_mdx.py"
set "principal=%cd%\%repoFolder%"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"
echo.
cls

echo WARNING: It's important not to run this installer as an administrator as it might cause issues.
echo WARNING: Remember to install Microsoft C++ Build Tools, Redistributable, Python, and Git before continuing.
echo.
echo Step-by-step guide: https://rentry.org/appliolocal
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
echo Python: Add this route to the windows envirment variables the user path variable: %principal%\runtime\Scripts
echo.
pause
cls

for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

echo Creating folder for the repository...
mkdir "%repoFolder%"
cd "%repoFolder%"
echo.

echo Downloading ZIP file...
powershell -command "& { Invoke-WebRequest -Uri '%repoUrl%' -OutFile '%principal%\repo.zip' }"
echo.

echo Extracting ZIP file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%principal%\repo.zip', '%principal%') }"
echo.

echo Copying folder and file structure from subdirectory to main directory...
robocopy "%principal%\Applio-RVC-Fork-main" "%principal%" /E
echo.

echo Deleting contents of subdirectory (files and folders)...
rmdir "%principal%\Applio-RVC-Fork-main" /S /Q
echo.

echo Cleaning up...
del "%principal%\repo.zip"
echo.
cls

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
curl -LJO "%URL_EXTRA%/runtime.zip"
echo.
cls
echo Extracting the runtime.zip file, this might take a while...
powershell -Command "Expand-Archive -Path 'runtime.zip' -DestinationPath '.'"
del runtime.zip
echo.
cls

echo Downloads completed!
echo.

echo Checking if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    runtime\python.exe "%fixesFolder%\%localFixesPy%"
) else (
    echo The "%localFixesBat%" file was not found in the "Fixes" folder.
)
echo.

echo Fixes Applied!
echo.

echo Applio has been downloaded!
echo.
pause
color 07
exit
