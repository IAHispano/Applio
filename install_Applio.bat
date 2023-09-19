@echo off
Title Applio - Installer
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
:::

set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork.git"
set "repoFolder=Applio-RVC-Fork"
set "principal=%cd%\%repoFolder%"
set "runtime_scripts=%cd%\%repoFolder%\runtime\Scripts"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"

echo.
cls
echo INFO: It's important not to run this installer as an administrator as it might cause issues, and it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
echo.
pause

cls
echo INFO: Please ensure you have installed the required dependencies before continuing. Refer to the installation guide for details.
echo.
echo Step-by-step guide: https://rentry.org/appliolocal
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
echo Python 3.9.8: https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe
echo.
echo INFO: Its recommend installing Python 3.9.X and ensuring that it has been added to the system's path.
echo.
pause
cls

for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

echo Cloning the repository...
git clone %repoUrl% %repoFolder%
cd %repoFolder%
echo.
cls

echo Proceeding to download the models...
echo.

echo Downloading the "pretrained" folder...
cd "assets/pretrained/"
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
cd "rmvpe"
curl -LJO "%URL_BASE%/rmvpe.pt"
echo.
cls
cd ".."

echo Downloading the hubert_base.pt file...
cd "hubert"
curl -LJO "%URL_BASE%/hubert_base.pt"
echo.
cls
cd ".."
cd ".."

echo Downloading the ffmpeg.exe and ffprobe.exe file...
curl -LJO "%URL_BASE%/ffmpeg.exe"
curl -LJO "%URL_BASE%/ffprobe.exe"

@echo off
setlocal

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

echo Torchcrepe downloaded successfully!
cls

echo Installing dependencies...

echo [1] Nvidia graphics cards
echo [2] AMD / Intel graphics cards
echo [3] Runtime
echo [4] I have already installed the dependencies
echo.

set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls
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
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements-dml.txt
echo.
echo.
cls
echo Dependencies installed!
echo.
goto dependenciesFinished
)

if "%choice%"=="3" (
cls
cd %repoFolder%
::curl -LJO "%URL_EXTRA%/runtime.zip" editame lol
curl -LJO "https://huggingface.co/iroaK/0xF3AC33/resolve/main/runtime.zip"
echo.
echo Extracting the runtime.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime.zip', '%principal%') }"
echo.
cls
echo INFO: Dependencies installed! Please add this folder to the system's path: "%runtime_scripts%" before continuing.
pause
echo.
goto dependenciesFinished
)

if "%choice%"=="4" (
echo Dependencies installed!
echo.
goto dependenciesFinished
)

:dependenciesFinished
cls 
echo Applio has been successfully downloaded, run the file go-applio.bat to run the web interface!
echo.
pause
exit
