@echo off
Title Applio - Installer
setlocal
cd %~dp0

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
del install_Applio.bat
del /q *.sh
echo.
cls

echo Installing dependencies...
echo.
echo Recommended for Nvidia GPU users: 
echo [1] Download Runtime (pre-installed dependencies)
echo.
echo Recommended for AMD/Intel GPU users (Broken): 
echo [2] Download DML Runtime (pre-installed dependencies)
echo.
echo Only recommended for experienced users:
echo [3] Nvidia graphics cards
echo [4] AMD / Intel graphics cards
echo.
echo [5] I have already installed the dependencies
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls
powershell -command "Invoke-WebRequest -Uri https://frippery.org/files/busybox/busybox.exe -OutFile busybox.exe"
busybox.exe wget %URL_EXTRA%/runtime.zip
echo.
echo Extracting the runtime.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime.zip', '%principal%') }"
echo.
del runtime.zip busybox.exe
cls
echo.
goto dependenciesFinished
)

if "%choice%"=="2" (
cls
powershell -command "Invoke-WebRequest -Uri https://frippery.org/files/busybox/busybox.exe -OutFile busybox.exe"
busybox.exe wget %URL_EXTRA%/runtime_dml.zip
echo.
echo Extracting the runtime_dml.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime_dml.zip', '%principal%') }"
echo.
del runtime_dml.zip busybox.exe
cd runtime
python -m pip install onnxruntime
cd ..
cls
echo.
goto dependenciesFinished
)

if "%choice%"=="3" (
cls
pip install -r assets/requirements/requirements.txt
echo.
pip uninstall torch torchvision torchaudio -y
echo.
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
echo.
echo.
cls
echo Dependencies successfully installed!
echo.
goto dependenciesFinished
)

if "%choice%"=="4" (
cls
pip uninstall onnxruntime onnxruntime-directml
echo.
pip install -r assets/requirements/requirements.txt
echo.
pip install -r assets/requirements/requirements-dml.txt
echo.
echo.
cls
echo Dependencies successfully installed!
echo.
goto dependenciesFinished
)

if "%choice%"=="5" (
echo Dependencies successfully installed!
echo.
goto dependenciesFinished
)

:dependenciesFinished
cls 
echo Applio has been successfully downloaded, run the file go-applio.bat to run the web interface!
echo.
pause
exit

