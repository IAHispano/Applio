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
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "INSTALL_ENV_DIR=%principal%\env"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Windows-x86_64.exe"
set "conda_exists=F"

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
echo [2] Download conda env (customized python environment designed for the installation of required dependencies)
echo.
echo Recommended for AMD/Intel GPU users (Broken): 
echo [3] Download DML Runtime (pre-installed dependencies)
echo.
echo Only recommended for experienced users:
echo [4] Nvidia graphics cards
echo [5] AMD / Intel graphics cards
echo.
echo [6] I have already installed the dependencies
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls
curl -LJO "%URL_EXTRA%/runtime.zip"
echo.
echo Extracting the runtime.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime.zip', '%principal%') }"
echo.
del runtime.zip
cls
echo.
goto dependenciesFinished
)

if "%choice%"=="2" (
cls
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

if "%conda_exists%" == "F" (
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%
curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

if not exist "%principal%\miniconda.exe" (
echo Download failed trying with the powershell method
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
)

echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" miniconda.exe /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%
del miniconda.exe
)

if not exist "%INSTALL_ENV_DIR%" (
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
echo Conda env installed !
)

call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"

echo Installing the dependencies...
pip install ffmpeg
pip install -r assets/requirements/requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

goto dependenciesFinished
)

if "%choice%"=="3" (
cls
curl -LJO "%URL_EXTRA%/runtime_dml.zip"
echo.
echo Extracting the runtime_dml.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime_dml.zip', '%principal%') }"
echo.
del runtime_dml.zip
cd runtime
python.exe -m pip install onnxruntime
cd ..
cls
echo.
goto dependenciesFinished
)

if "%choice%"=="4" (
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

if "%choice%"=="5" (
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

if "%choice%"=="6" (
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
