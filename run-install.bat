@echo off
setlocal
title Installer

set "principal=%cd%"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "INSTALL_ENV_DIR=%principal%\env"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Windows-x86_64.exe"
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"

del Makefile
del Dockerfile
del docker-compose.yaml
del /q *.sh

if not exist "%cd%\env.zip" (
    echo Downloading the fairseq build...
    curl -s -LJO %URL_EXTRA%/env.zip -o env.zip
)

if not exist "%cd%\env.zip" (
    echo Download failed, trying with the powershell method
    powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/env.zip' -OutFile 'mingit.zip'}"
)

if not exist "%cd%\env" (
    echo Extracting the file...
    powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%cd%\env.zip', '%cd%') }"
)

if not exist "%cd%\env" (
    echo Extracting failed trying with the tar method...
    tar -xf %cd%\env.zip
)

if exist "%cd%\env" (
    del env.zip
) else (
    echo Theres a problem extracting the file please download the file and extract it manually.
    echo https://huggingface.co/IAHispano/applio/resolve/main/env.zip
    pause
    exit
)

if not exist "%CONDA_EXECUTABLE%" (
    echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%...
    curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

    if not exist "%principal%\miniconda.exe" (
        echo Download failed trying with the powershell method.
        powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
    )

    echo Installing Miniconda to %CONDA_ROOT_PREFIX%...
    start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
    del miniconda.exe
)

call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9

echo Installing the dependencies...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
pip install --upgrade setuptools
pip install -r "%principal%\requirements.txt"
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
echo.

echo Applio has been installed successfully, run 'run-applio.bat' to start it!
pause
cls
