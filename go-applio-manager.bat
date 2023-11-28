@echo off
title Applio-RVC-Fork - Manager
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

setlocal 
set "fixesFolder=lib/fixes"
set "localFixesPy=local_fixes.py"
set "principal=%cd%"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"
set "mingit_path=%cd%\lib\tools\mingit\cmd\git.exe"
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"
set "INSTALL_ENV_DIR=%principal%\env"

:menu
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A

echo [1] Reinstall Applio
echo [2] Update Applio
echo [3] Download conda env
echo [4] Download Runtime
echo [5] Update Applio + Dependencies (Local Python)
echo [6] Fix Tensorboard (Local Python)
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
    goto condaEnv
    pause
    cls
    goto menu
)

if "%choice%"=="4" (
    cls
    echo.
    goto runtime
    pause
    cls
    goto menu

)

if "%choice%"=="5" (
    cls
    echo.
    goto updaterDependencies
    pause
    cls
    goto menu

)

if "%choice%"=="6" (
    cls
    echo.
    pip uninstall tb-nightly tensorboardX tensorboard
    pip install tensorboard
    cls
    echo Tensorboard re-installed correctly!
    echo.	
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


rem Start Reinstaller
:reinstaller

cls
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

echo Reseting the repository...
git reset --hard
git pull
echo.
cls

echo Installing dependencies...
echo.
echo Recommended for Nvdia/AMD/Intel GPU and non GPU users:
echo [1] Download Nvdia conda env (customized python environment designed for the installation of required dependencies)
echo.
echo Recommended for AMD/Intel GPU users (Broken): 
echo [2] Download DML conda env (customized python environment designed for the installation of required dependencies)
echo.
echo Only recommended for experienced users:
echo [3] Nvidia graphics cards
echo [4] AMD / Intel graphics cards (Broken)
echo.
echo [5] I have already installed the dependencies
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
cls

if not exist "%CONDA_EXECUTABLE%" (
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%
curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

if not exist "%principal%\miniconda.exe" (
echo Download failed trying with the powershell method
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
)

echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
del miniconda.exe
)

if not exist "%INSTALL_ENV_DIR%" (
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
echo Conda env installed !
)

echo Installing the dependencies...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
conda install -c anaconda git -y
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
pip install -r %principal%\assets\requirements\requirements.txt
pip install future==0.18.2
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
goto reinstallFinished
)

if "%choice%"=="2" (
cls

if not exist "%CONDA_EXECUTABLE%" (
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%
curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

if not exist "%principal%\miniconda.exe" (
echo Download failed trying with the powershell method
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
)

echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
del miniconda.exe
)

if not exist "%INSTALL_ENV_DIR%" (
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
echo Conda env installed !
)

echo Installing the dependencies...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
conda install -c anaconda git -y
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
pip uninstall onnxruntime onnxruntime-directml
pip install -r %principal%\assets\requirements\requirements.txt
pip install -r %principal%\assets\requirements\requirements-dml.txt
pip install future==0.18.2
goto reinstallFinished
)

if "%choice%"=="3" (
cls
echo INFO: Please ensure you have installed the required dependencies before continuing. Refer to the installation guide for details.
echo Step-by-step guide: https://rentry.org/appliolocal
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
echo Python 3.9.8: https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe
pause
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
goto reinstallFinished
)

if "%choice%"=="4" (
cls
echo INFO: Please ensure you have installed the required dependencies before continuing. Refer to the installation guide for details.
echo Step-by-step guide: https://rentry.org/appliolocal
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
echo Python 3.9.8: https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe
pause
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
goto reinstallFinished
)

if "%choice%"=="5" (
echo Dependencies successfully installed!
echo.
goto reinstallFinished
)

:reinstallFinished
echo Applio has been reinstalled!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
rem End Reinstaller


rem Start Updater
:updater

echo Updating the repository...
if exist "%mingit_path%" (
    %mingit_path% pull
) else (
    git pull
)
echo Applio has been updated!
echo.
echo Press 'Enter' to access the main menu... 
pause>nul
cls
goto menu
rem End Updater


rem Start Updater Dependencies
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
pip uninstall onnxruntime onnxruntime-directml
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
pip uninstall onnxruntime onnxruntime-directml
echo.
pip install -r assets/requirements/requirements.txt
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
echo Applio has been updated!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
rem End Updater Dependencies


rem Start Conda env
:condaEnv

echo Installing the conda env...
echo.
echo Recommended for Nvdia/AMD/Intel GPU and non GPU users:
echo [1] Download Nvdia conda env (customized python environment designed for the installation of required dependencies)
echo.
echo Recommended for AMD/Intel GPU users (Broken): 
echo [2] Download DML conda env (customized python environment designed for the installation of required dependencies)
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
    goto NVDIAcondaEnv
)
if "%choice%"=="2" (
    goto DMLcondaEnv
)

:NVDIAcondaEnv
cls
if not exist "%CONDA_EXECUTABLE%" (
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%
curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

if not exist "%principal%\miniconda.exe" (
echo Download failed trying with the powershell method
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
)

echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
del miniconda.exe
)

if not exist "%INSTALL_ENV_DIR%" (
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
echo Conda env installed !
)

echo Installing the dependencies...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
conda install -c anaconda git -y
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
pip install -r %principal%\assets\requirements\requirements.txt
pip install future==0.18.2
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
cls
echo Conda env downloaded!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu

:DMLcondaEnv
cls
if not exist "%CONDA_EXECUTABLE%" (
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL%
curl %MINICONDA_DOWNLOAD_URL% -o miniconda.exe

if not exist "%principal%\miniconda.exe" (
echo Download failed trying with the powershell method
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
)

echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
del miniconda.exe
)

if not exist "%INSTALL_ENV_DIR%" (
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
echo Conda env installed !
)

echo Installing the dependencies...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
conda install -c anaconda git -y
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
pip uninstall onnxruntime onnxruntime-directml
pip install -r %principal%\assets\requirements\requirements.txt
pip install -r %principal%\assets\requirements\requirements-dml.txt
pip install future==0.18.2
cls
echo Conda DML env downloaded!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
rem End Conda env


rem Start Runtime
:runtime

cls
echo Installing the runtime...
echo.
echo Recommended for Nvdia/AMD/Intel GPU and non GPU users:
echo [1] Download Nvdia Runtime (pre-installed dependencies)
echo.
echo Recommended for AMD/Intel GPU users (Broken): 
echo [2] Download DML Runtime (pre-installed dependencies)
echo.
set /p choice=Select the option according to your GPU: 
set choice=%choice: =%

if "%choice%"=="1" (
    goto nvdiaRuntime
)
if "%choice%"=="2" (
    goto dmlRuntime
)


:nvdiaRuntime
cls
if exist "%principal%\runtime" (
    rmdir "%principal%\runtime" /s /q
) else (
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
)

cls
curl -LJO "%URL_EXTRA%/runtime.zip"
echo.
if not exist "%principal%\runtime.zip" (
    echo Download failed trying with the powershell method
    powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/runtime.zip' -OutFile 'runtime.zip'}"
)
echo.
echo Extracting the runtime.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime.zip', '%principal%') }"
echo.
del runtime.zip
cls
echo NVDIA Runtime downloaded!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu


:dmlRuntime
cls
if exist "%principal%\runtime" (
    rmdir "%principal%\runtime" /s /q
) else (
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
)

cls
curl -LJO "%URL_EXTRA%/runtime_dml.zip"
echo.
if not exist "%principal%\runtime_dml.zip" (
    echo Download failed trying with the powershell method
    powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/runtime_dml.zip' -OutFile 'runtime_dml.zip'}"
)
echo.
echo Extracting the runtime_dml.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime_dml.zip', '%principal%') }"
echo.
del runtime_dml.zip
cls
echo AMD Runtime downloaded!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu
rem End Runtime
