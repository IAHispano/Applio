@echo off
title Applio - Manager
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
echo [3] Download NVDIA Runtime
echo [4] Download AMD Runtime
echo [5] Update Applio + Dependencies
echo [6] Fix Tensorboard
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
    goto nvdiaRuntime
    pause
    cls
    goto menu

)

if "%choice%"=="4" (
    cls
    echo.
    goto amdRuntime
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

:reinstaller

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

echo Reseting the repository...
git reset --hard
git pull
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
python.exe -m pip install onnxruntime
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

echo Applio has been reinstalled!
echo.
echo Press 'Enter' to access the main menu...
pause>nul
cls
goto menu


:updater

echo Updating the repository...
git pull
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
echo Verifying if the local_fixes.py file exists in the Fixes folder...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Running the file...
    if exist "%principal%\runtime" (
        runtime\python.exe "%fixesFolder%\%localFixesPy%"
    ) else (
        python.exe "%fixesFolder%\%localFixesPy%"
    )
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

:nvdiaRuntime
if exist "%principal%\runtime" (
    rmdir "%principal%\runtime" /s /q
)
cls
curl -LJO "%URL_EXTRA%/runtime.zip"
echo.
echo Extracting the runtime.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime.zip', '%principal%') }"
echo.
del runtime.zip
cls
echo NVDIA Runtime downloaded!
echo.
goto menu

:amdRuntime
if exist "%principal%\runtime" (
    rmdir "%principal%\runtime" /s /q
)

cls
curl -LJO "%URL_EXTRA%/runtime_dml.zip"
echo.
echo Extracting the runtime_dml.zip file...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('runtime_dml.zip', '%principal%') }"
echo.
del runtime_dml.zip
cls
echo AMD Runtime downloaded!
echo.
goto menu


