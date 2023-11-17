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
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"
set "tempFile=%cd%\powershell_output.txt"
set "buildToolsUrl=https://aka.ms/vs/17/release/vs_BuildTools.exe"

echo.
cls
echo INFO: It's important not to run this installer as an administrator as it might cause issues, and it's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
echo.
pause
cls
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Cloning the repository...
    git clone %repoUrl% %repoFolder%
    echo Moving the mingit folder...
    cd %repoFolder%
    del install_Applio.bat
    del /q *.sh
    cls

) else (
    if not exist "%cd%\mingit.zip" (
    echo Downloading MinGit from %URL_EXTRA%/mingit.zip
    curl -s -LJO %URL_EXTRA%/mingit.zip -o mingit.zip
    )

    if not exist "%cd%\mingit.zip" (
    echo Download failed trying with the powershell method
    powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/mingit.zip' -OutFile 'mingit.zip'}"
    )

    set powershellOutput="goodExtract"

    if not exist "%cd%\mingit" (
        set "powershellScript=try { Add-Type -AssemblyName 'System.IO.Compression.FileSystem'; [System.IO.Compression.ZipFile]::ExtractToDirectory('%cd%\mingit.zip', '%cd%'); Write-Output 'goodExtract' } catch { Write-Output 'errorExtract' }"
        powershell -NoProfile -ExecutionPolicy Bypass -Command "%powershellScript%" > "%tempFile%"
        set /p powershellOutput=<"%tempFile%"
        del "%tempFile%"
    )

    if "%powershellOutput%"=="goodExtract" (
        del mingit.zip
        echo Cloning the repository...
        %cd%\mingit\cmd\git.exe clone %repoUrl% %repoFolder%
        echo Moving the mingit folder...
        robocopy "%cd%\mingit" "%principal%\lib\tools\mingit" /e /move /dcopy:t > NUL
        if errorlevel 8 echo Warnings or errors occurred during the move.
        cd %repoFolder%
        del install_Applio.bat
        del /q *.sh
        echo.
        echo Do you want to continue?
        pause>nul
        cls
        
    ) else (
        echo Error extracting! Deleting file and trying directly with PowerShell method!
        del mingit.zip
        powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/mingit.zip' -OutFile 'mingit.zip'}"
        powershell -NoProfile -ExecutionPolicy Bypass -Command "%powershellScript%" > "%tempFile%"
        set /p powershellOutput=<"%tempFile%"
        del "%tempFile%"
        if "%powershellOutput%"=="errorExtract" (
            echo Theres a problem extracting the file please download the file and extract it manually 
            echo https://huggingface.co/IAHispano/applio/resolve/main/mingit.zip
            pause
            exit
        ) else (
            del mingit.zip
            echo Cloning the repository...
            %cd%\mingit\cmd\git.exe clone %repoUrl% %repoFolder%
            echo Moving the mingit folder...
            robocopy "%cd%\mingit" "%principal%\lib\tools\mingit" /e /move /dcopy:t > NUL
            if errorlevel 8 echo Warnings or errors occurred during the move.
            cd %repoFolder%
            del install_Applio.bat
            del /q *.sh
            echo.
            echo Do you want to continue?
            pause>nul
            cls
        )
    )
)

:menu
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

if not exist "%principal%\vs_BuildTools.exe" (
    curl -s -LJO %buildToolsUrl% -o vs_BuildTools.exe
    echo Downloading vs_BuildTools from %buildToolsUrl%
)
if not exist "%principal%\vs_BuildTools.exe" (
    echo Download failed trying with the powershell method
    del vs_BuildTools.exe
    powershell -Command "& {Invoke-WebRequest -Uri '%buildToolsUrl%' -OutFile 'vs_BuildTools.exe'}"
)
vs_BuildTools.exe --add Microsoft.VisualStudio.Workload.ManagedDesktopBuildTools --add Microsoft.VisualStudio.Workload.VCTools --passive
echo Installing vs_BuildTools...
echo.
echo Wait till the installation is finished (the installer will close automatically), then press any key to continue...
pause>nul
cls

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
goto dependenciesFinished
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
goto dependenciesFinished
)

if "%choice%"=="3" (
cls
echo INFO: Please ensure you have installed the required dependencies before continuing. Refer to the installation guide for details.
echo Step-by-step guide: https://rentry.org/appliolocal
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
goto dependenciesFinished
)

if "%choice%"=="4" (
cls
echo INFO: Please ensure you have installed the required dependencies before continuing. Refer to the installation guide for details.
echo Step-by-step guide: https://rentry.org/appliolocal
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
