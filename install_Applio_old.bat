@echo off
Title Applio-RVC-Fork - Installer
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

:processArguments
set "useManual=false"

for %%i in (%*) do (
    if /I "%%i"=="--manual" (
        set "useManual=true"
    ) else if /I "%%i"=="--condaNVDIA" (
        set "choice=1"
    ) else if /I "%%i"=="--condaDML" (
        set "choice=2"
    )
)



cls
echo INFO: It's recommended to disable antivirus or firewall, as errors might occur when downloading pretrained models.
echo.
pause
cls


net session >nul 2>&1
if %errorLevel% == 0 (
    echo You are executing the script with administrator privileges. Please run it without elevated permissions.
    echo Please, hit enter to exit.
    pause>nul
    exit
)

:endProcessArguments
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Cloning the repository...
    git clone %repoUrl% %repoFolder%
    echo Moving the mingit folder...
    cd %repoFolder%
    del install_Applio.bat
    del Makefile
    del Dockerfile
    del docker-compose.yaml
    del stftpitchshift
    del /q *.sh
    cls

) else (
    if not exist "%cd%\mingit.zip" (
    echo Downloading MinGit from %URL_EXTRA%/mingit.zip
    curl -s -LJO %URL_EXTRA%/mingit.zip -o mingit.zip
    )

    if not exist "%cd%\mingit.zip" (
    echo Download failed, trying with the powershell method
    powershell -Command "& {Invoke-WebRequest -Uri '%URL_EXTRA%/mingit.zip' -OutFile 'mingit.zip'}"
    )

    if not exist "%cd%\mingit" (
        echo Extracting the file...
        powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%cd%\mingit.zip', '%cd%') }"
    )

    if not exist "%cd%\mingit" (
        echo Extracting failed trying with the tar method...
        tar -xf %cd%\mingit.zip
    )

    if exist "%cd%\mingit" (
        del mingit.zip
        echo Cloning the repository...
        %cd%\mingit\cmd\git.exe clone %repoUrl% %repoFolder%
        echo Moving the mingit folder...
        robocopy "%cd%\mingit" "%principal%\lib\tools\mingit" /e /move /dcopy:t > NUL
        if errorlevel 8 echo Warnings or errors occurred during the move.
        cd %repoFolder%
        del install_Applio.bat
        del Makefile
        del Dockerfile
        del docker-compose.yaml
        del stftpitchshift
        del /q *.sh
        cls
    ) else (
        echo Theres a problem extracting the file please download the file and extract it manually 
        echo https://huggingface.co/IAHispano/applio/resolve/main/mingit.zip
        pause
        exit
    )
)


if /I "%useManual%"=="true" goto :skipMenu

:menu
echo.
echo You can install Applio-RVC-Fork in various ways. The first method is highly recommended for most users, as it sets up all the dependencies in a virtual environment, preventing compatibility issues. The remaining options may be in experimental stages or cater to more advanced users.
echo.
echo Recommended for Nvidia/AMD/Intel GPU and non-GPU users:
echo [1] Download the dependencies using a Python virtual environment (Conda)
echo.
echo Recommended for AMD/Intel GPU users (Potentially buggy):
echo [2] Download the DML dependencies using a Python virtual environment (Conda)
echo.
echo [3] Skip dependency installation
echo.
set /p choice=Select the most appropriate option in your case (Option 1 is generally recommended): 
set choice=%choice: =%

:skipMenu
cls


if not exist "%cd%\env.zip" (
    echo Downloading the fairseq build from %URL_EXTRA%/env.zip
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
    echo Theres a problem extracting the file please download the file and extract it manually in the applio folder
    echo https://huggingface.co/IAHispano/applio/resolve/main/env.zip
    pause
    exit
)


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

    echo Packages to install: %PACKAGES_TO_INSTALL%
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
    echo Conda env installed !

    echo Installing the dependencies...
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
    conda install -c anaconda git -y
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
    pip install --upgrade setuptools
    pip install -r "%principal%\assets\requirements\requirements.txt"
    pip install future==0.18.2
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
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

    echo Packages to install: %PACKAGES_TO_INSTALL%
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
    echo Conda env installed !

    echo Installing the dependencies...
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
    conda install -c anaconda git -y
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
    pip uninstall onnxruntime onnxruntime-directml
    pip install -r "%principal%\assets\requirements\requirements-dml.txt"
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate
    goto dependenciesFinished
)

if "%choice%"=="3" (
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
