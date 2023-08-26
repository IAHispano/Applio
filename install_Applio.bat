@echo off
Title Instalador de Applio
chcp 65001 > nul
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
::: Versión 1.0.0 - Desarrollado por Aitron
::: 

set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork"
set "repoFolder=Applio-RVC-Fork"
set "fixesFolder=Fixes"
set "localFixesPy=local_fixes.py"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
echo.
cls

echo AVISO: Recuerda instalar las Microsoft C++ Build Tools, El Redistributable, Python y Git antes de continuar.
echo.
echo Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
echo Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Git: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.1/Git-2.42.0-64-bit.exe
echo Python: https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe
echo.
pause
cls

echo.

echo Creando carpeta para el repositorio...
mkdir "%repoFolder%"
cd "%repoFolder%"
echo.

echo Clonando el repositorio...
git clone "%repoUrl%" .
echo.

echo Instalando dependencias para ejecutar el archivo Fixes
pip install requests

echo.
echo Verificando si el archivo local_fixes.py existe en la carpeta Fixes...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Ejecutando el archivo...
    python "%fixesFolder%\%localFixesPy%"
) else (
    echo El archivo "%localFixesBat%" no se encontró en la carpeta "Fixes".
)

echo Pasando a descargar los modelos...
echo.

echo AVISO: En este punto, se recomienda desactivar el antivirus o el firewall, ya que existe la posibilidad de que ocurran errores al descargar los modelos preentrenados.
pause
cls

echo Descargando la carpeta "pretrained"...
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

echo Descargando la carpeta "pretrained_v2"...
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

echo Descargando la carpeta "uvr5_weights"...
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

echo Descargando el archivo rmvpe.pt...
curl -LJO "%URL_BASE%/rmvpe.pt"
echo.
cls

echo Descargando el archivo ffmpeg.exe...
curl -LJO "%URL_BASE%/ffmpeg.exe"
echo.
cls

echo Descargando el archivo ffprobe.exe...
curl -LJO "%URL_BASE%/ffprobe.exe"
echo.
echo Descargas completadas. Procediendo con las dependencias.
cls

echo ¿Tienes una GPU?
echo Esto determinará si se descargan dependencias ligeras (sin GPU) o pesadas (con GPU).
echo.


set /p op=Escribe "Si" o "No": 
if "%op%"=="Si" goto gpu
if "%op%"=="No" goto non_gpu

:gpu
echo Se ha seleccionado GPU, continuando...
echo.
echo Descargando las dependencias...
echo.
pip install -r requirements-gpu.txt
pip uninstall torch torchvision torchaudio -y
echo.
echo NOTA: El ordenador puede experimentar lentitud durante este proceso; no te preocupes.
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endlocal
echo.
cls
echo ¡Applio ha sido descargado!
echo.
pause
color 07
exit

:non_gpu
echo No se ha seleccionado GPU, continuando...
echo.
echo Descargando las dependencias...
echo.
pip install -r requirements.txt
echo.
echo ¡Applio ha sido descargado!
endlocal
echo.
pause
color 07
exit
