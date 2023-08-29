@echo off
Title Actualizador de Applio
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

set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork/archive/refs/heads/main.zip"
set "fixesFolder=Fixes"
set "localFixesPy=local_fixes.py"
set "subdir=temp_udpate"
set "principal=%cd%"
echo.
cls

echo Descargando el archivo ZIP...
powershell -command "& { Invoke-WebRequest -Uri '%repoUrl%' -OutFile '%principal%\repo.zip' }"
echo.

echo Extrayendo archivo ZIP...
powershell -command "& { Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory('%principal%\repo.zip', '%principal%') }"
echo.

echo Copiando estructura de carpetas y archivos desde el subdirectorio al directorio principal...
robocopy "%principal%\Applio-RVC-Fork-main" "%principal%" /E
echo.

echo Eliminando el contenido del subdirectorio (archivos y carpetas)...
rmdir "%principal%\Applio-RVC-Fork-main" /S /Q
echo.

echo Limpiando...
del "%principal%\repo.zip"
echo.
cls

echo Verificando si el archivo local_fixes.py existe en la carpeta Fixes...
if exist "%fixesFolder%\%localFixesPy%" (
    echo Ejecutando el archivo...
    python "%fixesFolder%\%localFixesPy%"
) else (
    echo El archivo "%localFixesBat%" no se encontró en la carpeta "Fixes".
)

echo Applio ha sido actualizado!
endlocal
echo.
pause
exit