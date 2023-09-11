@echo off
Title Applio No Runtime

echo Making the existing folder to a git repository
git init
echo.

echo Setting the repository to applio
git remote add origin https://github.com/IAHispano/Applio-RVC-Fork.git
echo.

echo Feching the origin
git fetch origin
echo.

echo Reseting the folder
git reset --hard origin/main
echo.

echo Trying to pullthe latest changes
git pull origin main
echo.

echo Installing dependencies...
pip install -r requirments.txt
echo.
pip uninstall torch torchvision torchaudio -y
echo.
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
echo.
pip install git+https://github.com/suno-ai/bark.git
echo.
cls
echo Dependencies installed!
echo.

echo Applio has been installed with no runtime!
echo.
pause
color 07
exit
