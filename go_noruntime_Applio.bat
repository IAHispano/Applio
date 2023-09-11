@echo off
Title Applio Dependencies

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

echo Applio has been downloaded!
echo.
pause
color 07
exit
