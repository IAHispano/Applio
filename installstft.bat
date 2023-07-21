runtime\python.exe -m pip install stftpitchshift --upgrade
runtime\python.exe -m pip install gradio==3.34.0 --upgrade


IF EXIST ".\rmvpe.pt" (echo RMVPE is already installed) ELSE (bitsadmin /transfer "rvmpedwnld" /download /priority FOREGROUND "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" "%~dp0rmvpe.pt")
pause