cd ..
runtime\python.exe Fixes\local_fixes.py

runtime\python.exe -m pip install --upgrade tensorboard markdown

runtime\python.exe -m pip install -r requirements.txt --upgrade

echo Press Enter to close...
pause >nul
