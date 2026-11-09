@echo off
setlocal
cd /d "%~dp0"

if not exist "silicon_env\Scripts\python.exe" (
    echo [ERROR] silicon_env not found. Create it first:
    echo   python -m venv silicon_env
    echo   silicon_env\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

call "silicon_env\Scripts\activate.bat"
set PYTHONIOENCODING=utf-8
python main.py %*
