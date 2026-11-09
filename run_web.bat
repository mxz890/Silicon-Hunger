@echo off
setlocal
cd /d "%~dp0"

if not exist "silicon_env\Scripts\streamlit.exe" (
    echo [ERROR] Streamlit is not installed in silicon_env.
    echo   silicon_env\Scripts\python.exe -m pip install -r requirements_web.txt
    exit /b 1
)

call "silicon_env\Scripts\activate.bat"
set PYTHONIOENCODING=utf-8

REM Streamlit insists on ~/.streamlit for credentials.  Redirect its home
REM into this project so it never writes outside the workspace sandbox.
set "USERPROFILE=%CD%"
set "HOME=%CD%"

streamlit run web_app.py %*
