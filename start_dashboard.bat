@echo off
echo Starting Streamlit Dashboard...
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
call venv\Scripts\activate.bat
echo.
echo Dashboard will open at: http://localhost:8501
echo.
streamlit run dashboard\app.py
pause
