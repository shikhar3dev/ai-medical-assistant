@echo off
echo Starting Enhanced AI Medical Dashboard...
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
call venv\Scripts\activate.bat
echo.
echo 🏥 AI Medical Assistant Dashboard starting...
echo 📱 Enhanced UI with camera support
echo 🔬 Advanced medical features
echo.
echo Dashboard will open at: http://localhost:8501
echo.
streamlit run enhanced_dashboard.py
pause
