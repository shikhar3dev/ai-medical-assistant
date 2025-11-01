@echo off
echo Starting Enhanced Medical AI Dashboard with Dermatological Analysis...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Install any missing dependencies
echo Installing/updating dependencies...
pip install opencv-python-headless>=4.8.0 --quiet
pip install Pillow>=10.0.0 --quiet

echo.
echo Starting Streamlit dashboard...
echo Navigate to the "Medical Imaging" section to test the enhanced analysis
echo.

streamlit run enhanced_dashboard.py --server.port 8502 --server.headless true
pause
