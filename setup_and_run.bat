@echo off
echo ============================================================
echo Privacy-Preserving Federated Learning System - Setup
echo ============================================================
echo.

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [Step 1/3] Downloading datasets...
python preprocessing\data_loader.py --download
echo.

echo [Step 2/3] Partitioning data for 3 clients...
python preprocessing\partitioner.py --num-clients 3 --dataset heart_disease
echo.

echo [Step 3/3] Setup complete!
echo.
echo ============================================================
echo DATA PREPARATION SUCCESSFUL!
echo ============================================================
echo.
echo Now you can start training:
echo.
echo Option 1: Use the batch files (EASIEST):
echo   - Double-click: start_server.bat
echo   - Double-click: start_client0.bat
echo   - Double-click: start_client1.bat
echo   - Double-click: start_client2.bat
echo.
echo Option 2: Manual commands in separate terminals:
echo   Terminal 1: start_server.bat
echo   Terminal 2: start_client0.bat
echo   Terminal 3: start_client1.bat
echo   Terminal 4: start_client2.bat
echo.
echo After training completes, launch dashboard:
echo   Double-click: start_dashboard.bat
echo.
echo ============================================================
pause
