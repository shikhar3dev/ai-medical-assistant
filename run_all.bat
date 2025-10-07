@echo off
REM Batch script to run the federated learning system

echo ============================================================
echo Privacy-Preserving Federated Learning System
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/6] Verifying installation...
python verify_installation.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Installation verification failed!
    pause
    exit /b 1
)

echo.
echo [2/6] Downloading and preparing data...
python preprocessing\data_loader.py --download
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Data download may have failed, continuing with synthetic data...
)

echo.
echo [3/6] Partitioning data for 3 clients...
python preprocessing\partitioner.py --num-clients 3 --dataset heart_disease
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data partitioning failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Data preparation complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. Start FL Server (in a new terminal):
echo    cd c:\ai_disease_prediction
echo    venv\Scripts\activate
echo    python federated\server.py --rounds 10 --min-clients 3
echo.
echo 2. Start Client 0 (in a new terminal):
echo    cd c:\ai_disease_prediction
echo    venv\Scripts\activate
echo    python federated\client.py --client-id 0
echo.
echo 3. Start Client 1 (in a new terminal):
echo    cd c:\ai_disease_prediction
echo    venv\Scripts\activate
echo    python federated\client.py --client-id 1
echo.
echo 4. Start Client 2 (in a new terminal):
echo    cd c:\ai_disease_prediction
echo    venv\Scripts\activate
echo    python federated\client.py --client-id 2
echo.
echo ============================================================
pause
