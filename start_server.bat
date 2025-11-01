@echo off
echo Starting Federated Learning Server...
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
call venv\Scripts\activate.bat
echo.
echo Server starting on localhost:8080
echo Waiting for 3 clients to connect...
echo.
python federated\server.py --rounds 10 --min-clients 3

pause
