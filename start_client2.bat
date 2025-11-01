@echo off
echo Starting FL Client 2...
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
call venv\Scripts\activate.bat
echo.
echo Connecting to server at localhost:8080
echo.
python federated\client.py --client-id 2

pause
