@echo off
echo ============================================
echo   AI Assistant Portal — EY Project
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Install dependencies if needed
echo [1/3] Checking dependencies...
pip install -r requirements.txt --quiet 2>nul

:: Create uploads directory if it doesn't exist
if not exist "uploads" mkdir uploads
if not exist "database" mkdir database

:: Run the application
echo [2/3] Starting AI Assistant Portal...
echo.
echo   Local:   http://127.0.0.1:5000
echo   Network: http://0.0.0.0:5000
echo.
echo   Press Ctrl+C to stop the server.
echo ============================================
echo.

python app.py
