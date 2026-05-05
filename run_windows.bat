@echo off
echo ==========================================
echo    Guardify - Auto Setup ^& Launch
echo ==========================================

:: 1. Check for Python
python --version >nul 2^>^&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b
)

:: 2. Backend Setup
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    call .\venv\Scripts\activate
    echo Installing AI libraries (this may take 2-3 mins)...
    pip install -e ".[api,training]"
    pip install python-dotenv
) else (
    call .\venv\Scripts\activate
)

:: 3. Environment Config
if not exist "apps\api\.env" (
    echo Creating .env config...
    copy apps\api\.env.example apps\api\.env
)

:: 4. Frontend Setup
pushd apps\web
if not exist "node_modules" (
    echo Installing Web Dashboard dependencies...
    call npm install
)

:: 5. Launch Servers
echo ==========================================
echo    STARTING SERVERS...
echo ==========================================

:: Start Backend in a new window
start "Guardify Backend API" cmd /k "cd ../.. && call .\venv\Scripts\activate && uvicorn apps.api.main:app --reload --port 8000 --env-file apps\api\.env"

:: Start Frontend in current window
echo Frontend starting...
call npm run dev

popd
pause
