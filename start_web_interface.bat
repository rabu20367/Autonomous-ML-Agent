@echo off
REM Startup script for Windows - Autonomous ML Agent Web Interface

echo 🤖 Autonomous ML Agent - Web Interface Launcher
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Generate example datasets
echo 📊 Generating example datasets...
python examples\datasets\generate_all_datasets.py

REM Start the web interface
echo 🚀 Starting Streamlit Dashboard...
echo    URL: http://localhost:8501
echo.
echo 💡 Press Ctrl+C to stop the server
echo.

start http://localhost:8501
python -m streamlit run autonomous_ml\web_dashboard.py --server.port 8501

pause
