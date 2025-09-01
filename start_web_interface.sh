#!/bin/bash
# Startup script for Linux/Mac - Autonomous ML Agent Web Interface

echo "🤖 Autonomous ML Agent - Web Interface Launcher"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Generate example datasets
echo "📊 Generating example datasets..."
python examples/datasets/generate_all_datasets.py

# Start the web interface
echo "🚀 Starting Streamlit Dashboard..."
echo "   URL: http://localhost:8501"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Open browser (Linux/Mac)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501 &
elif command -v open &> /dev/null; then
    open http://localhost:8501 &
fi

# Start Streamlit
python -m streamlit run autonomous_ml/web_dashboard.py --server.port 8501
