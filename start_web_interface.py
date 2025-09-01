#!/usr/bin/env python3
"""
Startup script for the Autonomous ML Agent Web Interface
Provides easy access to all web interfaces
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import flask
        import fastapi
        import uvicorn
        print("✅ All web dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_streamlit_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Streamlit Dashboard...")
    print("   URL: http://localhost:8501")
    
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "autonomous_ml/web_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n⏹️ Streamlit dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def start_flask_app():
    """Start the Flask web application"""
    print("🚀 Starting Flask Web Application...")
    print("   URL: http://localhost:5000")
    
    try:
        # Change to the web_interface directory
        os.chdir(Path(__file__).parent / "web_interface")
        
        # Start Flask app
        subprocess.run([
            sys.executable, "app.py"
        ])
    except KeyboardInterrupt:
        print("\n⏹️ Flask application stopped")
    except Exception as e:
        print(f"❌ Error starting Flask: {e}")

def start_fastapi_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI Backend...")
    print("   URL: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Start FastAPI
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "autonomous_ml.web_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n⏹️ FastAPI backend stopped")
    except Exception as e:
        print(f"❌ Error starting FastAPI: {e}")

def generate_example_datasets():
    """Generate example datasets"""
    print("📊 Generating example datasets...")
    
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Run dataset generation
        subprocess.run([
            sys.executable, "examples/datasets/generate_all_datasets.py"
        ])
        
        print("✅ Example datasets generated successfully!")
        print("   You can now upload these datasets through the web interface")
        
    except Exception as e:
        print(f"❌ Error generating datasets: {e}")

def open_browser(url):
    """Open browser to the specified URL"""
    try:
        webbrowser.open(url)
        print(f"🌐 Opening browser to {url}")
    except Exception as e:
        print(f"❌ Error opening browser: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Autonomous ML Agent Web Interface Launcher")
    parser.add_argument(
        "interface", 
        choices=["streamlit", "flask", "fastapi", "datasets", "all"],
        help="Which interface to start"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    print("🤖 Autonomous ML Agent - Web Interface Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if args.check_deps:
        return 0
    
    # Generate datasets if requested
    if args.interface in ["datasets", "all"]:
        generate_example_datasets()
        if args.interface == "datasets":
            return 0
    
    # Start the requested interface
    if args.interface == "streamlit":
        if not args.no_browser:
            time.sleep(2)  # Give Streamlit time to start
            open_browser("http://localhost:8501")
        start_streamlit_dashboard()
    
    elif args.interface == "flask":
        if not args.no_browser:
            time.sleep(2)  # Give Flask time to start
            open_browser("http://localhost:5000")
        start_flask_app()
    
    elif args.interface == "fastapi":
        if not args.no_browser:
            time.sleep(2)  # Give FastAPI time to start
            open_browser("http://localhost:8000/docs")
        start_fastapi_backend()
    
    elif args.interface == "all":
        print("🚀 Starting all interfaces...")
        print("   Streamlit: http://localhost:8501")
        print("   Flask: http://localhost:5000")
        print("   FastAPI: http://localhost:8000")
        print("\n💡 Use Ctrl+C to stop all services")
        
        if not args.no_browser:
            time.sleep(3)
            open_browser("http://localhost:8501")
        
        # Start all services (this would require multiprocessing in a real implementation)
        print("⚠️ Starting all services simultaneously is not implemented yet.")
        print("   Please start them separately or use a process manager like PM2.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
