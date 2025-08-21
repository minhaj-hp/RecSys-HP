#!/usr/bin/env python3
"""
Start the complete recommendation system demo:
1. API server (FastAPI)
2. React frontend
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutting down demo...")
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    print("Demo stopped.")
    sys.exit(0)

def run_api_server():
    """Start the FastAPI server."""
    print("Starting API server...")
    
    try:
        process = subprocess.Popen(
            ["python", "api/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(process)
        
        # Stream output
        for line in process.stdout:
            print(f"[API] {line.rstrip()}")
            
    except Exception as e:
        print(f"Error starting API server: {e}")

def run_react_frontend():
    """Start the React development server."""
    print("Starting React frontend...")
    
    try:
        # Check if node_modules exists
        if not Path("frontend/node_modules").exists():
            print("Installing React dependencies...")
            install_process = subprocess.run(
                ["npm", "install"],
                cwd="frontend",
                check=True
            )
        
        # Start React server
        process = subprocess.Popen(
            ["npm", "start"],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, "BROWSER": "none"}  # Don't auto-open browser
        )
        processes.append(process)
        
        # Stream output
        for line in process.stdout:
            print(f"[React] {line.rstrip()}")
            
    except Exception as e:
        print(f"Error starting React frontend: {e}")

def check_requirements():
    """Check if the system is ready to run the demo."""
    print("Checking requirements...")
    
    # Check if artifacts exist
    required_artifacts = [
        "src/artifacts/vocabularies.pkl",
        "src/artifacts/item_tower_weights.index",
        "src/artifacts/faiss_item_index.bin"
    ]
    
    missing = []
    for artifact in required_artifacts:
        if not Path(artifact).exists():
            missing.append(artifact)
    
    if missing:
        print(f"❌ Missing required artifacts: {missing}")
        print("Please run the training pipeline first: python run_training_pipeline.py")
        return False
    
    # Check if React app exists
    if not Path("frontend/package.json").exists():
        print("❌ React frontend not found")
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    """Main function to start the demo."""
    
    print("Two-Tower Recommendation System Demo")
    print("=" * 50)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("\nStarting demo servers...")
    print("The API will be available at: http://localhost:8000")
    print("The React app will be available at: http://localhost:3000")
    print("\nPress Ctrl+C to stop all servers.\n")
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server)
    api_thread.daemon = True
    api_thread.start()
    
    # Wait a bit for API server to start
    time.sleep(3)
    
    # Start React frontend (this will block)
    try:
        run_react_frontend()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()