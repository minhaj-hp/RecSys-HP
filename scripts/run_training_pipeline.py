#!/usr/bin/env python3
"""
Complete training pipeline for the two-tower recommendation system.
Runs all training steps in sequence: item pretraining, user dataset creation, 
FAISS indexing, and joint training.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ COMPLETED: {description} ({elapsed:.1f}s)")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {description} ({elapsed:.1f}s)")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def check_dependencies():
    """Check if required directories and files exist."""
    print("Checking dependencies...")
    
    required_paths = [
        "datasets/items.csv",
        "datasets/users.csv", 
        "datasets/interactions.csv",
        "src/models/item_tower.py",
        "src/models/user_tower.py",
        "src/preprocessing/data_loader.py"
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print(f"❌ Missing required files/directories: {missing}")
        return False
    
    print("✅ All dependencies found")
    return True

def main():
    """Run the complete training pipeline."""
    
    print("Two-Tower Recommendation System Training Pipeline")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create artifacts directory
    os.makedirs("src/artifacts", exist_ok=True)
    
    # Training pipeline steps
    steps = [
        {
            "command": "python -m src.training.item_pretraining",
            "description": "Item Tower Pre-training"
        },
        {
            "command": "python -m src.preprocessing.user_data_preparation",
            "description": "User Dataset Creation"
        },
        {
            "command": "python -m src.inference.faiss_index",
            "description": "FAISS Index Creation"
        },
        {
            "command": "python -m src.training.joint_training",
            "description": "Joint Training of Both Towers"
        },
        {
            "command": "python -m src.inference.recommendation_engine",
            "description": "Testing Recommendation Engine"
        }
    ]
    
    # Execute pipeline
    failed_steps = []
    start_time = time.time()
    
    for i, step in enumerate(steps, 1):
        print(f"\nSTEP {i}/{len(steps)}: {step['description']}")
        
        if not run_command(step["command"], step["description"]):
            failed_steps.append(step["description"])
            
            # Ask user if they want to continue
            response = input(f"\nStep failed. Continue anyway? (y/n): ").lower()
            if response != 'y':
                break
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if failed_steps:
        print(f"❌ Failed steps: {failed_steps}")
        print("Check the logs above for details.")
    else:
        print("✅ All steps completed successfully!")
        print("\nYou can now:")
        print("1. Start the API server: python api/main.py")
        print("2. Start the React frontend: cd frontend && npm start")
    
    print(f"\nArtifacts saved in: src/artifacts/")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)