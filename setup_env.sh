#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Smartphone Price Prediction Setup..."

# 1. Initialize Git Repository
if [ -d ".git" ]; then
    echo "✅ Git repository already initialized."
else
    echo "Initializing Git repository..."
    git init
fi

# 2. Python Virtual Environment Setup
if [ -d "venv" ]; then
    echo "✅ Virtual environment 'venv' already exists."
else
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install Dependencies
echo "Installing dependencies from requirements.txt..."
# Use python -m pip to ensure we use the venv's pip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Initialize Data Version Control (DVC)
if [ -d ".dvc" ]; then
    echo "✅ DVC repository already initialized."
else
    echo "Initializing DVC repository..."
    dvc init
fi

echo ""
echo "🎉 Setup complete! To start working, run:"
echo "    source venv/bin/activate"
