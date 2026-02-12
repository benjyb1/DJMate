#!/bin/bash

# DJMate Setup Script
# Run this script to set up your development environment

set -e  # Exit on error

echo "🎧 DJMate - AI DJ Curation API Setup"
echo "===================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo ""
    echo "⚠️  No virtual environment detected."
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Create a virtual environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        echo "✓ Virtual environment created in ./venv"
        echo ""
        echo "To activate it, run:"
        echo "  source venv/bin/activate  (Linux/Mac)"
        echo "  venv\\Scripts\\activate     (Windows)"
        echo ""
        read -p "Press enter to continue..."
    fi
fi

# Install package in editable mode
echo ""
echo "Installing DJMate package..."
pip install -e .

echo ""
echo "✓ Installation complete!"
echo ""
echo "To start the development server, run:"
echo "  uvicorn main:app --reload"
echo ""
echo "Or simply:"
echo "  python main.py"
echo ""
echo "API documentation will be available at:"
echo "  http://localhost:8000/docs"
echo ""
