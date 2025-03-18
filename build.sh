#!/bin/bash

# Exit on error
set -e

echo "Creating a virtual environment..."
python3 -m venv imago

echo "Activating the virtual environment..."
source imago/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Deactivating the virtual environment..."
deactivate

echo "Setup complete! Use 'source imago/bin/activate' to activate the environment."
