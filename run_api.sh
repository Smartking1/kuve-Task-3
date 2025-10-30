#!/bin/bash

# Script to run the FastAPI server

echo "Starting Text Classification API..."

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Check if model exists
if [ ! -f "models/nb_model_package.pkl" ]; then
    echo "Warning: Model file not found at models/nb_model_package.pkl"
    echo "Please ensure your model is in the correct location."
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the server
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload