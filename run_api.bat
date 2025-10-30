@echo off
REM Script to run the FastAPI server on Windows

echo Starting Text Classification API...

REM Check if config exists
if not exist "config.yaml" (
    echo Error: config.yaml not found!
    exit /b 1
)

REM Check if model exists
if not exist "models\nb_model_package.pkl" (
    echo Warning: Model file not found at models\nb_model_package.pkl
    echo Please ensure your model is in the correct location.
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run the server
echo Starting server on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload