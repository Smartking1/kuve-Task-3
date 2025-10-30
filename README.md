# Kuve-3: Text Classification API & Streamlit App

Kuve-3 is a Python project for text classification using machine learning models. It provides a FastAPI backend for inference and a Streamlit web app for interactive testing and analysis.

## Features
- FastAPI REST API for text classification
- Streamlit app for interactive predictions and analysis
- Batch and single text prediction
- Configurable preprocessing (lowercase, stopwords, etc.)
- Supports scikit-learn and Keras/TensorFlow models
- Model packaging with tokenizer/vectorizer and threshold
- Docker support for easy deployment

## Project Structure
```
kuve-3/
├── config.yaml                # Main configuration file
├── models/                    # Model packages (.pkl, .h5, etc.)
├── src/
│   ├── api/                   # FastAPI app and schemas
│   ├── inference/             # Prediction logic
│   ├── model/                 # Model loader
│   ├── preprocessing/         # Text preprocessing
│   └── scripts/               # CLI and test scripts
├── streamlit_app.py           # Streamlit web app
├── requirements.txt           # Python dependencies
├── docker/                    # Docker and compose files
└── .gitignore                 # Git ignore rules
```

## Quick Start
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your model
- Place your model package (e.g., `nb_model_package.pkl`) in the `models/` folder.
- Update `config.yaml` with the correct paths and settings.

### 3. Run the API
```bash
uvicorn src.api.app:app --reload
```

### 4. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### 5. Test via CLI
```bash
python src/scripts/predict_cli.py --interactive
```

## Configuration
Edit `config.yaml` to set model paths, preprocessing options, API settings, and more.

## Docker
To run everything in Docker:
```bash
docker-compose up --build
