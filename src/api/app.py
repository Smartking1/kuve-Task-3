from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yaml
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    TextInput, BatchTextInput, PredictionResponse,
    BatchPredictionResponse, TopKResponse, HealthResponse,
    ErrorResponse, ModelInfoResponse
)
from src.preprocessing.text_processor import TextProcessor
from src.model.loader import ModelLoader
from src.inference.predictor import TextClassificationPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="API for text classification inference using scikit-learn models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
config = None
model_info = {}


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_model():
    """Initialize model, preprocessor, and predictor."""
    global predictor, config, model_info
    
    try:
        # Load config
        config = load_config()
        logger.info("Configuration loaded")
        
        # Load model package
        model_loader = ModelLoader(config['model'])
        model, vectorizer, threshold = model_loader.load_model_package(
            config['model']['package_path']
        )
        logger.info("Model package loaded successfully")
        
        # Store model info
        model_info = {
            'model_type': type(model).__name__,
            'vectorizer_type': type(vectorizer).__name__,
            'threshold': threshold,
            'num_classes': len(model.classes_) if hasattr(model, 'classes_') else None,
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
        }
        
        # Initialize text processor with vectorizer
        text_processor = TextProcessor(config['preprocessing'], vectorizer=vectorizer)
        logger.info("Text processor initialized")
        
        # Initialize predictor
        predictor = TextClassificationPredictor(
            model=model,
            text_processor=text_processor,
            threshold=threshold,
            config=config['inference']
        )
        logger.info("Predictor initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting up...")
    try:
        initialize_model()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise - allow server to start but mark as unhealthy


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Classification API",
        "version": "1.0.0",
        "status": "healthy" if predictor is not None else "model not loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_top_k": "/predict/top-k",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        device="cpu"  # scikit-learn uses CPU
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: TextInput):
    """
    Predict class for a single text.
    
    - **text**: The text to classify
    
    Returns prediction with label and confidence score.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        result = predictor.predict_single(input_data.text, return_probabilities=True)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchTextInput):
    """
    Predict classes for multiple texts.
    
    - **texts**: List of texts to classify (max 100)
    - **return_probabilities**: Include probability scores for all classes
    
    Returns predictions for all input texts.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        results = predictor.predict_batch(
            input_data.texts,
            return_probabilities=input_data.return_probabilities
        )
        
        predictions = [PredictionResponse(**r) for r in results]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/top-k", response_model=TopKResponse, tags=["Prediction"])
async def predict_top_k(input_data: TextInput, k: int = 3):
    """
    Get top-k predictions for a text.
    
    - **text**: The text to classify
    - **k**: Number of top predictions to return (default: 3)
    
    Returns the k most likely classes with their probabilities.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    if k < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="k must be at least 1"
        )
    
    try:
        top_predictions = predictor.get_top_k_predictions(input_data.text, k=k)
        return TopKResponse(
            text=input_data.text,
            top_predictions=top_predictions
        )
    except Exception as e:
        logger.error(f"Top-k prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Top-k prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        return ModelInfoResponse(
            model_type=model_info.get('model_type', 'Unknown'),
            num_classes=model_info.get('num_classes'),
            classes=model_info.get('classes'),
            device="cpu",
            max_input_length=None  # No limit for scikit-learn models
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load config for port
    try:
        cfg = load_config()
        host = cfg.get('api', {}).get('host', '0.0.0.0')
        port = cfg.get('api', {}).get('port', 8000)
        reload = cfg.get('api', {}).get('reload', False)
    except:
        host = '0.0.0.0'
        port = 8000
        reload = False
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload
    )