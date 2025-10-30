from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union


class TextInput(BaseModel):
    """Single text input for classification."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class BatchTextInput(BaseModel):
    """Batch of texts for classification."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    return_probabilities: bool = Field(False, description="Whether to return class probabilities")
    
    @validator('texts')
    def validate_texts(cls, v):
        # Check each text is not empty
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} is empty')
        return [text.strip() for text in v]


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    text: str = Field(..., description="Input text")
    predicted_class: int = Field(..., description="Predicted class ID")
    label: Union[str, int] = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    is_confident: bool = Field(..., description="Whether confidence exceeds threshold")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of texts processed")


class TopKPrediction(BaseModel):
    """Single prediction with probability."""
    label: Union[str, int] = Field(..., description="Class label")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability")


class TopKResponse(BaseModel):
    """Response for top-k predictions."""
    text: str = Field(..., description="Input text")
    top_predictions: List[TopKPrediction] = Field(..., description="Top-k predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str = Field(..., description="Type of model")
    num_classes: Optional[int] = Field(None, description="Number of classes")
    classes: Optional[List[str]] = Field(None, description="Class labels")
    device: str = Field(..., description="Device being used")
    max_input_length: Optional[int] = Field(None, description="Maximum input length")