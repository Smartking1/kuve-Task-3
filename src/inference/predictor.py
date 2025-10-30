import numpy as np
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class TextClassificationPredictor:
    """Handles inference for Keras/TensorFlow text classification models."""
    
    def __init__(self, model, text_processor, threshold: float = 0.5, config: dict = None):
        self.model = model
        self.text_processor = text_processor
        self.threshold = threshold
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 32)
        
        # Determine if it's a Keras model or sklearn model
        self.is_keras = hasattr(model, 'predict') and not hasattr(model, 'predict_proba')
        
        # For binary classification (sigmoid output)
        self.is_binary = True
        
        logger.info(f"Predictor initialized with threshold: {threshold}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Is Keras model: {self.is_keras}")
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        Predict class for a single text.
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        return self.predict_batch([text], return_probabilities)[0]
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> List[Dict]:
        """
        Predict classes for a batch of texts.
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
        
        Returns:
            List of dictionaries with prediction results
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._predict_batch_internal(batch_texts, return_probabilities)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, texts: List[str], return_probabilities: bool) -> List[Dict]:
        """Internal method to process a single batch."""
        # Transform texts to padded sequences
        X = self.text_processor.transform(texts)
        
        # Get predictions from model
        # Keras models use predict() directly, which returns probabilities for sigmoid
        predictions = self.model.predict(X, verbose=0)
        
        # For Keras models with sigmoid activation
        # predictions shape: (batch_size, 1) or (batch_size,)
        predictions = predictions.flatten()
        
        # Format results
        results = []
        for idx, text in enumerate(texts):
            prob_positive = float(predictions[idx])
            
            # Apply threshold
            predicted_class = int(prob_positive >= self.threshold)
            
            # Confidence is the probability of the predicted class
            confidence = prob_positive if predicted_class == 1 else (1 - prob_positive)
            
            # Labels for binary classification
            label = "positive" if predicted_class == 1 else "negative"
            
            result = {
                'text': text,
                'predicted_class': predicted_class,
                'label': label,
                'confidence': float(confidence),
                'is_confident': confidence >= self.threshold,
                'probability_positive': prob_positive,
                'probability_negative': 1 - prob_positive
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'negative': float(1 - prob_positive),
                    'positive': float(prob_positive)
                }
            
            results.append(result)
        
        return results
    
    def predict_with_threshold(self, text: str, threshold: float = None) -> Dict:
        """
        Predict with custom confidence threshold.
        Returns 'uncertain' if confidence is below threshold.
        
        Args:
            text: Input text
            threshold: Custom threshold (uses default if None)
        
        Returns:
            Prediction result with uncertainty handling
        """
        threshold = threshold or self.threshold
        
        # Get prediction with original threshold
        original_threshold = self.threshold
        self.threshold = threshold
        result = self.predict_single(text, return_probabilities=True)
        self.threshold = original_threshold
        
        if result['confidence'] < threshold:
            result['final_prediction'] = 'uncertain'
            result['reason'] = f"Confidence {result['confidence']:.3f} below threshold {threshold}"
        else:
            result['final_prediction'] = result['label']
        
        return result
    
    def get_top_k_predictions(self, text: str, k: int = 2) -> List[Dict]:
        """
        Get top-k predictions with probabilities.
        For binary classification, k is limited to 2.
        
        Args:
            text: Input text
            k: Number of top predictions to return (max 2 for binary)
        
        Returns:
            List of top-k predictions sorted by probability
        """
        result = self.predict_single(text, return_probabilities=True)
        
        if 'probabilities' not in result:
            return [{
                'label': result['label'],
                'probability': result['confidence']
            }]
        
        probabilities = result['probabilities']
        
        # Sort by probability
        sorted_predictions = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:min(k, 2)]  # Binary classification has max 2 classes
        
        top_k = []
        for label, prob in sorted_predictions:
            top_k.append({
                'label': label,
                'probability': float(prob)
            })
        
        return top_k
    
    def predict_with_raw_output(self, texts: List[str]) -> Dict:
        """
        Get raw model outputs (useful for debugging).
        
        Args:
            texts: List of input texts
        
        Returns:
            Dictionary with raw predictions and probabilities
        """
        X = self.text_processor.transform(texts)
        
        result = {
            'texts': texts,
            'input_shape': X.shape,
            'max_tokens': self.text_processor.max_tokens
        }
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0).flatten()
        result['raw_predictions'] = predictions.tolist()
        result['probabilities_positive'] = predictions.tolist()
        result['probabilities_negative'] = (1 - predictions).tolist()
        
        # Apply threshold
        threshold_predictions = (predictions >= self.threshold).astype(int)
        result['threshold_predictions'] = threshold_predictions.tolist()
        result['threshold_used'] = self.threshold
        
        return result
    
    def evaluate_thresholds(self, text: str, thresholds: List[float] = None) -> List[Dict]:
        """
        Evaluate predictions across multiple thresholds.
        
        Args:
            text: Input text
            thresholds: List of thresholds to test
        
        Returns:
            List of predictions for each threshold
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Get base prediction
        X = self.text_processor.transform([text])
        prob_positive = float(self.model.predict(X, verbose=0).flatten()[0])
        
        results = []
        for thresh in thresholds:
            pred_class = int(prob_positive >= thresh)
            label = "positive" if pred_class == 1 else "negative"
            confidence = prob_positive if pred_class == 1 else (1 - prob_positive)
            
            results.append({
                'threshold': thresh,
                'predicted_class': pred_class,
                'label': label,
                'probability_positive': prob_positive,
                'confidence': confidence
            })
        
        return results