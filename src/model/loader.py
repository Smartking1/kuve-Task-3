import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of scikit-learn models from .pkl files."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def load_model_package(self, package_path: str) -> Tuple[Any, Any, float]:
        """
        Load model package from .pkl file.
        
        Args:
            package_path: Path to the .pkl file
        
        Returns:
            Tuple of (model, vectorizer, threshold)
        """
        package_path = Path(package_path)
        
        if not package_path.exists():
            raise FileNotFoundError(f"Model package file not found: {package_path}")
        
        logger.info(f"Loading model package from {package_path}")
        
        try:
            # Load the package
            package = joblib.load(package_path)
            
            # Handle different package formats
            if isinstance(package, dict):
                # Package contains separate components
                model = package.get("model")
                vectorizer = package.get("vectorizer") or package.get("tokenizer")
                threshold = package.get("threshold", self.config.get('threshold', 0.5))
                
                if model is None:
                    raise ValueError("Model not found in package")
                if vectorizer is None:
                    raise ValueError("Vectorizer/Tokenizer not found in package")
                
                logger.info(f"Loaded model: {type(model).__name__}")
                logger.info(f"Loaded vectorizer: {type(vectorizer).__name__}")
                logger.info(f"Decision threshold: {threshold}")
                
                # Log additional info if available
                if hasattr(model, 'classes_'):
                    logger.info(f"Model classes: {model.classes_}")
                
                return model, vectorizer, threshold
            else:
                # Package is just the model
                logger.warning("Package is not a dictionary. Assuming it's just the model.")
                model = package
                vectorizer = None
                threshold = self.config.get('threshold', 0.5)
                return model, vectorizer, threshold
                
        except Exception as e:
            logger.error(f"Failed to load model package: {e}")
            raise
    
    @staticmethod
    def get_package_info(package_path: str) -> Dict[str, Any]:
        """
        Extract information from a model package without fully loading it.
        
        Args:
            package_path: Path to the .pkl file
        
        Returns:
            Dictionary with package information
        """
        package_path = Path(package_path)
        
        info = {
            'file_size_mb': package_path.stat().st_size / (1024 * 1024),
            'exists': package_path.exists()
        }
        
        try:
            package = joblib.load(package_path)
            
            if isinstance(package, dict):
                info['keys'] = list(package.keys())
                info['package_type'] = 'dictionary'
                
                if 'model' in package:
                    model = package['model']
                    info['model_type'] = type(model).__name__
                    
                    if hasattr(model, 'classes_'):
                        info['num_classes'] = len(model.classes_)
                        info['classes'] = model.classes_.tolist()
                
                if 'vectorizer' in package:
                    vectorizer = package['vectorizer']
                    info['vectorizer_type'] = type(vectorizer).__name__
                    
                    if hasattr(vectorizer, 'vocabulary_'):
                        info['vocab_size'] = len(vectorizer.vocabulary_)
                
                if 'threshold' in package:
                    info['threshold'] = package['threshold']
            else:
                info['package_type'] = 'single_object'
                info['object_type'] = type(package).__name__
        
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def save_model_package(model, vectorizer, threshold: float, save_path: str):
        """
        Save model, vectorizer, and threshold as a package.
        
        Args:
            model: Trained scikit-learn model
            vectorizer: Fitted vectorizer
            threshold: Decision threshold
            save_path: Path to save the package
        """
        package = {
            "model": model,
            "vectorizer": vectorizer,
            "threshold": threshold
        }
        
        joblib.dump(package, save_path)
        logger.info(f"Model package saved to {save_path}")
        
        # Log package info
        file_size = Path(save_path).stat().st_size / (1024 * 1024)
        logger.info(f"Package size: {file_size:.2f} MB")