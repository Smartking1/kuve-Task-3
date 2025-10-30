import re
import nltk
from typing import List, Union
import numpy as np

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords


class TextProcessor:
    """Handles text preprocessing for scikit-learn models."""
    
    def __init__(self, config: dict, vectorizer=None):
        self.config = config
        self.lowercase = config.get('lowercase', True)
        self.remove_special_chars = config.get('remove_special_chars', False)
        self.remove_stopwords = config.get('remove_stopwords', False)
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        self.vectorizer = vectorizer
    
    def set_vectorizer(self, vectorizer):
        """Set the vectorizer (loaded from model package)."""
        self.vectorizer = vectorizer
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special_chars:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.remove_stopwords:
            words = text.split()
            text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text
    
    def preprocess(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Preprocess text or list of texts."""
        if isinstance(text, str):
            return self.clean_text(text)
        else:
            return [self.clean_text(t) for t in text]
    
    def transform(self, texts: Union[str, List[str]]):
        """Transform texts using the loaded vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded. Cannot transform texts.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean texts
        cleaned_texts = self.preprocess(texts)

        # Transform using vectorizer or tokenizer
        if hasattr(self.vectorizer, "transform"):
            X = self.vectorizer.transform(cleaned_texts)
        elif hasattr(self.vectorizer, "texts_to_sequences"):
            X = self.vectorizer.texts_to_sequences(cleaned_texts)
            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                # Use maxlen from config if available, else default to 100
                maxlen = self.config.get('maxlen', 100)
                X = pad_sequences(X, maxlen=maxlen, padding='post')
            except ImportError:
                raise ImportError("pad_sequences not available. Please install tensorflow.")
        else:
            raise AttributeError("Vectorizer/Tokenizer does not have a supported transform method (transform or texts_to_sequences)")
        return X