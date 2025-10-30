import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model package
def load_model_package(model_path='models/lstm_model_package.pkl'):
    """Load the complete model package from pickle file"""
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    return model_package

# Preprocess text for prediction
def preprocess_text(text, tokenizer, max_tokens):
    """Convert text to tokenized and padded sequence"""
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences to match training length
    padded = pad_sequences(sequences, maxlen=max_tokens, padding='post', truncating='post')
    return padded

# Make prediction with diagnostics
def predict_text(text, model_package, show_tokens=False):
    """
    Predict on a single text input
    
    Args:
        text: String to classify
        model_package: Dictionary containing model and preprocessing info
        show_tokens: Whether to show tokenization details
    
    Returns:
        prediction: Binary prediction (0 or 1)
        probability: Confidence score (0-1)
    """
    model = model_package['model']
    tokenizer = model_package['tokenizer']
    max_tokens = model_package['max_tokens']
    threshold = model_package['threshold']
    
    # Preprocess the text
    processed_text = preprocess_text(text, tokenizer, max_tokens)
    
    if show_tokens:
        sequences = tokenizer.texts_to_sequences([text])
        print(f"  Tokens: {sequences[0][:20]}...")  # Show first 20 tokens
        print(f"  Total tokens: {len(sequences[0])}")
        print(f"  Non-zero tokens in padded: {np.count_nonzero(processed_text)}")
    
    # Get prediction probability
    prob = model.predict(processed_text, verbose=0)[0][0]
    
    # Apply threshold
    prediction = 1 if prob >= threshold else 0
    
    return prediction, float(prob)

# Test on multiple texts
def test_multiple_texts(texts, model_package, show_tokens=False):
    """
    Test model on multiple text samples
    
    Args:
        texts: List of strings to classify
        model_package: Dictionary containing model and preprocessing info
        show_tokens: Whether to show tokenization details
    
    Returns:
        results: List of tuples (text, prediction, probability)
    """
    results = []
    for text in texts:
        pred, prob = predict_text(text, model_package, show_tokens)
        results.append((text, pred, prob))
    return results

# Main testing function
if __name__ == "__main__":
    # Load the model
    print("Loading model package...")
    model_package = load_model_package('models/lstm_model_package.pkl')
    
    print(f"Model type: {model_package['model_type']}")
    print(f"Max tokens: {model_package['max_tokens']}")
    print(f"Threshold: {model_package['threshold']}")
    print(f"Vocab size: {model_package['vocab_size']}")
    print("-" * 70)
    
    # Movie review test samples (more appropriate for this model)
    movie_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the story kept me engaged from start to finish. Highly recommend!",
        "Waste of time. The plot was boring, acting was terrible, and I couldn't wait for it to end. Don't bother watching.",
        "One of the best films I've seen this year. The cinematography was beautiful and the performances were outstanding.",
        "Horrible movie. Bad script, bad acting, bad everything. I want my money back.",
        "An okay film. Nothing special but not terrible either. Some good moments but overall forgettable.",
        "Absolutely loved it! The director did an amazing job. Every scene was perfect. A masterpiece!",
        "This film is a disaster. Poor direction, wooden acting, and a nonsensical plot. Avoid at all costs.",
        "Pretty good movie. The story was interesting and the cast did a decent job. Worth watching.",
        "What a disappointment. I had high expectations but the movie fell flat. Very boring and predictable.",
        "Brilliant! This is cinema at its finest. Every aspect of this film was executed flawlessly. Must watch!"
    ]
    
    print("\nTesting on Movie Reviews:")
    print("=" * 70)
    
    results = test_multiple_texts(movie_reviews, model_package, show_tokens=False)
    
    positive_count = sum(1 for _, pred, _ in results if pred == 1)
    negative_count = len(results) - positive_count
    
    for i, (text, pred, prob) in enumerate(results, 1):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        emoji = "😊" if pred == 1 else "😞"
        
        # Color-code confidence
        if prob > 0.7 or prob < 0.3:
            confidence_level = "High"
        elif prob > 0.6 or prob < 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        print(f"\n{i}. {emoji} {sentiment} (Confidence: {confidence_level})")
        print(f"   Probability: {prob:.4f}")
        print(f"   Review: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    print("\n" + "=" * 70)
    print(f"Summary: {positive_count} Positive, {negative_count} Negative")
    print("=" * 70)
    
    # Interactive mode with better guidance
    print("\n--- Interactive Mode ---")
    print("This model was trained on movie reviews.")
    print("For best results, input movie review-style text.")
    print("Enter text to classify (or 'quit' to exit):")
    
    while True:
        print("\n" + "-" * 70)
        user_input = input("Your review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        if len(user_input.split()) < 5:
            print("⚠ Warning: Very short text. Model works better with longer reviews.")
        
        print()
        pred, prob = predict_text(user_input, model_package, show_tokens=True)
        
        sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        print(f"\nResult: {sentiment}")
        print(f"Confidence: {prob:.4f}")
        
        if 0.4 < prob < 0.6:
            print("⚠ Note: Low confidence - the model is uncertain about this text.")