import argparse
import sys
import yaml
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.text_processor import TextProcessor
from src.model.loader import ModelLoader
from src.inference.predictor import TextClassificationPredictor


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    # If config_path is not absolute, make it relative to project root
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent.parent.parent / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_predictor(config_path: str = "config.yaml"):
    """Initialize predictor with model and preprocessor."""
    print("Loading configuration...")
    config = load_config(config_path)
    
    print("Loading model package...")
    # Load model package
    model_loader = ModelLoader(config['model'])
    # Resolve model path relative to project root
    model_path = Path(__file__).parent.parent.parent / config['model']['package_path']
    model, vectorizer, threshold = model_loader.load_model_package(
        str(model_path)
    )
    
    print(f"Model: {type(model).__name__}")
    print(f"Vectorizer: {type(vectorizer).__name__}")
    print(f"Threshold: {threshold}")
    
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
    
    # Initialize text processor with vectorizer
    text_processor = TextProcessor(config['preprocessing'], vectorizer=vectorizer)
    
    # Initialize predictor
    predictor = TextClassificationPredictor(
        model=model,
        text_processor=text_processor,
        threshold=threshold,
        config=config['inference']
    )
    
    return predictor


def predict_single(predictor, text: str, show_probs: bool = False, top_k: int = None):
    """Predict for a single text."""
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"{'='*60}")
    
    if top_k:
        top_predictions = predictor.get_top_k_predictions(text, k=top_k)
        print(f"\nTop-{top_k} Predictions:")
        for i, pred in enumerate(top_predictions, 1):
            print(f"  {i}. {pred['label']:<15} → {pred['probability']:.4f}")
    else:
        result = predictor.predict_single(text, return_probabilities=show_probs)
        print(f"\nPredicted Class: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if 'probability_positive' in result:
            print(f"Probability (positive class): {result['probability_positive']:.4f}")
            print(f"Applied threshold: {predictor.threshold}")
        
        if show_probs and 'probabilities' in result:
            print("\nClass Probabilities:")
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {label:<15} → {prob:.4f}")


def predict_from_file(predictor, file_path: str, show_probs: bool = False, output_file: str = None):
    """Predict for texts in a file (one per line)."""
    print(f"\nReading texts from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts to classify")
    print("Processing...")
    
    results = predictor.predict_batch(texts, return_probabilities=show_probs)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        text_preview = result['text'][:70] + "..." if len(result['text']) > 70 else result['text']
        print(f"\n{i}. Text: {text_preview}")
        print(f"   Predicted: {result['label']} (confidence: {result['confidence']:.4f})")
        
        if 'probability_positive' in result:
            print(f"   Prob(positive): {result['probability_positive']:.4f}")
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


def predict_interactive(predictor):
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter text to classify (or 'quit' to exit)")
    print("Commands:")
    print("  'quit' or 'q' - Exit")
    print("  'info' - Show model information")
    print("="*60 + "\n")
    
    while True:
        try:
            text = input("\n📝 Text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if text.lower() == 'info':
                print(f"\nModel Type: {type(predictor.model).__name__}")
                print(f"Threshold: {predictor.threshold}")
                if predictor.classes_ is not None:
                    print(f"Classes: {predictor.classes_}")
                continue
            
            if not text:
                print("⚠️  Please enter some text.\n")
                continue
            
            result = predictor.predict_single(text, return_probabilities=True)
            
            print(f"\n✨ Predicted Class: {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.4f}")
            
            if 'probability_positive' in result:
                print(f"📈 Prob(positive): {result['probability_positive']:.4f}")
            
            if 'probabilities' in result:
                print("\n📋 All Probabilities:")
                for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    bar = '█' * int(prob * 20)
                    print(f"  {label:<12} {bar} {prob:.4f}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Text Classification CLI for scikit-learn models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single text
  python predict_cli.py --text "This property has excellent location"
  
  # Predict with probabilities
  python predict_cli.py --text "This property has excellent location" --probs
  
  # Get top-k predictions
  python predict_cli.py --text "Sample text" --top-k 3
  
  # Predict from file
  python predict_cli.py --file texts.txt --output results.json
  
  # Interactive mode
  python predict_cli.py --interactive
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Single text to classify'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing texts to classify (one per line)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for batch predictions (JSON format)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--probs', '-p',
        action='store_true',
        help='Show class probabilities'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        help='Show top-k predictions'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if at least one input method is specified
    if not any([args.text, args.file, args.interactive]):
        parser.print_help()
        sys.exit(1)
    
    # Initialize predictor
    print("\n🚀 Initializing model...")
    try:
        predictor = initialize_predictor(args.config)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Run appropriate prediction mode
    try:
        if args.interactive:
            predict_interactive(predictor)
        elif args.text:
            predict_single(predictor, args.text, args.probs, args.top_k)
        elif args.file:
            predict_from_file(predictor, args.file, args.probs, args.output)
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()