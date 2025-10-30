import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.text_processor import TextProcessor
#from kuve-3.src.preprocessing.text_processor import TextProcessor
from src.model.loader import ModelLoader


def test_model_loading():
    """Test if model can be loaded successfully."""
    print("=" * 70)
    print("TESTING SCIKIT-LEARN MODEL LOADING")
    print("=" * 70)
    
    # Load config
    print("\n1️⃣  Loading configuration...")
    try:
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ Configuration loaded")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Get package info
    print("\n2️⃣  Analyzing model package...")
    try:
        model_loader = ModelLoader(config['model'])
        # Resolve model path relative to project root
        model_path = Path(__file__).parent.parent.parent / config['model']['package_path']
        info = model_loader.get_package_info(str(model_path))
        
        print(f"   📦 Package type: {info.get('package_type', 'Unknown')}")
        print(f"   💾 File size: {info.get('file_size_mb', 0):.2f} MB")
        
        if 'keys' in info:
            print(f"   🔑 Package keys: {info['keys']}")
        
        if 'model_type' in info:
            print(f"   🤖 Model type: {info['model_type']}")
        
        if 'vectorizer_type' in info:
            print(f"   📝 Vectorizer type: {info['vectorizer_type']}")
        
        if 'num_classes' in info:
            print(f"   🎯 Number of classes: {info['num_classes']}")
        
        if 'classes' in info:
            print(f"   🏷️  Class labels: {info['classes']}")
        
        if 'threshold' in info:
            print(f"   ⚖️  Decision threshold: {info['threshold']}")
        
        if 'vocab_size' in info:
            print(f"   📚 Vocabulary size: {info['vocab_size']:,}")
        
        print("   ✅ Package analysis complete")
        
    except Exception as e:
        print(f"   ❌ Failed to analyze package: {e}")
        return False
    
    # Load model package
    print("\n3️⃣  Loading model package...")
    try:
        model, vectorizer, threshold = model_loader.load_model_package(
            str(model_path)  # Use the already resolved model_path
        )
        
        print(f"   ✅ Model loaded: {type(model).__name__}")
        print(f"   ✅ Vectorizer loaded: {type(vectorizer).__name__}")
        print(f"   ✅ Threshold: {threshold}")
        
        # Model info
        if hasattr(model, 'classes_'):
            print(f"   🏷️  Classes: {model.classes_}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"   📊 Features: {model.n_features_in_:,}")
        
        # Vectorizer info
        if hasattr(vectorizer, 'vocabulary_'):
            print(f"   📚 Vocabulary size: {len(vectorizer.vocabulary_):,}")
        
    except Exception as e:
        print(f"   ❌ Failed to load model package: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize text processor
    print("\n4️⃣  Initializing text processor...")
    try:
        text_processor = TextProcessor(config['preprocessing'], vectorizer=vectorizer)
        print("   ✅ Text processor initialized")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize text processor: {e}")
        return False
    
    # Test inference
    print("\n5️⃣  Testing inference...")
    test_texts = [
        "This property has excellent location and amenities.",
        "The house requires significant renovation work.",
        "Amazing value for money, highly recommended!",
        "rubbish movie with bad actors and directors",
        "This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 or 8 years were brilliant, but things dropped off after that. By 1990, the show was not really funny anymore, and it's continued its decline further to the complete waste of time it is today.<br /><br />It's truly disgraceful how far this show has fallen. The writing is painfully bad, the performances are almost as bad - if not for the mildly entertaining respite of the guest-hosts, this show probably wouldn't still be on the air. I find it so hard to believe that the same creator that hand-selected the original cast also chose the band of hacks that followed. How can one recognize such brilliance and then see fit to replace it with such mediocrity? I felt I must give 2 stars out of respect for the original cast that made this show such a huge success. As it is now, the show is just awful. I can't believe it's still on the air."
    ]
    
    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: {text}")
            
            # Transform text
            X = text_processor.transform(text)
            print(f"   ➡️  Feature shape: {X.shape}")
            
            # Predict
            prediction = model.predict(X)[0]
            print(f"   🎯 Prediction: {prediction}")
            
            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                print(f"   📊 Probabilities: {proba}")
                if len(proba) == 2:
                    prob_positive = proba[1]
                    threshold_pred = int(prob_positive >= threshold)
                    print(f"   ⚖️  Prob(positive): {prob_positive:.4f}")
                    print(f"   🎯 Threshold prediction: {threshold_pred}")
            elif hasattr(model, "predict"):
                proba = model.predict(X)
                # If output is probability (Keras), handle accordingly
                if isinstance(proba, np.ndarray) and proba.shape[-1] == 1:
                    prob_positive = proba[0][0]
                    threshold_pred = int(prob_positive >= threshold)
                    print(f"   ⚖️  Prob(positive): {prob_positive:.4f}")
                    print(f"   🎯 Threshold prediction: {threshold_pred}")
                else:
                    print(f"   📊 Model predict output: {proba}")
            else:
                print("   ℹ️  Model doesn't support probability prediction")
            
            print("   ✅ Inference successful")
            
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch prediction
    print("\n6️⃣  Testing batch prediction...")
    try:
        X_batch = text_processor.transform(test_texts)
        predictions = model.predict(X_batch)
        
        print(f"   📦 Batch shape: {X_batch.shape}")
        print(f"   🎯 Predictions: {predictions}")
        
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_batch)
            print(f"   📊 Probabilities shape: {probas.shape}")
            if probas.shape[1] == 2:
                prob_positive = probas[:, 1]
                threshold_preds = (prob_positive >= threshold).astype(int)
                print(f"   ⚖️  Prob(positive): {prob_positive}")
                print(f"   🎯 Threshold predictions: {threshold_preds}")
        elif hasattr(model, "predict"):
            probas = model.predict(X_batch)
            if isinstance(probas, np.ndarray) and probas.ndim == 2 and probas.shape[1] == 1:
                prob_positive = probas[:, 0]
                threshold_preds = (prob_positive >= threshold).astype(int)
                print(f"   ⚖️  Prob(positive): {prob_positive}")
                print(f"   🎯 Threshold predictions: {threshold_preds}")
            else:
                print(f"   📊 Model batch predict output: {probas}")
        
        print("   ✅ Batch prediction successful")
        
    except Exception as e:
        print(f"   ❌ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour model is ready to use. You can now:")
    print("  • Start the API: uvicorn src.api.app:app --reload")
    print("  • Use CLI: python scripts/predict_cli.py --interactive")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_model_loading()
    
    if not success:
        print("\n❌ Tests failed. Please check your model package and configuration.")
        sys.exit(1)
    
    sys.exit(0)