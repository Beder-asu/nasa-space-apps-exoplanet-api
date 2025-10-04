import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Now we can import from the models module
from models.ml_model import ModelManager

def test_model_manager():
    """Test the ModelManager functionality"""
    print("Testing ModelManager...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Initialize model manager
    try:
        manager = ModelManager()
        print("✅ ModelManager initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize ModelManager: {e}")
        return
    
    # List available models
    print(f"\nAvailable models: {len(manager.available_models)}")
    for model_id, model_info in manager.available_models.items():
        print(f"  - {model_id}: {model_info['name']} ({model_info['dataset']})")
        print(f"    File: {model_info['path']}")
        print(f"    Exists: {os.path.exists(model_info['path'])}")
    
    # Try to load the first available model
    if manager.available_models:
        first_model_id = list(manager.available_models.keys())[0]
        print(f"\nTrying to load model: {first_model_id}")
        
        try:
            success = manager.load_model(first_model_id)
            if success:
                print("✅ Model loaded successfully!")
                print(f"Model info: {manager.current_model_info}")
                print(f"Model type: {type(manager.current_model)}")
                if hasattr(manager.current_model, 'feature_importances_'):
                    print(f"Feature importances available: Yes")
                else:
                    print(f"Feature importances available: No")
            else:
                print("❌ Failed to load model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No models found")
        print("Expected model files:")
        expected_files = [
            'dt_baseline_20251004T045842Z.pkl',
            'rf_exoplanet_model_20251003_015506.pkl',
            'xgb_baseline_20251003T084753Z.pkl',
            'merged_dataset_rf_model.pkl'
        ]
        models_dir = os.path.join(src_path, 'models')
        for filename in expected_files:
            filepath = os.path.join(models_dir, filename)
            exists = os.path.exists(filepath)
            print(f"  - {filename}: {'✅' if exists else '❌'}")

if __name__ == "__main__":
    test_model_manager()