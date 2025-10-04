import sys
import os
import pickle

# Add src to path
sys.path.insert(0, 'src')

print("=== DEBUGGING MODEL LOADING ===")

# Test direct pickle loading
print("\n1. Testing direct pickle load...")
try:
    with open('src/models/rf_exoplanet_model_20251003_015506.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print(f'✅ Model loaded successfully')
    print(f'Model type: {type(model_data)}')
    print(f'Model classes: {getattr(model_data, "classes_", None)}')
    print(f'Model features: {getattr(model_data, "feature_names_in_", None)}')
    print(f'Model n_features: {getattr(model_data, "n_features_in_", None)}')
except Exception as e:
    print(f'❌ Error: {e}')

# Test ModelManager import
print("\n2. Testing ModelManager import...")
try:
    from models.ml_model import ModelManager
    print("✅ ModelManager imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test ModelManager initialization
print("\n3. Testing ModelManager initialization...")
try:
    manager = ModelManager()
    print("✅ ModelManager initialized")
    print(f"Available models: {len(manager.available_models)}")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)

# Test model loading
print("\n4. Testing model loading...")
try:
    print("Before loading:")
    print(f"  current_model: {manager.current_model}")
    print(f"  current_model_info: {manager.current_model_info}")
    
    result = manager.load_model('k2_random_forest')
    print(f"Load result: {result}")
    print(f"Load result type: {type(result)}")
    
    print("After loading:")
    print(f"  current_model: {manager.current_model}")
    print(f"  current_model_info: {manager.current_model_info}")
    
except Exception as e:
    print(f"❌ Loading error: {e}")
    import traceback
    traceback.print_exc()