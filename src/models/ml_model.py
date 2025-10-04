import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Any, Tuple, Optional
import os
from pathlib import Path

class ModelManager:
    """Manages multiple ML models for exoplanet classification"""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = Path(__file__).parent
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.current_model_info = None
        self.available_models = self._discover_models()
        
    def _discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available models in the models directory"""
        models = {}
        
        # Define model mappings based on your pickle files
        model_files = {
            'k2_random_forest': {
                'file': 'rf_exoplanet_model_20251003_015506.pkl',
                'dataset': 'K2',
                'type': 'Random Forest',
                'name': 'K2 Random Forest Model'
            },
            'k2_decision_tree': {
                'file': 'dt_baseline_20251004T045842Z.pkl',
                'dataset': 'K2',
                'type': 'Decision Trees',
                'name': 'K2 Decision Tree Model'
            },
            'k2_xgboost': {
                'file': 'xgb_baseline_20251003T084753Z.pkl',
                'dataset': 'K2',
                'type': 'XGBoost',
                'name': 'K2 XGBoost Model'
            },
            'merged_random_forest': {
                'file': 'merged_dataset_rf_model.pkl',
                'dataset': 'K2_Kepler',
                'type': 'Random Forest',
                'name': 'K2+Kepler Random Forest Model',
                'components_file': 'merged_dataset_rf_components.pkl'
            }
        }
        
        for model_id, model_info in model_files.items():
            model_path = self.models_dir / model_info['file']
            if model_path.exists():
                models[model_id] = {
                    'id': model_id,
                    'path': str(model_path),
                    'dataset': model_info['dataset'],
                    'modelType': model_info['type'],
                    'name': model_info['name'],
                    'components_path': str(self.models_dir / model_info.get('components_file', '')) if model_info.get('components_file') else None
                }
        
        return models
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return list(self.available_models.values())
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model by ID"""
        if model_id not in self.available_models:
            return False
        
        try:
            model_info = self.available_models[model_id]
            
            # Load the main model file
            if model_info['path'].endswith('.pkl'):
                with open(model_info['path'], 'rb') as f:
                    model_data = pickle.load(f)
            else:
                model_data = joblib.load(model_info['path'])
            
            # Handle different model file structures
            if isinstance(model_data, dict):
                # Model is stored in a dictionary
                self.current_model = model_data.get('model', model_data)
                self.current_model_info = {
                    'id': model_id,
                    'name': model_info['name'],
                    'dataset': model_info['dataset'],
                    'type': model_info['modelType'],
                    'features': model_data.get('features', model_data.get('feature_columns', [])),
                    'classes': getattr(model_data.get('model', model_data), 'classes_', None),
                    'feature_importance': model_data.get('feature_importance', None)
                }
            else:
                # Model is stored directly
                self.current_model = model_data
                self.current_model_info = {
                    'id': model_id,
                    'name': model_info['name'],
                    'dataset': model_info['dataset'],
                    'type': model_info['modelType'],
                    'features': [],
                    'classes': getattr(model_data, 'classes_', None),
                    'feature_importance': getattr(model_data, 'feature_importances_', None)
                }
            
            # Load components file if available (for merged dataset)
            if model_info.get('components_path') and os.path.exists(model_info['components_path']):
                with open(model_info['components_path'], 'rb') as f:
                    components = pickle.load(f)
                    if isinstance(components, dict):
                        self.current_model_info['features'] = components.get('feature_columns', [])
                        self.current_model_info['imputer'] = components.get('imputer')
                        self.current_model_info['label_encoder'] = components.get('label_encoder')
            
            # If no features found yet, try to get from the model itself
            if not self.current_model_info['features'] and hasattr(self.current_model, 'feature_names_in_'):
                self.current_model_info['features'] = list(self.current_model.feature_names_in_)
            
            return True
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on input data"""
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        try:
            # Get predictions
            predictions = self.current_model.predict(data)
            
            # Get prediction probabilities if available
            if hasattr(self.current_model, 'predict_proba'):
                probabilities = self.current_model.predict_proba(data)
                confidence = np.max(probabilities, axis=1)
            else:
                confidence = np.ones(len(predictions))  # Default confidence
            
            return predictions, confidence
            
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")
    
    def get_model_performance(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Get performance metrics for the current model"""
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        try:
            # Make predictions
            y_pred = self.current_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Format confusion matrix for API response
            classes = self.current_model_info.get('classes', ['Candidate', 'Confirmed'])
            cm_dict = {}
            for i, true_class in enumerate(classes):
                cm_dict[true_class.lower()] = {}
                for j, pred_class in enumerate(classes):
                    cm_dict[true_class.lower()][pred_class.lower()] = int(cm[i][j])
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1Score': float(f1),
                'confusionMatrix': cm_dict,
                'currentModel': {
                    'name': self.current_model_info['name'],
                    'trainingSamples': getattr(self.current_model, 'n_samples_', 0),
                    'testSamples': len(X_test),
                    'features': len(self.current_model_info.get('features', []))
                }
            }
            
        except Exception as e:
            raise ValueError(f"Performance calculation error: {e}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance for the current model"""
        if self.current_model is None or not hasattr(self.current_model, 'feature_importances_'):
            return None
        
        features = self.current_model_info.get('features', [])
        importances = self.current_model.feature_importances_
        
        if len(features) == len(importances):
            return dict(zip(features, importances.astype(float)))
        
        return None
    
    def get_training_data_info(self) -> Dict[str, Any]:
        """Get information about the training data"""
        if self.current_model is None:
            return {}
        
        features = self.current_model_info.get('features', [])
        
        # Create dummy feature types (in real implementation, you'd store this info)
        feature_types = {}
        for feature in features:
            if any(keyword in feature.lower() for keyword in ['magnitude', 'temperature', 'mass', 'radius']):
                feature_types[feature] = 'numeric'
            elif any(keyword in feature.lower() for keyword in ['flag', 'has_', 'is_']):
                feature_types[feature] = 'boolean'
            else:
                feature_types[feature] = 'numeric'  # default
        
        return {
            'features': features,
            'featureTypes': feature_types,
            'numFeatures': len(features),
            'modelType': self.current_model_info.get('type', 'Unknown'),
            'dataset': self.current_model_info.get('dataset', 'Unknown')
        }
    
    def prepare_data_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare uploaded data for prediction"""
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        expected_features = self.current_model_info.get('features', [])
        
        # Handle missing features by adding them with default values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # or np.nan, depending on your model's training
        
        # Reorder columns to match training order
        if expected_features:
            df = df[expected_features]
        
        return df