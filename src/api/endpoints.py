from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from models.ml_model import ModelManager
import pandas as pd
import numpy as np
import io
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize model manager
model_manager = ModelManager()

# Pydantic models for request/response
class ModelSettings(BaseModel):
    selectedModelId: str

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    visualizationData: Dict[str, Any]

@router.get("/api/model-performance")
async def get_model_performance():
    """Get performance metrics for the current model"""
    try:
        if model_manager.current_model is None:
            # Load a default model if none is loaded
            available_models = model_manager.get_available_models()
            if available_models:
                model_manager.load_model(available_models[0]['id'])
            else:
                raise HTTPException(status_code=404, detail="No models available")
        
        # For demo purposes, return mock performance data
        # In production, you'd use actual test data
        performance_data = {
            "accuracy": 0.9287,
            "precision": 0.9134,
            "recall": 0.9456,
            "f1Score": 0.9293,
            "confusionMatrix": {
                "confirmed": {"confirmed": 1542, "candidate": 67},
                "candidate": {"confirmed": 123, "candidate": 952}
            },
            "currentModel": {
                "name": model_manager.current_model_info['name'] if model_manager.current_model_info else "Unknown",
                "trainingSamples": 6854,
                "testSamples": 1714,
                "features": len(model_manager.current_model_info.get('features', [])) if model_manager.current_model_info else 0
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    try:
        models = model_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/settings")
async def update_model_settings(settings: ModelSettings):
    """Update the current model"""
    try:
        success = model_manager.load_model(settings.selectedModelId)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/predict")
async def make_prediction(file: UploadFile = File(...), modelId: str = Form(...)):
    """Make predictions on uploaded CSV data"""
    try:
        # Load the specified model
        if not model_manager.load_model(modelId):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Prepare data for prediction
        prepared_data = model_manager.prepare_data_for_prediction(df)
        
        # Make predictions
        predictions, confidence = model_manager.predict(prepared_data)
        
        # Format predictions for response
        formatted_predictions = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            # Map prediction to human-readable format
            disposition = "Confirmed" if pred == 1 else "Candidate"
            
            prediction_data = {
                "id": i + 1,
                "disposition": disposition,
                "confidence": float(conf),
                "orbitalPeriod": float(df.iloc[i].get('orbital_period', 0)) if 'orbital_period' in df.columns else 0,
                "planetaryRadius": float(df.iloc[i].get('planet_radius_earth', 0)) if 'planet_radius_earth' in df.columns else 0,
                "transitDuration": float(df.iloc[i].get('transit_duration', 0)) if 'transit_duration' in df.columns else 0,
                "transitDepth": float(df.iloc[i].get('transit_depth', 0)) if 'transit_depth' in df.columns else 0
            }
            formatted_predictions.append(prediction_data)
        
        # Calculate statistics
        confirmed_count = int(np.sum(predictions == 1))
        candidate_count = int(np.sum(predictions == 0))
        avg_confidence = float(np.mean(confidence))
        
        statistics = {
            "totalPredictions": len(predictions),
            "confirmed": confirmed_count,
            "candidates": candidate_count,
            "averageConfidence": avg_confidence
        }
        
        # Create visualization data (sample points for plotting)
        uploaded_viz_data = []
        for i, pred in enumerate(predictions):
            if i < 100:  # Limit to first 100 points for performance
                uploaded_viz_data.append({
                    "x": float(df.iloc[i].get('orbital_period', np.random.uniform(1, 100))),
                    "y": float(df.iloc[i].get('planet_radius_earth', np.random.uniform(0.5, 5))),
                    "label": "Confirmed" if pred == 1 else "Candidate"
                })
        
        # Mock training data for comparison (in production, store this with the model)
        training_viz_data = [
            {"x": float(np.random.uniform(1, 100)), "y": float(np.random.uniform(0.5, 5)), "label": "Confirmed"}
            for _ in range(50)
        ] + [
            {"x": float(np.random.uniform(1, 100)), "y": float(np.random.uniform(0.5, 5)), "label": "Candidate"}
            for _ in range(50)
        ]
        
        visualization_data = {
            "uploadedData": uploaded_viz_data,
            "trainingData": training_viz_data
        }
        
        return {
            "predictions": formatted_predictions,
            "statistics": statistics,
            "visualizationData": visualization_data
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/dashboard")
async def get_dashboard_data():
    """Get dashboard data including classification status and scatter plot data"""
    try:
        if model_manager.current_model is None:
            # Load a default model if none is loaded
            available_models = model_manager.get_available_models()
            if available_models:
                model_manager.load_model(available_models[0]['id'])
        
        # Mock dashboard data (in production, this would come from your actual data)
        classification_status = {
            "disposition": "Confirmed",
            "confidence": 0.94,
            "modelUsed": model_manager.current_model_info['name'] if model_manager.current_model_info else "Unknown"
        }
        
        # Generate sample scatter plot data
        scatter_plot_data = []
        for _ in range(200):
            scatter_plot_data.append({
                "orbitalPeriod": float(np.random.uniform(1, 400)),
                "planetaryRadius": float(np.random.uniform(0.3, 15)),
                "disposition": np.random.choice(["Confirmed", "Candidate"])
            })
        
        statistics = {
            "totalConfirmed": 3426,
            "candidates": 4208
        }
        
        return {
            "classificationStatus": classification_status,
            "scatterPlotData": scatter_plot_data,
            "statistics": statistics
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/dataset-info")
async def get_dataset_info():
    """Get information about the training dataset"""
    try:
        if model_manager.current_model is None:
            available_models = model_manager.get_available_models()
            if available_models:
                model_manager.load_model(available_models[0]['id'])
        
        dataset_info = model_manager.get_training_data_info()
        
        # Add feature importance if available
        feature_importance = model_manager.get_feature_importance()
        if feature_importance:
            dataset_info['featureImportance'] = feature_importance
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(model_manager.available_models)}