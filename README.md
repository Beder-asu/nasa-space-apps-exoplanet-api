# NASA Space Apps Exoplanet Classification API

A FastAPI application that provides machine learning model predictions for exoplanet classification. This API serves multiple trained models (Random Forest, Decision Trees, XGBoost) on K2 and merged K2+Kepler datasets.

## 🚀 Features

- **Multiple ML Models**: Random Forest, Decision Tree, and XGBoost classifiers
- **Multiple Datasets**: K2 dataset and merged K2+Kepler dataset
- **Model Performance**: Get accuracy, precision, recall, F1-score, and confusion matrix
- **CSV Upload**: Upload CSV files for batch predictions
- **Visualization Data**: Get data for plotting uploaded vs training data
- **Dataset Information**: Explore training data features and statistics

## 📁 Project Structure

```
nasa-space-apps-api/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py           # API route definitions
│   └── models/
│       ├── __init__.py
│       ├── ml_model.py            # ModelManager class
│       ├── dt_baseline_*.pkl      # Decision Tree model
│       ├── rf_exoplanet_*.pkl     # Random Forest model (K2)
│       ├── xgb_baseline_*.pkl     # XGBoost model
│       ├── merged_dataset_rf_model.pkl        # RF model (K2+Kepler)
│       └── merged_dataset_rf_components.pkl   # RF components
├── requirements.txt               # Python dependencies
├── railway.toml                  # Railway deployment config
├── Procfile                      # Process file for deployment
├── test_models.py               # Local testing script
└── README.md                    # This file
```

## 🛠️ Available Models

| Model ID | Dataset | Type | Description |
|----------|---------|------|-------------|
| `k2_random_forest` | K2 | Random Forest | K2 dataset Random Forest classifier |
| `k2_decision_tree` | K2 | Decision Trees | K2 dataset Decision Tree classifier |
| `k2_xgboost` | K2 | XGBoost | K2 dataset XGBoost classifier |
| `merged_random_forest` | K2+Kepler | Random Forest | Merged dataset Random Forest classifier |

## 📋 API Endpoints

### Get Model Performance
```http
GET /api/model-performance
```
Returns accuracy, precision, recall, F1-score, and confusion matrix for the current model.

### Get Available Models
```http
GET /api/models
```
Returns list of all available models with their metadata.

### Update Model Settings
```http
POST /api/settings
Content-Type: application/json

{
  "selectedModelId": "k2_random_forest"
}
```

### Make Predictions
```http
POST /api/predict
Content-Type: multipart/form-data

file: [CSV file]
modelId: "k2_random_forest"
```

### Get Dashboard Data
```http
GET /api/dashboard
```
Returns classification status and visualization data.

### Get Dataset Information
```http
GET /api/dataset-info
```
Returns training data features, types, and feature importance.

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd nasa-space-apps-api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Test model loading**
```bash
python test_models.py
```

4. **Run the API**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API**
- API Documentation: http://localhost:8000/docs
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health

## 🚀 Railway Deployment

### Prerequisites
- Railway account
- Railway CLI (optional)

### Deployment Steps

1. **Connect to Railway**
   - Go to [Railway](https://railway.app)
   - Create new project
   - Connect your GitHub repository

2. **Configure Environment**
   Railway will automatically detect the configuration from `railway.toml`

3. **Deploy**
   - Push your code to the connected repository
   - Railway will automatically build and deploy

### Environment Variables
No additional environment variables are required. The app uses:
- `PORT`: Automatically set by Railway

## 📊 Model Files Required

Make sure these pickle files are in the `src/models/` directory:
- `dt_baseline_*.pkl` - Decision Tree model
- `rf_exoplanet_*.pkl` - Random Forest model (K2)
- `xgb_baseline_*.pkl` - XGBoost model  
- `merged_dataset_rf_model.pkl` - Random Forest model (K2+Kepler)
- `merged_dataset_rf_components.pkl` - Model components

## 🧪 Testing

### Test Model Loading
```bash
python test_models.py
```

### Test API Endpoints
```bash
# Start the server
uvicorn src.main:app --reload

# Test in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/api/models
curl http://localhost:8000/api/model-performance
```

## 📈 Usage Examples

### Python Client Example
```python
import requests
import pandas as pd

# Get available models
response = requests.get("http://your-api-url/api/models")
models = response.json()["models"]

# Set model
requests.post("http://your-api-url/api/settings", 
             json={"selectedModelId": "k2_random_forest"})

# Upload CSV for prediction
with open("test_data.csv", "rb") as f:
    files = {"file": f}
    data = {"modelId": "k2_random_forest"}
    response = requests.post("http://your-api-url/api/predict", 
                           files=files, data=data)
    predictions = response.json()
```

### JavaScript/Frontend Example
```javascript
// Get model performance
const performance = await fetch('/api/model-performance').then(r => r.json());

// Upload CSV file
const formData = new FormData();
formData.append('file', csvFile);
formData.append('modelId', 'k2_random_forest');

const predictions = await fetch('/api/predict', {
    method: 'POST',
    body: formData
}).then(r => r.json());
```

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**
   - Check that all `.pkl` files are in the correct directory
   - Run `python test_models.py` to verify

2. **Memory issues on Railway**
   - Models are loaded on-demand to save memory
   - Consider model compression for production

3. **Large file uploads**
   - Default limit is 10MB
   - Implement chunked uploads for larger files

## 📝 License

This project is part of NASA Space Apps Challenge 2025.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request