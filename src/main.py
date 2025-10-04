from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add current directory to Python path to fix imports
sys.path.insert(0, os.path.dirname(__file__))

from api.endpoints import router as api_router

# Create FastAPI app
app = FastAPI(
    title="NASA Space Apps Exoplanet API",
    description="API for exoplanet classification using multiple ML models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the NASA Space Apps Exoplanet Classification API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)