"""
Enhanced FastAPI Application for Penguin Species Classification
Features: Health check, single/batch predictions, model info, enhanced error handling
Dataset: Palmer Penguins (Antarctic penguin species)
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List
import pickle
import numpy as np
import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🐧 Enhanced Penguin Species Classification API",
    description="FastAPI service for Antarctic penguin species prediction using Random Forest. Classifies penguins into Adelie, Chinstrap, or Gentoo species based on physical measurements.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Penguin species mapping (will be loaded from metrics)
PENGUIN_SPECIES = {
    0: "Adelie",
    1: "Chinstrap", 
    2: "Gentoo"
}

# Global variables for model and metrics
model = None
model_metrics = None

# Pydantic models for request/response validation
class PenguinFeatures(BaseModel):
    """Penguin physical measurements"""
    bill_length_mm: float = Field(..., ge=0, le=100, description="Bill length in millimeters")
    bill_depth_mm: float = Field(..., ge=0, le=50, description="Bill depth in millimeters")
    flipper_length_mm: float = Field(..., ge=0, le=300, description="Flipper length in millimeters")
    body_mass_g: float = Field(..., ge=0, le=10000, description="Body mass in grams")
    
    @validator('*')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('All measurements must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0
            }
        }

class BatchPenguinFeatures(BaseModel):
    """Batch of penguin samples"""
    samples: List[PenguinFeatures] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "bill_length_mm": 39.1,
                        "bill_depth_mm": 18.7,
                        "flipper_length_mm": 181.0,
                        "body_mass_g": 3750.0
                    },
                    {
                        "bill_length_mm": 46.5,
                        "bill_depth_mm": 17.9,
                        "flipper_length_mm": 192.0,
                        "body_mass_g": 3500.0
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Single prediction response"""
    species: str
    species_id: int
    confidence: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_samples: int
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    dataset: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: str
    train_size: int
    test_size: int
    n_features: int
    feature_names: List[str]
    n_classes: int

# Startup event to load model
@app.on_event("startup")
async def load_model():
    """Load the trained model and metrics on startup"""
    global model, model_metrics, PENGUIN_SPECIES
    
    try:
        model_path = './model/penguin_rf_model.pkl'
        metrics_path = './model/model_metrics.json'
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metrics
        if os.path.exists(metrics_path):
            logger.info(f"Loading metrics from {metrics_path}")
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
                # Update species mapping from metrics if available
                if 'species_mapping' in model_metrics:
                    PENGUIN_SPECIES = {v: k for k, v in model_metrics['species_mapping'].items()}
        
        logger.info("✓ Model and metrics loaded successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.warning("Please run train.py first to create the model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health and model status
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_type="RandomForestClassifier" if model is not None else "None",
        timestamp=datetime.now().isoformat()
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "🐧 Enhanced Penguin Species Classification API",
        "version": "2.0.0",
        "dataset": "Palmer Penguins Dataset",
        "description": "Classify Antarctic penguins (Adelie, Chinstrap, Gentoo) based on physical measurements",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }

# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model performance metrics and information
    """
    if model_metrics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metrics not available"
        )
    
    return ModelInfoResponse(**model_metrics)

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PenguinFeatures):
    """
    Predict penguin species for a single sample
    
    Predicts one of three Antarctic penguin species:
    - Adelie: Smallest of the three, found throughout Antarctic coast
    - Chinstrap: Named for thin black band under head
    - Gentoo: Largest, with bright orange-red bill
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained."
        )
    
    try:
        # Prepare input
        input_data = np.array([[
            features.bill_length_mm,
            features.bill_depth_mm,
            features.flipper_length_mm,
            features.body_mass_g
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction])
        
        logger.info(f"Prediction made: {PENGUIN_SPECIES[prediction]} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            species=PENGUIN_SPECIES[prediction],
            species_id=int(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchPenguinFeatures):
    """
    Predict penguin species for multiple samples
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained."
        )
    
    try:
        # Prepare batch input
        input_data = np.array([[
            sample.bill_length_mm,
            sample.bill_depth_mm,
            sample.flipper_length_mm,
            sample.body_mass_g
        ] for sample in batch.samples])
        
        # Make batch predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Format responses
        responses = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            responses.append(PredictionResponse(
                species=PENGUIN_SPECIES[pred],
                species_id=int(pred),
                confidence=float(probs[pred]),
                timestamp=datetime.now().isoformat()
            ))
        
        logger.info(f"Batch prediction completed for {len(batch.samples)} samples")
        
        return BatchPredictionResponse(
            predictions=responses,
            total_samples=len(batch.samples),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )