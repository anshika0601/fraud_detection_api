"""
FastAPI production server for fraud detection.
Loads the best model (XGBoost/LightGBM) and serves real-time predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "fraud_detector_xgb_v1.pkl")
)
# Fallback to the trained XGBoost artifact or legacy model if the preferred model is unavailable
FALLBACK_MODEL_PATH = os.getenv(
    "FALLBACK_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "fraud_detector_v1.pkl")
)
LEGACY_LIGHTGBM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_best_model.pkl")

# ----------------------------------------------------------------------
# Pydantic schemas (request/response validation)
# ----------------------------------------------------------------------
class Transaction(BaseModel):
    """Single transaction features (exactly 30 features: V1-V28 + Amount + Time)"""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    @validator('Amount')
    def amount_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }


class PredictionRequest(BaseModel):
    """Request can contain one or multiple transactions."""
    transactions: List[Transaction]

    @validator('transactions')
    def not_empty(cls, v):
        if not v:
            raise ValueError('At least one transaction required')
        return v


class PredictionResponse(BaseModel):
    """Prediction response for a batch of transactions."""
    predictions: List[int]          # 0 = legitimate, 1 = fraud
    probabilities: List[float]      # fraud probability (0-1)
    latency_ms: float               # inference time in milliseconds
    model_version: str              # which model was used
    timestamp: str                  # ISO format timestamp


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str


# ----------------------------------------------------------------------
# FastAPI app initialization
# ----------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud prediction using XGBoost/LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Model loading (once at startup)
# ----------------------------------------------------------------------
model = None
model_version = None
loaded_model_path = None

def load_model():
    """Load the best available model."""
    global model, model_version, loaded_model_path
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model_path = MODEL_PATH
            model = joblib.load(MODEL_PATH)
            model_version = "xgboost_optuna_v1"
            logger.info(f"Loaded XGBoost model from {MODEL_PATH}")
        elif os.path.exists(FALLBACK_MODEL_PATH):
            loaded_model_path = FALLBACK_MODEL_PATH
            model = joblib.load(FALLBACK_MODEL_PATH)
            model_version = "legacy_fraud_detector_v1"
            logger.info(f"Loaded fallback model from {FALLBACK_MODEL_PATH}")
        elif os.path.exists(LEGACY_LIGHTGBM_MODEL_PATH):
            loaded_model_path = LEGACY_LIGHTGBM_MODEL_PATH
            model = joblib.load(LEGACY_LIGHTGBM_MODEL_PATH)
            model_version = "lightgbm_best_v1"
            logger.info(f"Loaded LightGBM fallback model from {LEGACY_LIGHTGBM_MODEL_PATH}")
        else:
            logger.error("No model found. Please train a model first.")
            model = None
            model_version = "none"
            loaded_model_path = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_version = "error"

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    load_model()
    logger.info("Fraud Detection API started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Fraud Detection API shutting down")

# ----------------------------------------------------------------------
# Helper: convert transaction list to DataFrame
# ----------------------------------------------------------------------
def transactions_to_df(transactions: List[Transaction]) -> pd.DataFrame:
    """Convert list of Transaction objects to DataFrame with correct column order."""
    records = [t.dict() for t in transactions]
    df = pd.DataFrame(records)
    # Ensure correct column order (matching training)
    expected_order = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    # Reindex, fill missing with 0 (should not happen)
    df = df.reindex(columns=expected_order, fill_value=0.0)
    return df

# ----------------------------------------------------------------------
# Health endpoint
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if API and model are ready."""
    resolved_path = loaded_model_path
    if resolved_path is None:
        resolved_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=resolved_path,
        version=model_version or "unknown"
    )

# ----------------------------------------------------------------------
# Prediction endpoint
# ----------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """
    Predict fraud for a batch of transactions.

    - **transactions**: List of 30 features (Time, V1-V28, Amount)
    
    Returns fraud predictions (0/1), probabilities, latency, and model info.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator."
        )
    
    import time
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = transactions_to_df(request.transactions)
        
        # Predict probabilities
        probabilities = model.predict_proba(df)[:, 1].tolist()
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 2),
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ----------------------------------------------------------------------
# Root endpoint
# ----------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # set True for development
        log_level="info"
    )