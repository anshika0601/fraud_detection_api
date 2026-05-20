"""
FastAPI production server for fraud detection.
Loads the best model (XGBoost/LightGBM) and serves real-time predictions.
Exposes /metrics for real-time monitoring.
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
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_detector_xgb_v1.pkl")
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL_PATH", "models/lightgbm_best_model.pkl")

# ----------------------------------------------------------------------
# Metrics tracking (in-memory)
# ----------------------------------------------------------------------
metrics_store = {
    "total_predictions": 0,
    "total_fraud_predictions": 0,
    "total_response_time_ms": 0.0,
    "avg_response_time_ms": 0.0,
    "fraud_rate": 0.0,
}

def update_metrics(fraud_count: int, response_time_ms: float, batch_size: int):
    """Update metrics after a batch prediction."""
    metrics_store["total_predictions"] += batch_size
    metrics_store["total_fraud_predictions"] += fraud_count
    metrics_store["total_response_time_ms"] += response_time_ms
    # Update average response time (simple moving average)
    if metrics_store["total_predictions"] > 0:
        metrics_store["avg_response_time_ms"] = (
            metrics_store["total_response_time_ms"] / metrics_store["total_predictions"]
        )
    # Update fraud rate
    if metrics_store["total_predictions"] > 0:
        metrics_store["fraud_rate"] = (
            metrics_store["total_fraud_predictions"] / metrics_store["total_predictions"]
        )

# ----------------------------------------------------------------------
# Pydantic schemas (request/response validation)
# ----------------------------------------------------------------------
class Transaction(BaseModel):
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
    transactions: List[Transaction]

    @validator('transactions')
    def not_empty(cls, v):
        if not v:
            raise ValueError('At least one transaction required')
        return v

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    latency_ms: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str

class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_rate: float
    avg_response_time_ms: float

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Model loading (once at startup)
# ----------------------------------------------------------------------
model = None
model_version = None

def load_model():
    global model, model_version
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            model_version = "xgboost_optuna_v1"
            logger.info(f"Loaded XGBoost model from {MODEL_PATH}")
        elif os.path.exists(FALLBACK_MODEL_PATH):
            model = joblib.load(FALLBACK_MODEL_PATH)
            model_version = "lightgbm_best_v1"
            logger.info(f"Loaded LightGBM fallback model from {FALLBACK_MODEL_PATH}")
        else:
            logger.error("No model found. Please train a model first.")
            model = None
            model_version = "none"
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_version = "error"

@app.on_event("startup")
async def startup_event():
    load_model()
    logger.info("Fraud Detection API started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Fraud Detection API shutting down")

# ----------------------------------------------------------------------
# Helper: convert transaction list to DataFrame
# ----------------------------------------------------------------------
def transactions_to_df(transactions: List[Transaction]) -> pd.DataFrame:
    records = [t.dict() for t in transactions]
    df = pd.DataFrame(records)

    expected_order = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]

    # Add __row_id only if the model artifact requires it.
    # We keep this conditional because some trained artifacts may not expect it.
    model_feature_names = None
    try:
        if model is not None and hasattr(model, "feature_names_in_"):
            model_feature_names = list(getattr(model, "feature_names_in_"))
        elif model is not None and hasattr(model, "get_booster"):
            booster = model.get_booster()
            model_feature_names = booster.feature_names
    except Exception:
        model_feature_names = None

    if model_feature_names and "__row_id" in model_feature_names and "__row_id" not in df.columns:
        df["__row_id"] = np.arange(len(df), dtype=np.int64)

    # Drop unexpected columns that can break models expecting an exact schema.
    # If __row_id is required, it must be included in the expected order.
    final_expected_order = expected_order
    if model_feature_names and "__row_id" in model_feature_names:
        # Preserve model's own feature order when available.
        final_expected_order = list(model_feature_names)

    df = df.drop(columns=[c for c in df.columns if c not in final_expected_order], errors="ignore")
    df = df.reindex(columns=final_expected_order, fill_value=0.0)
    return df



# ----------------------------------------------------------------------
# Health endpoint
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH,
        version=model_version or "unknown"
    )

# ----------------------------------------------------------------------
# Metrics endpoint (new)
# ----------------------------------------------------------------------
@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Return real-time inference metrics:
    - total_predictions: number of transactions predicted so far
    - fraud_rate: proportion of predictions flagged as fraud
    - avg_response_time_ms: average response time per transaction (ms)
    """
    return MetricsResponse(
        total_predictions=metrics_store["total_predictions"],
        fraud_rate=round(metrics_store["fraud_rate"], 6),
        avg_response_time_ms=round(metrics_store["avg_response_time_ms"], 2),
    )

# ----------------------------------------------------------------------
# Prediction endpoint (updated to update metrics)
# ----------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator."
        )
    
    start_time = time.time()
    
    try:
        df = transactions_to_df(request.transactions)

        # Support both sklearn-style estimators (predict_proba) and native XGBoost / LightGBM predictors
        # Native XGBoost models/Booster typically require xgboost.DMatrix at predict-time.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[:, 1].tolist()
            logger.debug("Using model.predict_proba for probability estimates")
        elif model.__class__.__module__.startswith("xgboost") or model.__class__.__name__.lower() in {"booster", "xgbclassifier"}:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(df)
            raw = model.predict(dmatrix)
            probabilities = np.asarray(raw).reshape(-1).astype(float).tolist()
            logger.debug("Using xgboost.DMatrix + model.predict")
        elif hasattr(model, "predict"):
            probabilities = np.asarray(model.predict(df)).reshape(-1).astype(float).tolist()
            logger.debug("Using model.predict for probability estimates")
        else:
            raise AttributeError("Loaded model does not support predict_proba or predict")


        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        batch_size = len(predictions)
        fraud_count = sum(predictions)
        update_metrics(fraud_count, latency_ms, batch_size)
        
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

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "predict": "/predict (POST)"
    }

if __name__ == "__main__":
    import uvicorn

    # Allow overriding port to avoid "address already in use" conflicts.
    # Example: set PORT=8001 before starting.
    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting uvicorn on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
