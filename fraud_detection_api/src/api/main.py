"""
FastAPI production server for fraud detection.
Loads the best model (XGBoost/LightGBM) and serves real-time predictions.
"""

import os
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import logging
from datetime import datetime
from pathlib import Path
import uuid
import csv

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
FALLBACK_MODEL_PATH = os.getenv(
    "FALLBACK_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "fraud_detector_v1.pkl")
)
LEGACY_LIGHTGBM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_best_model.pkl")

# Prediction logging for drift analysis
PREDICTIONS_LOG_DIR = os.getenv("PREDICTIONS_LOG_DIR", os.path.join(BASE_DIR, "logs"))
PREDICTIONS_CSV = os.path.join(PREDICTIONS_LOG_DIR, "predictions.csv")

Path(PREDICTIONS_LOG_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Pydantic schemas with enhanced constraints
# ----------------------------------------------------------------------
class TransactionInput(BaseModel):
    """
    Single transaction features – all 30 features expected by the model.
    Constraints:
        - Time >= 0 (seconds since first transaction)
        - Amount > 0 (dollar amount)
        - V1..V28: any float, but with reasonable outliers warning.
    """
    Time: float = Field(..., ge=0, description="Seconds elapsed from first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., gt=0, description="Transaction amount in USD")

    @validator('Time')
    def time_non_negative(cls, v):
        if v < 0:
            raise ValueError('Time must be >= 0')
        return v

    @validator('Amount')
    def amount_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be > 0')
        return v

    # Optional: warn if any V feature is extremely out of range (beyond typical PCA values)
    # Not raising error, just logging warning (can be changed to error if needed)
    @validator('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
               'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
               'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28')
    def check_v_outliers(cls, v, field):
        # Typical PCA values in fraud dataset are roughly between -5 and 5.
        # Values beyond +/-10 are rare and might indicate data issues.
        if abs(v) > 10:
            logger.warning(f"{field.name} value {v} is unusually large (|value| > 10). "
                           "This may affect prediction quality.")
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
    transactions: List[TransactionInput]

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
    model_version: str
    model_path: str
    uptime_seconds: float


class PredictionLogRequest(BaseModel):
    """Request to log a prediction for drift analysis."""
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    features: dict = Field(..., description="Input features used for prediction")
    model_version: Optional[str] = None


class PredictionLogResponse(BaseModel):
    status: str
    message: str
    log_id: str


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
# Global variables
# ----------------------------------------------------------------------
model = None
model_version = None
loaded_model_path = None
startup_time = None

def load_model():
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
            logger.info(f"Loaded LightGBM model from {LEGACY_LIGHTGBM_MODEL_PATH}")
        else:
            logger.error("No model found.")
            model = None
            model_version = "none"
            loaded_model_path = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_version = "error"

@app.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = time.time()
    load_model()
    logger.info(f"Fraud Detection API started at {datetime.utcnow().isoformat()}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Fraud Detection API shutting down")

# ----------------------------------------------------------------------
# Helper: convert list of TransactionInput to DataFrame
# ----------------------------------------------------------------------
def transactions_to_df(transactions: List[TransactionInput]) -> pd.DataFrame:
    records = [t.dict() for t in transactions]
    df = pd.DataFrame(records)
    expected_order = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    df = df.reindex(columns=expected_order, fill_value=0.0)
    return df


def log_prediction_to_csv(prediction: int, probability: float, features: dict, model_version: str) -> str:
    """
    Log prediction to CSV for drift analysis.

    Returns:
        log_id: Unique identifier for this prediction log
    """
    log_id = str(uuid.uuid4())[:8]

    try:
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'log_id': log_id,
            'prediction': prediction,
            'probability': probability,
            'model_version': model_version,
        }
        row.update(features)

        file_exists = os.path.exists(PREDICTIONS_CSV)

        with open(PREDICTIONS_CSV, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'log_id', 'prediction', 'probability', 'model_version'] + list(features.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        logger.info(f"Logged prediction {log_id}")
        return log_id

    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        raise

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    resolved_path = loaded_model_path
    if resolved_path is None:
        resolved_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    uptime = time.time() - startup_time if startup_time is not None else 0.0
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version or "unknown",
        model_path=resolved_path,
        uptime_seconds=round(uptime, 2)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )
    start_time = time.time()
    try:
        df = transactions_to_df(request.transactions)
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


@app.post("/log-prediction", response_model=PredictionLogResponse, tags=["Monitoring"])
async def log_prediction(request: PredictionLogRequest):
    """
    Log a prediction for drift analysis.
    Saves to CSV for later analysis with /generate-drift-report.
    """
    try:
        log_id = log_prediction_to_csv(
            prediction=request.prediction,
            probability=request.probability,
            features=request.features,
            model_version=request.model_version or model_version
        )
        return PredictionLogResponse(
            status="success",
            message=f"Prediction logged successfully",
            log_id=log_id
        )
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log prediction: {str(e)}"
        )


@app.get("/drift-status", tags=["Monitoring"])
async def drift_status():
    """
    Get current status of prediction logging for drift analysis.
    Returns count of logged predictions and CSV path.
    """
    try:
        if os.path.exists(PREDICTIONS_CSV):
            df = pd.read_csv(PREDICTIONS_CSV)
            return {
                "status": "active",
                "total_predictions_logged": len(df),
                "csv_path": PREDICTIONS_CSV,
                "message": "Use run_drift_detection.py to generate drift report"
            }
        else:
            return {
                "status": "no_data",
                "total_predictions_logged": 0,
                "csv_path": PREDICTIONS_CSV,
                "message": "No predictions logged yet"
            }
    except Exception as e:
        logger.error(f"Error getting drift status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking drift status: {str(e)}"
        )


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "log-prediction": "/log-prediction (POST)",
        "generate-drift-report": "/generate-drift-report (GET)"
    }
        "health": "/health",
        "predict": "/predict (POST)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )