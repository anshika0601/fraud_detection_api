"""
FastAPI production server for fraud detection.
Loads the best model (XGBoost/LightGBM) and serves real-time predictions.
Exposes /metrics for real-time monitoring.
"""

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, model_validator
from pydantic.config import ConfigDict
from typing import List
import joblib
import logging
from datetime import datetime
import time
import hashlib
import json
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-bound ML inference (so it doesn't block the event loop)
ML_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml-inference")


async def run_in_threadpool(func, *args, **kwargs):
    """Run blocking function in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ML_THREAD_POOL, lambda: func(*args, **kwargs))

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
    if metrics_store["total_predictions"] > 0:
        metrics_store["avg_response_time_ms"] = (
            metrics_store["total_response_time_ms"] / metrics_store["total_predictions"]
        )
        metrics_store["fraud_rate"] = (
            metrics_store["total_fraud_predictions"] / metrics_store["total_predictions"]
        )

# ----------------------------------------------------------------------
# Pydantic v2 schemas
# ----------------------------------------------------------------------
class Transaction(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra='forbid', 
        json_schema_extra={
            "example": {
                "Time": 0.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388, "V7": 0.239599,
                "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.551600,
                "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
                "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474,
                "V24": 0.066928, "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053, "Amount": 149.62
            }
        }
    )

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

    @field_validator('Amount')
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v


class PredictionRequest(BaseModel):
    transactions: List[Transaction]

    @field_validator('transactions')
    @classmethod
    def not_empty(cls, v: List[Transaction]) -> List[Transaction]:
        if not v:
            raise ValueError('At least one transaction required')
        return v


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: List[int]
    probabilities: List[float]
    latency_ms: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_rate: float
    avg_response_time_ms: float



# ----------------------------------------------------------------------
# Request context (for correlation IDs across logs)
# ----------------------------------------------------------------------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ----------------------------------------------------------------------
# Custom logging formatter with request ID
# ----------------------------------------------------------------------
class RequestIDFilter(logging.Filter):
    """Inject request_id into every log record."""
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True


# Re-configure logger with request ID
for handler in logging.root.handlers:
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(request_id)s] - %(name)s - %(levelname)s - %(message)s'
    ))


# ----------------------------------------------------------------------
# Helper: hash input payload for audit trail (privacy-safe)
# ----------------------------------------------------------------------
def hash_payload(payload: dict) -> str:
    """Generate SHA256 hash of request payload for audit/debugging."""
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except Exception:
        return "unhashable"


# ----------------------------------------------------------------------
# Logging Middleware
# ----------------------------------------------------------------------
class PredictionLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with:
    - timestamp
    - unique request_id
    - method + path
    - input hash (SHA256, first 16 chars)
    - response status
    - latency
    - prediction results (for /predict endpoint)
    """

    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        req_id = str(uuid.uuid4())[:8]
        request_id_ctx.set(req_id)

        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()

        # Capture request body for hashing (without consuming the stream)
        body_bytes = await request.body()
        input_hash = "no-body"
        if body_bytes:
            try:
                payload = json.loads(body_bytes)
                input_hash = hash_payload(payload)
            except json.JSONDecodeError:
                input_hash = hashlib.sha256(body_bytes).hexdigest()[:16]

        # Reconstruct the body for downstream handlers
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        # Log incoming request
        logger.info(
            f"REQUEST | {request.method} {request.url.path} | "
            f"input_hash={input_hash} | client={request.client.host if request.client else 'unknown'}"
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"REQUEST_FAILED | error={str(e)}")
            raise

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Capture response body for prediction logging (only for /predict)
        response_log = ""
        if request.url.path == "/predict" and response.status_code == 200:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            try:
                result = json.loads(response_body)
                fraud_count = sum(result.get("predictions", []))
                total = len(result.get("predictions", []))
                avg_prob = (
                    sum(result.get("probabilities", [])) / total if total > 0 else 0
                )
                response_log = (
                    f" | predictions={total} | fraud_detected={fraud_count} | "
                    f"avg_prob={avg_prob:.4f} | model_version={result.get('model_version', 'unknown')}"
                )
            except Exception as e:
                response_log = f" | response_parse_error={e}"

            # Rebuild the response since we consumed the body
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Log response
        logger.info(
            f"RESPONSE | status={response.status_code} | latency={latency_ms:.2f}ms"
            f"{response_log}"
        )

        # Add correlation ID to response headers (useful for debugging)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

        return response

# ----------------------------------------------------------------------
# FastAPI app
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
app.add_middleware(PredictionLoggingMiddleware)

# ----------------------------------------------------------------------
# Model loading (once at startup)
# ----------------------------------------------------------------------
model = None
model_version = None


def load_model():
    global model, model_version
    try:
        if os.path.exists(MODEL_PATH):
            loaded = joblib.load(MODEL_PATH)
            model_version = "xgboost_optuna_v1"
            logger.info(f"Loaded XGBoost model from {MODEL_PATH}")
        elif os.path.exists(FALLBACK_MODEL_PATH):
            loaded = joblib.load(FALLBACK_MODEL_PATH)
            model_version = "lightgbm_best_v1"
            logger.info(f"Loaded LightGBM fallback model from {FALLBACK_MODEL_PATH}")
        else:
            logger.error("No model found. Please train a model first.")
            model = None
            model_version = "none"
            return

        # Some serialized XGBoost artifacts are raw boosters (xgboost.core.Booster)
        # which don't expose predict_proba. Tests expect model.predict_proba to exist so it can be patched.
        if not hasattr(loaded, "predict_proba") and hasattr(loaded, "predict") and hasattr(loaded, "get_booster") is False:
            # Wrap booster with a minimal predict_proba interface.
            try:
                import xgboost as xgb

                class _BoosterPredictProbaWrapper:
                    def __init__(self, booster):
                        self._booster = booster

                    def predict_proba(self, df: pd.DataFrame):
                        dmatrix = xgb.DMatrix(df)
                        raw = self._booster.predict(dmatrix)
                        fraud_proba = np.asarray(raw).reshape(-1).astype(float)
                        fraud_proba = np.clip(fraud_proba, 0.0, 1.0)
                        legit_proba = 1.0 - fraud_proba
                        return np.column_stack([legit_proba, fraud_proba])

                    # Provide access to feature names when possible.
                    @property
                    def feature_names_in_(self):
                        try:
                            return self._booster.feature_names
                        except Exception:
                            raise AttributeError

                model = _BoosterPredictProbaWrapper(loaded)
                logger.info("Wrapped raw XGBoost Booster with predict_proba adapter")
            except Exception as e:
                logger.warning(f"Could not wrap booster with predict_proba: {e}")
                model = loaded
        else:
            model = loaded

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_version = "error"



@app.on_event("startup")
async def startup_event():
    # Load model in a thread pool to not block startup
    await run_in_threadpool(load_model)
    logger.info("Fraud Detection API started")
    logger.info(f"ML thread pool: {ML_THREAD_POOL._max_workers} workers")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ML thread pool...")
    ML_THREAD_POOL.shutdown(wait=True)
    logger.info("Fraud Detection API shut down cleanly")
# ----------------------------------------------------------------------
# Helper: convert transaction list to DataFrame
# ----------------------------------------------------------------------
def transactions_to_df(transactions: List[Transaction]) -> pd.DataFrame:
    records = [t.model_dump() for t in transactions]
    df = pd.DataFrame(records)

    model_feature_names = None
    try:
        if model is not None:
            if hasattr(model, "feature_names_in_"):
                model_feature_names = list(model.feature_names_in_)
            elif hasattr(model, "get_booster"):
                model_feature_names = model.get_booster().feature_names
            elif hasattr(model, "feature_name"):
                model_feature_names = model.feature_name()
    except Exception as e:
        logger.warning(f"Could not read model feature names: {e}")

    if model_feature_names:
        # Check if the incoming data is missing columns before we reindex (which fills them with 0)
        # This allows the test_missing_column_after_reindex to trigger an error
        missing = set(model_feature_names) - set(df.columns)
        if missing and "__row_id" not in missing:
             # Raise an error so the /predict endpoint returns 500/422 as expected by tests
             raise ValueError(f"Missing required feature columns: {missing}")

        logger.info(f"Model expects features: {model_feature_names}")
        if "__row_id" in model_feature_names:
            df["__row_id"] = np.arange(len(df), dtype=np.int64)
        df = df.reindex(columns=model_feature_names, fill_value=0.0)
    else:
        # Fallback logic...
        fallback_cols = [...] 
        df = df.reindex(columns=fallback_cols, fill_value=0.0)

    return df

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "predict": "/predict (POST)",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH,
        version=model_version or "unknown",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Real-time inference metrics:
    - total_predictions : total transactions scored
    - fraud_rate        : proportion flagged as fraud
    - avg_response_time_ms : average latency per transaction
    """
    return MetricsResponse(
        total_predictions=metrics_store["total_predictions"],
        fraud_rate=round(metrics_store["fraud_rate"], 6),
        avg_response_time_ms=round(metrics_store["avg_response_time_ms"], 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    start_time = time.time()

    try:
        # DataFrame conversion (lightweight, can stay sync)
        df = transactions_to_df(request.transactions)

        # -------------------- INPUT VALIDATION --------------------
        if df.empty or len(df) == 0:
            raise ValueError("Input DataFrame is empty (zero rows)")

        expected_features = None
        try:
            if model is not None:
                if hasattr(model, "feature_names_in_"):
                    expected_features = list(model.feature_names_in_)
                elif hasattr(model, "get_booster"):
                    expected_features = model.get_booster().feature_names
                elif hasattr(model, "feature_name"):
                    expected_features = model.feature_name()
        except Exception as e:
            logger.warning(f"Could not read model feature names: {e}")

        if expected_features:
            required = set(f for f in expected_features if not f.startswith("__"))
            actual = set(df.columns)

            missing_cols = required - actual
            if missing_cols:
                raise ValueError(f"Input is missing required feature columns: {sorted(missing_cols)}")

            extra_cols = actual - set(expected_features)
            if extra_cols:
                raise ValueError(f"Input contains unexpected extra columns: {sorted(extra_cols)}")

            nan_cols = [c for c in required if c in df.columns and df[c].isna().any()]
            if nan_cols:
                raise ValueError(f"Input contains NaN in required features: {sorted(nan_cols)}")
        else:
            fallback_cols = [
                'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                'V28', 'Amount',
            ]
            missing_cols = set(fallback_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Input is missing required feature columns: {sorted(missing_cols)}")
            extra_cols = set(df.columns) - set(fallback_cols)
            if extra_cols:
                raise ValueError(f"Input contains unexpected extra columns: {sorted(extra_cols)}")

        # ---------- ASYNC ML INFERENCE (offloaded to thread pool) ----------
        if hasattr(model, "predict_proba"):
            proba_result = await run_in_threadpool(model.predict_proba, df)
            probabilities = proba_result[:, 1].tolist()
            logger.debug("Using model.predict_proba (async)")
        elif (
            model.__class__.__module__.startswith("xgboost")
            or model.__class__.__name__.lower() in {"booster", "xgbclassifier"}
        ):
            import xgboost as xgb
            def _xgb_predict():
                dmatrix = xgb.DMatrix(df)
                return model.predict(dmatrix)
            raw = await run_in_threadpool(_xgb_predict)
            probabilities = np.asarray(raw).reshape(-1).astype(float).tolist()
        elif hasattr(model, "predict"):
            raw = await run_in_threadpool(model.predict, df)
            probabilities = np.asarray(raw).reshape(-1).astype(float).tolist()
        else:
            raise AttributeError("Loaded model does not support predict_proba or predict")

        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        latency_ms = (time.time() - start_time) * 1000

        # Metrics update (sync, very fast)
        update_metrics(
            fraud_count=sum(predictions),
            response_time_ms=latency_ms,
            batch_size=len(predictions),
        )

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 2),
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8003"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting uvicorn on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")