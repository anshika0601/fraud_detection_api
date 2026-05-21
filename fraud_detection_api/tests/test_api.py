"""
Unit tests for Fraud Detection API using FastAPI TestClient.
Tests input validation, boundary conditions, and model output shape.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

# NOTE: FastAPI/Starlette TestClient relies on httpx version compatibility.
# In this environment, Starlette TestClient is incompatible with the installed httpx,
# so we use the synchronous TestClient from starlette directly.
from starlette.testclient import TestClient as StarletteTestClient

client = StarletteTestClient(app)



# ----------------------------------------------------------------------
# Helper: sample valid transaction
# ----------------------------------------------------------------------
def valid_transaction(amount=149.62, time=0.0):
    return {
        "Time": time,
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
        "Amount": amount
    }


# ----------------------------------------------------------------------
# Tests for /health and /metrics
# ----------------------------------------------------------------------
def test_health_endpoint():
    """Health endpoint should return 200 and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" in data


def test_metrics_endpoint():
    """Metrics endpoint should return numeric values."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "fraud_rate" in data
    assert "avg_response_time_ms" in data
    assert isinstance(data["total_predictions"], int)
    assert isinstance(data["fraud_rate"], (int, float))
    assert isinstance(data["avg_response_time_ms"], (int, float))


# ----------------------------------------------------------------------
# Valid input tests
# ----------------------------------------------------------------------
def test_valid_single_transaction():
    """Single valid transaction should return 200 with predictions."""
    payload = {"transactions": [valid_transaction()]}
    response = client.post("/predict", json=payload)
    
    # If model is not loaded, test may return 503 – skip or handle.
    if response.status_code == 503:
        pytest.skip("Model not loaded – skipping prediction test")
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "latency_ms" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert len(data["predictions"]) == 1
    assert len(data["probabilities"]) == 1
    # Probabilities should be between 0 and 1
    assert 0 <= data["probabilities"][0] <= 1


def test_valid_batch_transactions():
    """Batch of valid transactions should return predictions for each."""
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=50.0)]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2


# ----------------------------------------------------------------------
# Invalid input tests (validation errors -> 422)
# ----------------------------------------------------------------------
def test_missing_transactions_field():
    """Missing 'transactions' field -> 422."""
    payload = {"invalid": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_transactions_list():
    """Empty transactions list -> validation error (pydantic validator)."""
    payload = {"transactions": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_feature():
    """Transaction missing a required field (e.g., V1) -> 422."""
    incomplete = valid_transaction()
    del incomplete["V1"]
    payload = {"transactions": [incomplete]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_amount():
    """Negative Amount -> 422 (pydantic validator)."""
    txn = valid_transaction(amount=-10.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_data_type():
    """String value for a numeric field -> 422."""
    txn = valid_transaction()
    txn["Amount"] = "not a number"
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_extra_field():
    """Extra fields should fail Pydantic validation (422), not short-circuit with 503."""
    # Force model to appear "loaded" so the request reaches Pydantic validation.
    import numpy as np
    import src.api.main as main_mod

    class MockModel:
        def predict_proba(self, df):
            # Return 2 columns: P(class0), P(class1)
            n = len(df)
            return np.column_stack([np.zeros(n), np.full(n, 0.6)])


    main_mod.model = MockModel()
    main_mod.model_version = "mock"

    txn = valid_transaction()
    txn["ExtraField"] = 123
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422



# ----------------------------------------------------------------------
# Boundary value tests
# ----------------------------------------------------------------------
def test_boundary_amount_zero():
    """Amount = 0 should be accepted (non-negative)."""
    txn = valid_transaction(amount=0.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    assert response.status_code == 200


def test_boundary_amount_very_large():
    """Very large Amount (e.g., 1e9) should still be accepted."""
    txn = valid_transaction(amount=1_000_000_000.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    assert response.status_code == 200
    data = response.json()
    # Probability should still be valid
    assert 0 <= data["probabilities"][0] <= 1


def test_boundary_extreme_v_values():
    """Extremely large positive/negative V values (e.g., ±1e6) should not crash."""
    txn = valid_transaction()
    txn["V1"] = 1e6
    txn["V2"] = -1e6
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    assert response.status_code == 200


def test_boundary_time_large():
    """Time (seconds) can be very large (e.g., 1e9)."""
    txn = valid_transaction(time=1_000_000_000.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    assert response.status_code == 200


# ----------------------------------------------------------------------
# Model output shape and content tests (only if model loaded)
# ----------------------------------------------------------------------
def test_output_shape_matches_batch_size():
    """Number of predictions/probabilities must equal number of input transactions."""
    batch_size = 5
    transactions = [valid_transaction(amount=10.0 * i) for i in range(batch_size)]
    payload = {"transactions": transactions}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    data = response.json()
    assert len(data["predictions"]) == batch_size
    assert len(data["probabilities"]) == batch_size


def test_output_probabilities_in_range():
    """All probability values must be between 0 and 1 inclusive."""
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=1000.0)]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    data = response.json()
    for prob in data["probabilities"]:
        assert 0.0 <= prob <= 1.0


def test_output_predictions_binary():
    """Predictions should be 0 or 1."""
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=5000.0)]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    data = response.json()
    for pred in data["predictions"]:
        assert pred in (0, 1)


# ----------------------------------------------------------------------
# Edge case: no model loaded (optional)
# ----------------------------------------------------------------------
def test_no_model_returns_503():
    """If model not loaded, /predict should return 503."""
    # This test is meaningful only when the model file is missing.
    # We'll force-reload the app? Not needed; just check if model is None.
    from src.api.main import model
    if model is None:
        response = client.post("/predict", json={"transactions": [valid_transaction()]})
        assert response.status_code == 503
    else:
        pytest.skip("Model is loaded, skipping 503 test")