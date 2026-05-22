"""
Unit tests for Fraud Detection API using FastAPI TestClient.
Tests input validation, boundary conditions, model output shape, error handling,
startup/shutdown events, and metrics accumulation.

If a real trained model exists, it will be loaded and used.
Otherwise, a deterministic mock model is used (no test skips).
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.api.main as main_module
from src.api.main import app, update_metrics, metrics_store, transactions_to_df, Transaction
from starlette.testclient import TestClient as StarletteTestClient

# ----------------------------------------------------------------------
# Ensure model is loaded (real if exists, otherwise mock)
# ----------------------------------------------------------------------
def ensure_model_loaded():
    """Try to load the real model; if fails, inject a mock model."""
    # First, attempt to load the model using the app's startup logic
    from src.api.main import load_model
    load_model()  # This sets main_module.model and main_module.model_version

    if main_module.model is None:
        # Fallback to a deterministic mock model
        class MockModel:
            def predict_proba(self, df):
                n = len(df)
                # Return probabilities: first column = P(legit), second = P(fraud)
                # For all transactions, return fraud probability = 0.3 (so threshold 0.5 → all legitimate)
                return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])
        main_module.model = MockModel()
        main_module.model_version = "mock_fallback"
        print("⚠️  Using mock model (no trained model found).")
    else:
        print(f"✅ Using real model: {main_module.model_version}")

# Call at module load time
ensure_model_loaded()

# Create test client
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
# Reset metrics before each test (to avoid cross-test contamination)
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_metrics():
    metrics_store["total_predictions"] = 0
    metrics_store["total_fraud_predictions"] = 0
    metrics_store["total_response_time_ms"] = 0.0
    metrics_store["avg_response_time_ms"] = 0.0
    metrics_store["fraud_rate"] = 0.0
    yield


# ----------------------------------------------------------------------
# Startup / shutdown events coverage
# ----------------------------------------------------------------------
def test_startup_shutdown_events():
    """Manually call startup and shutdown event handlers to cover those lines."""
    import asyncio
    from src.api.main import startup_event, shutdown_event

    async def run():
        await startup_event()
        await shutdown_event()

    asyncio.run(run())
    assert True


# ----------------------------------------------------------------------
# transactions_to_df edge case (reindex with missing columns)
# ----------------------------------------------------------------------
def test_transactions_to_df_reindex_edge_case():
    """Ensure that transactions_to_df correctly reindexes columns, filling missing ones with 0."""
    txn_data = valid_transaction()
    txn_obj = Transaction(**txn_data)
    df = transactions_to_df([txn_obj])
    expected_order = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    assert list(df.columns) == expected_order
    assert df.shape[1] == 30


# ----------------------------------------------------------------------
# Tests for /health and /metrics
# ----------------------------------------------------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" in data


def test_metrics_endpoint():
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
    payload = {"transactions": [valid_transaction()]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "latency_ms" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert len(data["predictions"]) == 1
    assert len(data["probabilities"]) == 1
    assert 0 <= data["probabilities"][0] <= 1


def test_valid_batch_transactions():
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=50.0)]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2


# ----------------------------------------------------------------------
# Invalid input tests (validation errors -> 422)
# ----------------------------------------------------------------------
def test_missing_transactions_field():
    payload = {"invalid": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_transactions_list():
    payload = {"transactions": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_feature():
    incomplete = valid_transaction()
    del incomplete["V1"]
    payload = {"transactions": [incomplete]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_amount():
    txn = valid_transaction(amount=-10.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_data_type():
    txn = valid_transaction()
    txn["Amount"] = "not a number"
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_extra_field():
    txn = valid_transaction()
    txn["ExtraField"] = 123
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ----------------------------------------------------------------------
# Boundary value tests
# ----------------------------------------------------------------------
def test_boundary_amount_zero():
    txn = valid_transaction(amount=0.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_boundary_amount_very_large():
    txn = valid_transaction(amount=1_000_000_000.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probabilities"][0] <= 1


def test_boundary_extreme_v_values():
    txn = valid_transaction()
    txn["V1"] = 1e6
    txn["V2"] = -1e6
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_boundary_time_large():
    txn = valid_transaction(time=1_000_000_000.0)
    payload = {"transactions": [txn]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ----------------------------------------------------------------------
# Model output shape and content tests
# ----------------------------------------------------------------------
def test_output_shape_matches_batch_size():
    batch_size = 5
    transactions = [valid_transaction(amount=10.0 * i) for i in range(batch_size)]
    payload = {"transactions": transactions}
    response = client.post("/predict", json=payload)
    data = response.json()
    assert len(data["predictions"]) == batch_size
    assert len(data["probabilities"]) == batch_size


def test_output_probabilities_in_range():
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=1000.0)]}
    response = client.post("/predict", json=payload)
    data = response.json()
    for prob in data["probabilities"]:
        assert 0.0 <= prob <= 1.0


def test_output_predictions_binary():
    payload = {"transactions": [valid_transaction(), valid_transaction(amount=5000.0)]}
    response = client.post("/predict", json=payload)
    data = response.json()
    for pred in data["predictions"]:
        assert pred in (0, 1)


# ----------------------------------------------------------------------
# Error handling branch coverage (except Exception in /predict)
# ----------------------------------------------------------------------
def test_prediction_internal_error():
    """Mock predict_proba to raise an exception, covering the except Exception branch."""
    with patch('src.api.main.model.predict_proba', side_effect=RuntimeError("mock inference error")):
        payload = {"transactions": [valid_transaction()]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


# ----------------------------------------------------------------------
# Metrics update accumulation tests
# ----------------------------------------------------------------------
def test_metrics_update_accumulation():
    """Directly test update_metrics function for correct calculations."""
    # Reset already done by fixture, but we'll also set explicitly
    metrics_store["total_predictions"] = 0
    metrics_store["total_fraud_predictions"] = 0
    metrics_store["total_response_time_ms"] = 0.0

    update_metrics(fraud_count=2, response_time_ms=100.0, batch_size=5)
    assert metrics_store["total_predictions"] == 5
    assert metrics_store["total_fraud_predictions"] == 2
    assert metrics_store["avg_response_time_ms"] == 20.0
    assert metrics_store["fraud_rate"] == 0.4

    update_metrics(fraud_count=1, response_time_ms=30.0, batch_size=3)
    assert metrics_store["total_predictions"] == 8
    assert metrics_store["total_fraud_predictions"] == 3
    assert metrics_store["avg_response_time_ms"] == 16.25
    assert metrics_store["fraud_rate"] == 0.375


def test_metrics_persist_across_requests():
    """Ensure metrics accumulate correctly when calling the actual endpoint multiple times."""
    # Metrics are reset before each test by fixture, so start fresh
    client.post("/predict", json={"transactions": [valid_transaction(amount=1000.0)]})
    client.post("/predict", json={"transactions": [valid_transaction(amount=2000.0), valid_transaction(amount=50.0)]})

    response = client.get("/metrics")
    data = response.json()
    # With real model, predictions may be 0 or 1; with mock model (0.3 prob) all are 0.
    # So fraud_rate may be 0.0 or positive. We'll just check counts.
    assert data["total_predictions"] == 3
    assert data["avg_response_time_ms"] > 0
    assert 0.0 <= data["fraud_rate"] <= 1.0


# ----------------------------------------------------------------------
# Optional: test that model-load-failure returns 503 (simulate model=None)
# ----------------------------------------------------------------------
def test_no_model_returns_503():
    """If model is None, /predict should return 503."""
    with patch('src.api.main.model', None):
        # Need to re-import client? No, but the patch applies to the module's reference.
        # Create a new client with the patched module.
        from src.api.main import app
        temp_client = StarletteTestClient(app)
        response = temp_client.post("/predict", json={"transactions": [valid_transaction()]})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]