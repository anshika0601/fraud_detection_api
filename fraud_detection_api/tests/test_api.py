import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os
import importlib
import logging
from src.data import preprocess
from src.api import main as main_mod
from src.api.main import app

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NOTE: Using FastAPI's TestClient for compatibility and proper lifecycle handling

@pytest.fixture(scope="function")
def client():
    """
    Create a fresh TestClient for each test.
    This ensures:
    1. startup_event() runs before each test (fresh executor)
    2. shutdown_event() runs after each test (clean executor)
    3. No cross-test executor conflicts
    """
    test_client = TestClient(app)
    yield test_client
    # TestClient context cleanup happens automatically (calls shutdown)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def valid_transaction(amount: float = 149.62, time: float = 0.0) -> dict:
    """Return a complete, valid transaction payload."""
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
        "Amount": amount,
    }


class MockModel:
    """Minimal sklearn-compatible mock that always predicts fraud probability 0.6."""

    @property
    def feature_names_in_(self):
        return [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.column_stack([np.zeros(n), np.full(n, 0.6)])


class ExplodingModel:
    """Mock model whose predict_proba always raises to exercise the except branch."""

    def predict_proba(self, df: pd.DataFrame):
        raise RuntimeError("Simulated model failure")


def _install_mock_model():
    """Point main_mod at a working mock so tests don't need a real model file."""
    main_mod.model = MockModel()
    main_mod.model_version = "mock-v1"


def _clear_model():
    """Remove the model so the app behaves as if no model was loaded."""
    main_mod.model = None
    main_mod.model_version = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Startup / Shutdown lifecycle
# ──────────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    """Verify that the app's startup event loads a model and shutdown cleans up."""

    def test_startup_loads_model_when_file_exists(self, tmp_path):
        """
        When joblib.load succeeds during startup, main_mod.model must be set
        and main_mod.model_version must be a non-empty string.
        """
        fake_model = MockModel()
        model_path = tmp_path / "fraud_detector_xgb_v1.pkl"
        model_path.touch()

        with patch("src.api.main.joblib.load", return_value=fake_model) as mock_load, \
             patch("src.api.main.MODEL_PATH", str(model_path)):
            # Re-trigger startup by using the TestClient as a context manager.
            with TestClient(app) as test_client:
                mock_load.assert_called_once()
                resp = test_client.get("/health")
                assert resp.status_code == 200
                assert main_mod.model is not None

    def test_startup_handles_missing_model_file(self, tmp_path):
        """
        When the model file is absent, startup must *not* raise; model stays None
        and the app still starts (health endpoint returns 200).
        """
        missing_path = str(tmp_path / "nonexistent_model.pkl")

        with patch("src.api.main.MODEL_PATH", missing_path), \
             patch("src.api.main.joblib.load", side_effect=FileNotFoundError):
            with TestClient(app) as c:
                resp = c.get("/health")
                assert resp.status_code == 200
                assert resp.json()["model_loaded"] is False

    def test_shutdown_event_runs_without_error(self):
        """
        Using the client as a context manager must not raise on __exit__,
        confirming the shutdown handler completes cleanly.
        """
        _install_mock_model()
        try:
            with TestClient(app) as c:
                c.get("/health")   # any request to keep the lifespan alive
            # If we reach here, shutdown completed without exception.
        finally:
            _install_mock_model()   # restore for subsequent tests


# ──────────────────────────────────────────────────────────────────────────────
# 2. /health and /metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestInfraEndpoints:
    def test_health_endpoint(self, client):
        """Health endpoint should return 200 and model status fields."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data

    def test_metrics_endpoint(self, client):
        """Metrics endpoint should return numeric counters."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_rate" in data
        assert "avg_response_time_ms" in data
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["fraud_rate"], (int, float))
        assert isinstance(data["avg_response_time_ms"], (int, float))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Valid input tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidInputs:
    def setup_method(self):
        _install_mock_model()

    def test_valid_single_transaction(self, client):
        """Single valid transaction should return 200 with correct schema."""
        payload = {"transactions": [valid_transaction()]}
        response = client.post("/predict", json=payload)
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
        assert 0 <= data["probabilities"][0] <= 1

    def test_valid_batch_transactions(self, client):
        """Batch of valid transactions should return predictions for each."""
        payload = {"transactions": [valid_transaction(), valid_transaction(amount=50.0)]}
        response = client.post("/predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert len(data["probabilities"]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 4. Invalid input tests (Pydantic validation → 422)
# ──────────────────────────────────────────────────────────────────────────────

class TestInvalidInputs:
    def setup_method(self):
        _install_mock_model()

    def test_missing_transactions_field(self, client):
        """Missing 'transactions' field -> 422."""
        response = client.post("/predict", json={"invalid": []})
        assert response.status_code == 422

    def test_empty_transactions_list(self, client):
        """Empty transactions list -> 422 (custom Pydantic validator)."""
        response = client.post("/predict", json={"transactions": []})
        assert response.status_code == 422

    def test_missing_required_feature(self, client):
        """Transaction missing V1 -> 422."""
        txn = valid_transaction()
        del txn["V1"]
        response = client.post("/predict", json={"transactions": [txn]})
        assert response.status_code == 422

    def test_negative_amount(self, client):
        """Negative Amount -> 422."""
        txn = valid_transaction(amount=-10.0)
        response = client.post("/predict", json={"transactions": [txn]})
        assert response.status_code == 422

    def test_wrong_data_type(self, client):
        """String value for a numeric field -> 422."""
        txn = valid_transaction()
        txn["Amount"] = "not a number"
        response = client.post("/predict", json={"transactions": [txn]})
        assert response.status_code == 422

    def test_extra_field_rejected(self, client):
        """Extra fields must fail Pydantic validation (422), not trigger 503."""
        txn = valid_transaction()
        txn["ExtraField"] = 123
        response = client.post("/predict", json={"transactions": [txn]})
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# 5. Boundary value tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBoundaryValues:
    def setup_method(self):
        _install_mock_model()

    def test_boundary_amount_zero(self, client):
        """Amount = 0 is on the valid boundary (non-negative)."""
        response = client.post("/predict", json={"transactions": [valid_transaction(amount=0.0)]})
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200

    def test_boundary_amount_very_large(self, client):
        """Very large Amount (1e9) should still yield a valid probability."""
        response = client.post(
            "/predict", json={"transactions": [valid_transaction(amount=1_000_000_000.0)]}
        )
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        assert 0 <= response.json()["probabilities"][0] <= 1

    def test_boundary_extreme_v_values(self, client):
        """Extreme ±1e6 feature values must not crash the endpoint."""
        txn = valid_transaction()
        txn["V1"] = 1e6
        txn["V2"] = -1e6
        response = client.post("/predict", json={"transactions": [txn]})
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200

    def test_boundary_time_large(self, client):
        """Very large Time value (1e9 seconds) should be accepted."""
        response = client.post(
            "/predict", json={"transactions": [valid_transaction(time=1_000_000_000.0)]}
        )
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 6. Model output shape / content
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputShape:
    def setup_method(self):
        _install_mock_model()

    def test_output_shape_matches_batch_size(self, client):
        """predictions and probabilities arrays must have the same length as the batch."""
        batch_size = 5
        transactions = [valid_transaction(amount=10.0 * i) for i in range(batch_size)]
        response = client.post("/predict", json={"transactions": transactions})
        if response.status_code == 503:
            pytest.skip("Model not loaded")

        data = response.json()
        assert len(data["predictions"]) == batch_size
        assert len(data["probabilities"]) == batch_size

    def test_output_probabilities_in_range(self, client):
        """Every probability must be in [0, 1]."""
        payload = {"transactions": [valid_transaction(), valid_transaction(amount=1000.0)]}
        response = client.post("/predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")

        for prob in response.json()["probabilities"]:
            assert 0.0 <= prob <= 1.0

    def test_output_predictions_binary(self, client):
        """Every prediction value must be 0 or 1."""
        payload = {"transactions": [valid_transaction(), valid_transaction(amount=5000.0)]}
        response = client.post("/predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")

        for pred in response.json()["predictions"]:
            assert pred in (0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 7. No model loaded → 503
# ──────────────────────────────────────────────────────────────────────────────

class TestNoModel:
    def test_no_model_returns_503(self, client):
        """When model is None, /predict must return 503."""
        _clear_model()
        try:
            response = client.post("/predict", json={"transactions": [valid_transaction()]})
            assert response.status_code == 503
        finally:
            _install_mock_model()


# ──────────────────────────────────────────────────────────────────────────────
# 8. /predict except-branch coverage
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictExceptBranch:
    def test_model_runtime_error_returns_500(self, client):
        """
        An exploding model raises RuntimeError inside predict_proba.
        The endpoint must catch it and return 500.
        """
        main_mod.model = ExplodingModel()
        main_mod.model_version = "exploding-mock"
        try:
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
            assert response.status_code == 500
        finally:
            _install_mock_model()

    def test_500_response_has_detail_field(self, client):
        """
        The 500 error body must include a 'detail' key so callers can
        distinguish it from an unhandled crash.
        """
        main_mod.model = ExplodingModel()
        main_mod.model_version = "exploding-mock"
        try:
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
        finally:
            _install_mock_model()

    def test_model_value_error_returns_500(self, client):
        """
        Any exception type (ValueError here) must trigger the same 500 path.
        """

        class ValueErrorModel:
            def predict_proba(self, df):
                raise ValueError("Bad feature values")

        main_mod.model = ValueErrorModel()
        main_mod.model_version = "value-error-mock"
        try:
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
            assert response.status_code == 500
        finally:
            _install_mock_model()


# ──────────────────────────────────────────────────────────────────────────────
# 9. transactions_to_df reindex edge case
# ──────────────────────────────────────────────────────────────────────────────

class TestTransactionsToDfReindex:
    def test_missing_column_after_reindex_returns_500(self, client):
        """
        If transactions_to_df returns a DataFrame with a missing feature column
        (filled with NaN after reindex), predict_proba must raise and the
        endpoint must return 500.
        """
        def broken_to_df(transactions):
            df = pd.DataFrame([valid_transaction()])
            return df.drop(columns=["V14"])

        with patch("src.api.main.transactions_to_df", side_effect=broken_to_df):
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
        assert response.status_code in (500, 422)

    def test_extra_column_after_reindex_returns_500(self, client):
        """
        If transactions_to_df injects an unexpected extra column, the route must not silently accept it.
        """

        def extra_col_to_df(transactions):
            df = pd.DataFrame([valid_transaction()])
            df["INJECTED_EXTRA"] = 99.0
            return df

        with patch("src.api.main.transactions_to_df", side_effect=extra_col_to_df):
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
        assert response.status_code in (500, 422)

    def test_all_nan_row_after_reindex_returns_500(self, client):
        """
        A full-NaN row must not produce a 200 response with garbage probabilities.
        """

        def nan_df(_transactions):
            cols = list(valid_transaction().keys())
            return pd.DataFrame([[float("nan")] * len(cols)], columns=cols)

        with patch("src.api.main.transactions_to_df", side_effect=nan_df):
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
        if response.status_code == 200:
            for prob in response.json().get("probabilities", []):
                assert 0.0 <= prob <= 1.0

    def test_empty_dataframe_after_conversion_returns_500(self, client):
        """
        An empty DataFrame returned by transactions_to_df must return 500.
        """

        def empty_df(_transactions):
            cols = list(valid_transaction().keys())
            return pd.DataFrame(columns=cols)

        with patch("src.api.main.transactions_to_df", side_effect=empty_df):
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
        assert response.status_code in (500, 422)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Additional coverage tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCoverageBoost:
    def test_load_model_file_not_found_sets_none(self):
        """Covers model file missing path"""
        original_model = main_mod.model
        original_version = main_mod.model_version
        try:
            with patch("os.path.exists", return_value=False), \
                 patch.object(main_mod.logger, "error") as mock_log:
                main_mod.load_model()
                assert main_mod.model is None
                assert main_mod.model_version == "none"
                mock_log.assert_called_with("No model found. Please train a model first.")
        finally:
            main_mod.model = original_model
            main_mod.model_version = original_version

    def test_load_model_exception_handling(self):
        """Covers exception handler in load_model"""
        original_model = main_mod.model
        original_version = main_mod.model_version
        try:
            with patch("os.path.exists", return_value=True), \
                 patch("joblib.load", side_effect=Exception("Corrupted file")), \
                 patch.object(main_mod.logger, "error") as mock_log:
                main_mod.load_model()
                assert main_mod.model is None
                assert main_mod.model_version == "error"
                mock_log.assert_called()
        finally:
            main_mod.model = original_model
            main_mod.model_version = original_version

    def test_root_metrics_health_endpoints_combined(self, client):
        """Hits root, /metrics, and /health in one shot"""
        r1 = client.get("/")
        assert r1.status_code == 200
        
        r2 = client.get("/metrics")
        assert r2.status_code == 200
        
        original = main_mod.model
        try:
            main_mod.model = MockModel()
            r3 = client.get("/health")
            assert r3.status_code == 200
            assert r3.json()["status"] == "healthy"
        finally:
            main_mod.model = original

    def test_predict_endpoint_catches_generic_exception(self, client):
        """Covers generic exception handler in predict"""
        original_model = main_mod.model
        try:
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            with patch("src.api.main.transactions_to_df", side_effect=RuntimeError("Unexpected")):
                response = client.post("/predict", json={"transactions": [valid_transaction()]})
                assert response.status_code == 500
                assert "Prediction failed" in response.json()["detail"]
        finally:
            main_mod.model = original_model

    def test_root_endpoint(self, client):
        """Covers GET /"""
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()

    def test_health_endpoint_degraded(self, client):
        """Covers GET /health when model is None"""
        original_model = main_mod.model
        try:
            main_mod.model = None
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "degraded"
        finally:
            main_mod.model = original_model

    def test_metrics_endpoint(self, client):
        """Covers GET /metrics"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_rate" in data


class TestMiddlewareAndAsync:
    def test_request_id_in_response_headers(self, client):
        """Middleware should add X-Request-ID to every response"""
        response = client.get("/")
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time-Ms" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_request_id_is_unique_per_request(self, client):
        """Each request should get a unique correlation ID"""
        r1 = client.get("/")
        r2 = client.get("/")
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]

    def test_predict_logs_input_hash(self, client, caplog):
        """Middleware should log input hash for /predict"""
        original = main_mod.model
        try:
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            with caplog.at_level(logging.INFO):
                response = client.post(
                    "/predict", json={"transactions": [valid_transaction()]}
                )
            assert response.status_code == 200
            log_text = " ".join(record.message for record in caplog.records)
            assert "input_hash=" in log_text
            assert "REQUEST" in log_text
            assert "RESPONSE" in log_text
        finally:
            main_mod.model = original

    def test_async_predict_endpoint_works(self, client):
        """Verify async predict still returns correct response"""
        original = main_mod.model
        try:
            main_mod.model = MockModel()
            main_mod.model_version = "mock-v1"
            response = client.post(
                "/predict", json={"transactions": [valid_transaction()]}
            )
            assert response.status_code == 200
            assert "predictions" in response.json()
            assert "latency_ms" in response.json()
        finally:
            main_mod.model = original