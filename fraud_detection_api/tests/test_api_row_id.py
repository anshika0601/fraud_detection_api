import numpy as np

import sys
import os

# Ensure repo root is on sys.path for test execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)



class _DummyModelWithFeatureNames:
    def __init__(self, feature_names_in_):
        self.feature_names_in_ = np.array(feature_names_in_, dtype=object)


class _DummyTransaction:
    def __init__(self, **kwargs):
        self._d = kwargs

    def dict(self):
        return self._d


def _make_transactions(n=3):
    base = {
        "Time": 0.0,
        **{f"V{i}": 0.1 * i for i in range(1, 29)},
        "Amount": 10.0,
    }
    txs = []
    for k in range(n):
        d = dict(base)
        d["Time"] = float(k)
        d["Amount"] = 10.0 + k
        txs.append(_DummyTransaction(**d))
    return txs


def test_row_id_added_when_model_requires_it(monkeypatch):
    # Arrange: model requires __row_id
    required_features = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "__row_id",
    ]
    dummy = _DummyModelWithFeatureNames(required_features)

    import fraud_detection_api.src.api.main as api_main

    monkeypatch.setattr(api_main, "model", dummy)

    # Act
    df = api_main.transactions_to_df(_make_transactions(4))

    # Assert
    assert "__row_id" in df.columns
    assert df["__row_id"].tolist() == [0, 1, 2, 3]


def test_row_id_not_added_when_model_does_not_require_it(monkeypatch):
    # Arrange: model does NOT require __row_id
    required_features = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ]
    dummy = _DummyModelWithFeatureNames(required_features)

    import fraud_detection_api.src.api.main as api_main

    monkeypatch.setattr(api_main, "model", dummy)

    # Act
    df = api_main.transactions_to_df(_make_transactions(2))

    # Assert
    assert "__row_id" not in df.columns
    assert list(df.columns) == required_features

