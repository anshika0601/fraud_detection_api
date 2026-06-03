"""
Data drift detection using Evidently.
Monitors prediction distributions and data quality over time.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset


class DriftDetector:
    """Detects data drift in fraud detection predictions."""

    def __init__(self, reference_data: pd.DataFrame, predictions_csv: str = "logs/predictions.csv"):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Training data distribution to compare against
            predictions_csv: Path to CSV with logged predictions
        """
        self.reference_data = reference_data
        self.predictions_csv = predictions_csv

    def load_recent_predictions(self, n_rows: Optional[int] = None) -> pd.DataFrame:
        """Load recent predictions from CSV."""
        if not os.path.exists(self.predictions_csv):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_csv}")

        df = pd.read_csv(self.predictions_csv)
        if n_rows:
            df = df.tail(n_rows)
        return df

    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "reports/drift_report.html"
    ) -> Tuple[Report, dict]:
        """
        Generate drift report comparing current predictions to reference.

        Args:
            current_data: Recent prediction data
            output_path: Where to save HTML report

        Returns:
            Tuple of (Report object, summary metrics dict)
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Select only feature columns (not metadata)
        feature_cols = [col for col in current_data.columns
                       if col not in ['timestamp', 'prediction', 'probability', 'request_id']]

        # Subset data
        ref_subset = self.reference_data[feature_cols]
        curr_subset = current_data[feature_cols]

        # Create drift detection report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(reference_data=ref_subset, current_data=curr_subset)

        # Save HTML report
        report.save_html(output_path)

        # Extract summary metrics
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_rows": len(ref_subset),
            "current_rows": len(curr_subset),
            "output_path": output_path,
        }

        return report, summary

    def detect_prediction_drift(
        self,
        current_predictions: pd.Series,
        threshold: float = 0.05
    ) -> dict:
        """
        Detect drift in prediction distribution (fraud rate).

        Args:
            current_predictions: Recent predictions (0/1)
            threshold: Alert if fraud rate changes by more than this

        Returns:
            Dict with drift metrics and alerts
        """
        # Fraud rate in reference (assume 0.5% fraud)
        ref_fraud_rate = self.reference_data.get('target', pd.Series([0])).mean()

        # Fraud rate in current
        curr_fraud_rate = current_predictions.mean()

        # Calculate drift
        drift_pct = abs(curr_fraud_rate - ref_fraud_rate)

        return {
            "reference_fraud_rate": float(ref_fraud_rate),
            "current_fraud_rate": float(curr_fraud_rate),
            "drift_percentage": float(drift_pct),
            "is_drifted": drift_pct > threshold,
            "threshold": threshold,
        }


def load_reference_data(path: str = "data/train_processed.csv") -> pd.DataFrame:
    """Load reference training data for comparison."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Reference data not found: {path}")
