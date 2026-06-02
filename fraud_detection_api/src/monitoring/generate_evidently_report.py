"""Generate an Evidently data drift (and optional model performance) report.

Usage examples:
python src/monitoring/generate_evidently_report.py \
  --reference-features data/X_train.csv \
  --reference-target data/y_train.csv \
  --current-features data/X_test.csv \
  --current-target data/y_test.csv \
  --model models/fraud_detector_xgb_v1.pkl \
  --output reports/data_drift_report.html

The script will:
- load reference and current feature CSVs
- optionally merge target files if provided
- optionally load a model to produce `prediction` and `prediction_proba` on current features
- run Evidently Profile with data drift (+ classification performance if target+prediction available)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import joblib
import pandas as pd

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection


def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_target(path: str, col_name: str = "Class") -> pd.Series:
    s = pd.read_csv(path)
    if s.shape[1] == 1:
        s.columns = [col_name]
        return s[col_name]
    # If target file contains header
    if col_name in s.columns:
        return s[col_name]
    # otherwise pick first column
    return s.iloc[:, 0]


def merge_features_target(X: pd.DataFrame, y: pd.Series, target_name: str = "Class") -> pd.DataFrame:
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    X[target_name] = y
    return X


def predict_with_model(model_path: str, X: pd.DataFrame):
    model = joblib.load(model_path)
    # Predict labels
    try:
        preds = model.predict(X)
    except Exception:
        preds = None
    # Predict probabilities if available
    proba = None
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = None
    return preds, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-features", default="data/X_train.csv")
    parser.add_argument("--reference-target", default="data/y_train.csv", help="optional target CSV for reference (used for performance section)")
    parser.add_argument("--current-features", default="data/X_test.csv")
    parser.add_argument("--current-target", default=None, help="optional target CSV for current data (used for performance section)")
    parser.add_argument("--model", default="models/fraud_detector_xgb_v1.pkl", help="optional model to generate predictions for current features")
    parser.add_argument("--output", default="reports/data_drift_report.html")
    parser.add_argument("--target-name", default="Class")
    args = parser.parse_args()

    # Load reference/current feature data
    print(f"Loading reference features from {args.reference_features}")
    ref_X = load_features(args.reference_features)
    print(f"Loading current features from {args.current_features}")
    curr_X = load_features(args.current_features)

    # Prepare lists of profile sections
    sections = [DataDriftProfileSection()]

    # Try to prepare classification performance data if target exists or model provided
    have_target = False
    if args.current_target and Path(args.current_target).exists():
        print(f"Loading current target from {args.current_target}")
        curr_y = load_target(args.current_target, col_name=args.target_name)
        have_target = True
    else:
        curr_y = None

    # If model is provided and exists, produce predictions for current features
    preds = None
    proba = None
    if args.model and Path(args.model).exists():
        print(f"Loading model and generating predictions from {args.model}")
        preds, proba = predict_with_model(args.model, curr_X)
        if preds is not None:
            curr_X = curr_X.reset_index(drop=True).copy()
            curr_X["prediction"] = preds
            have_target = have_target or (curr_y is not None)
        if proba is not None:
            curr_X["prediction_proba"] = proba

    # If we have both prediction and target, include classification performance section
    if have_target and ("prediction" in curr_X.columns or preds is not None):
        # merge target into curr_X if not already
        if curr_y is not None and args.target_name not in curr_X.columns:
            curr_X = merge_features_target(curr_X, curr_y, target_name=args.target_name)
        sections.append(ClassificationPerformanceProfileSection())

    profile = Profile(sections=sections)

    # Run profile: DataDriftProfileSection expects feature-only DataFrames
    print("Calculating profile (this may take a moment)...")
    profile.calculate(reference_df=ref_X, current_df=curr_X)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.save_html(str(output_path))
    print(f"Saved Evidently report to {output_path}")


if __name__ == "__main__":
    main()
