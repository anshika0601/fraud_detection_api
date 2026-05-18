# Fraud Detection API

This repository contains the LightGBM fraud detection training pipeline with Optuna hyperparameter tuning and MLflow experiment tracking.

## Final Model
- Saved final model: `fraud_detection_api/models/fraud_detector_v1.pkl`
- Final test ROC-AUC: `0.9755`
- MLflow registered model name: `fraud-detector-v1`
- Registered model version: `1`

## Notes
- Run `mlflow ui` from the repository root and open `http://localhost:5000` to inspect runs.
- The training script is `fraud_detection_api/notebooks/train_lightGBM.py`.
