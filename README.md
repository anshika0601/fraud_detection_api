# Fraud Detection API

This repository contains the LightGBM fraud detection training pipeline with Optuna hyperparameter tuning and MLflow experiment tracking.

## Final Model
- Saved final LightGBM model: `fraud_detection_api/models/fraud_detector_v1.pkl`
- Saved final XGBoost model: `fraud_detection_api/models/fraud_detector_xgb_v1.pkl`
- Final XGBoost test ROC-AUC: `0.9832`
- Final XGBoost test PR-AUC: `0.8694`
- MLflow registered model name: `fraud-detector-v1`
- Registered model version: `1`

## Notes
- Run `mlflow ui` from the repository root and open `http://localhost:5000` to inspect runs.
- LightGBM training script: `fraud_detection_api/notebooks/train_lightGBM.py`.
- XGBoost Optuna tuning script: `fraud_detection_api/notebooks/train_xgboost_optuna.py`.
