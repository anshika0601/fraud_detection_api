"""
Training script for Fraud Detection using Random Forest and MLflow.
Logs parameters, metrics, and the model artifact.
"""

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

def _ensure_preprocessed_csvs() -> None:
    """Ensure data/X_{split}.csv and data/y_{split}.csv exist.

    If missing, run preprocess_data.py from the project root.
    """
    required = [
        os.path.join(DATA_DIR, 'X_train.csv'),
        os.path.join(DATA_DIR, 'X_val.csv'),
        os.path.join(DATA_DIR, 'y_train.csv'),
        os.path.join(DATA_DIR, 'y_val.csv'),
     
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if not missing:
        return

    print(f"⚠️ Missing preprocessed CSVs ({len(missing)}). Running preprocessing...")

    preprocess_script = os.path.join(PROJECT_DIR, 'preprocess_data.py')
    if not os.path.exists(preprocess_script):
        raise FileNotFoundError(f"Preprocess script not found: {preprocess_script}")

    import sys
    import subprocess

    subprocess.check_call([sys.executable, preprocess_script], cwd=PROJECT_DIR)


#mlflow tracking setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_random_forest")


def load_data():
    """Load preprocessed data"""
    print("\n📂 Loading preprocessed data...")
    
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze()

    
    print(f"✅ Train: {X_train.shape} (fraud: {y_train.mean():.4%})")
    print(f"✅ Validation: {X_val.shape} (fraud: {y_val.mean():.4%})")
    
    return X_train, X_val, y_train, y_val


def train():
    # Set experiment name
    mlflow.set_experiment("Fraud_Detection")
    
    # Define parameters
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    X_train, X_val, y_train, y_val = load_data()
    
    with mlflow.start_run():
        logger.info(f"Starting MLflow run with n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Initialize and train model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest model...")
        rf.fit(X_train, y_train)
        
        # Predictions
        logger.info("Evaluating model...")
        y_pred = rf.predict(X_val)
        y_prob = rf.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc_roc = roc_auc_score(y_val, y_prob)
        
        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)
        
        logger.info(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc_roc:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # Save model locally as well (optional)
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/random_forest_fraud_model.pkl')
        logger.info("Model saved to models/random_forest_fraud_model.pkl and MLflow artifacts")

if __name__ == "__main__":
    train()
