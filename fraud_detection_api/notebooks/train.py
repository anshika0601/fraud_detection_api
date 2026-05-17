"""
Fast Random Forest training for Fraud Detection with MLflow tracking.

"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_random_forest")


def load_data():
    """Load preprocessed data (train/val/test)."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def train():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Fast parameters
    params = {
        "n_estimators": 50,           # fewer trees
        "max_depth": 10,              # shallower trees
        "min_samples_split": 10,      # prevents deep splits
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,                 # use all CPU cores
        "max_features": "sqrt",
    }

    with mlflow.start_run(run_name="Random_Forest_Fast"):
        mlflow.log_params(params)
        logger.info("Training Random Forest ...")

        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # Validation predictions
        y_prob_val = rf.predict_proba(X_val)[:, 1]
        y_pred_val = rf.predict(X_val)

        # Metrics
        roc_auc = roc_auc_score(y_val, y_prob_val)
        pr_auc = average_precision_score(y_val, y_prob_val)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
        fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.log_metric("val_pr_auc", pr_auc)
        mlflow.log_metric("val_fraud_recall", fraud_recall)
        mlflow.log_metric("val_false_positive_rate", fpr)

        print(f"✅ Validation PR-AUC: {pr_auc:.4f}, Fraud Recall: {fraud_recall:.2%}, FPR: {fpr:.4%}")

        # Test evaluation
        y_prob_test = rf.predict_proba(X_test)[:, 1]
        test_roc = roc_auc_score(y_test, y_prob_test)
        test_pr = average_precision_score(y_test, y_prob_test)
        mlflow.log_metric("test_roc_auc", test_roc)
        mlflow.log_metric("test_pr_auc", test_pr)
        print(f"✅ Test PR-AUC: {test_pr:.4f}")

        # Quick confusion matrix plot (optional – skip to save time)
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_val, y_pred_val)
        ax.matshow(cm, cmap='Blues', alpha=0.7)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        ax.set_xticklabels(['', 'Legit', 'Fraud'])
        ax.set_yticklabels(['', 'Legit', 'Fraud'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(DATA_DIR, 'rf_cm_fast.png')
        plt.savefig(cm_path, dpi=100)
        mlflow.log_artifact(cm_path)
        plt.close(fig)

        # Save model locally
        joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_fast.pkl'))
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Save metrics
        with open(os.path.join(MODELS_DIR, 'random_forest_fast_metrics.json'), 'w') as f:
            json.dump({
                "params": params,
                "val_pr_auc": pr_auc,
                "test_pr_auc": test_pr,
                "fraud_recall": fraud_recall,
                "fpr": fpr
            }, f, indent=2)

      


if __name__ == "__main__":
    train()