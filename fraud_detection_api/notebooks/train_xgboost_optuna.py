"""XGBoost hyperparameter tuning with Optuna and MLflow.

This script compares XGBoost models using PR-AUC and ROC-AUC,
selects the best model, and saves the final pickle artifact.
"""

import os
import sys
import json
import warnings
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
MLFLOW_TRACKING_URI = Path(os.path.join(PROJECT_DIR, 'mlruns')).absolute().as_uri()
EXPERIMENT_NAME = 'fraud_detection_xgboost_optuna'
MODEL_NAME = 'fraud_detector_xgb_v1'


def _ensure_preprocessed_csvs() -> None:
    required = [
        os.path.join(DATA_DIR, 'X_train.csv'),
        os.path.join(DATA_DIR, 'X_val.csv'),
        os.path.join(DATA_DIR, 'X_test.csv'),
        os.path.join(DATA_DIR, 'y_train.csv'),
        os.path.join(DATA_DIR, 'y_val.csv'),
        os.path.join(DATA_DIR, 'y_test.csv'),
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if not missing:
        return

    print(f"⚠️ Missing preprocessed CSVs ({len(missing)}). Running preprocessing...")
    preprocess_script = os.path.join(PROJECT_DIR, 'preprocess_data.py')
    if not os.path.exists(preprocess_script):
        raise FileNotFoundError(f"Preprocess script not found: {preprocess_script}")

    subprocess.check_call([sys.executable, preprocess_script], cwd=PROJECT_DIR)


def load_data():
    _ensure_preprocessed_csvs()

    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))

    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

    print(f"✅ Train: {X_train.shape} (fraud: {y_train.mean():.4%})")
    print(f"✅ Validation: {X_val.shape} (fraud: {y_val.mean():.4%})")
    print(f"✅ Test: {X_test.shape} (fraud: {y_test.mean():.4%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(model, X, y, label):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    accuracy = (y_pred == y).mean()
    fraud_recall = (y_pred[y == 1] == 1).mean() if (y == 1).sum() > 0 else 0.0
    fraud_precision = (y_pred[y_pred == 1] == 1).mean() if (y_pred == 1).sum() > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'fraud_recall': fraud_recall,
        'fraud_precision': fraud_precision,
        'false_positive_rate': false_positive_rate,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }

    print(f"\n📊 {label} metrics:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"   Fraud Recall: {fraud_recall:.2%}")
    print(f"   Fraud Precision: {fraud_precision:.2%}")
    print(f"   False Positive Rate: {false_positive_rate:.4%}")

    return metrics


def save_model_artifacts(model, best_params, test_metrics):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    local_model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}.pkl')
    joblib.dump(model, local_model_path)
    print(f"✅ Final model saved to: {local_model_path}")

    metrics_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'test_metrics': {k: float(v) for k, v in test_metrics.items() if isinstance(v, (np.floating, float))},
        }, f, indent=2)

    print(f"✅ Metrics saved to: {metrics_path}")
    return local_model_path, metrics_path


def train_xgboost_optuna(n_trials: int = 20):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
    print(f"\n⚖️ Class imbalance ratio: {num_neg}:{num_pos} -> scale_pos_weight={scale_pos_weight:.2f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 0,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='aucpr',
            early_stopping_rounds=30,
            verbose=False,
        )

        val_metrics = calculate_metrics(model, X_val, y_val, 'Validation')

        with mlflow.start_run(run_name=f'XGBoost_Optuna_trial_{trial.number}') as run:
            mlflow.log_params(params)
            mlflow.log_metric('val_roc_auc', val_metrics['roc_auc'])
            mlflow.log_metric('val_pr_auc', val_metrics['pr_auc'])
            mlflow.log_metric('val_fraud_recall', val_metrics['fraud_recall'])
            mlflow.log_metric('val_fraud_precision', val_metrics['fraud_precision'])
            mlflow.log_metric('val_false_positive_rate', val_metrics['false_positive_rate'])
            mlflow.log_param('trial_number', trial.number)

        return val_metrics['pr_auc']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"\n✅ Optuna completed. Best trial #{best_trial.number} with PR-AUC={best_trial.value:.4f}")
    print(f"   Best params: {best_params}")

    final_params = {
        **best_params,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
    }
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        pd.concat([X_train, X_val], axis=0),
        pd.concat([y_train, y_val], axis=0),
        eval_set=[(X_test, y_test)],
        eval_metric='aucpr',
        early_stopping_rounds=30,
        verbose=False,
    )

    test_metrics = calculate_metrics(final_model, X_test, y_test, 'Test')
    model_path, metrics_path = save_model_artifacts(final_model, final_params, test_metrics)

    with mlflow.start_run(run_name='XGBoost_Optuna_Final') as run:
        mlflow.log_params(final_params)
        mlflow.log_metrics({
            'test_roc_auc': test_metrics['roc_auc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'test_fraud_recall': test_metrics['fraud_recall'],
            'test_fraud_precision': test_metrics['fraud_precision'],
            'test_false_positive_rate': test_metrics['false_positive_rate'],
        })
        mlflow.xgboost.log_model(final_model, 'model')

    print(f"\n✅ Best XGBoost model saved and logged. Model file: {model_path}")
    return best_trial, final_model, test_metrics


if __name__ == '__main__':
    train_xgboost_optuna(n_trials=20)
