"""XGBoost training with MLflow for Fraud Detection.

Fixes:
- Robust file paths (works regardless of current working directory)
- Auto-runs preprocessing if preprocessed CSVs are missing
- Safe handling for scale_pos_weight ratio divide-by-zero
"""

import os
import sys
import json
import warnings
import subprocess

import pandas as pd
import numpy as np

import mlflow

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore')


# -------------------------
# Path helpers
# -------------------------

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


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
    
#mlflow tracking setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_lightgbm_tuning")    


def load_data():
    """Load preprocessed data (auto-runs preprocessing if missing)."""
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


def calculate_metrics(model, X, y, split_name: str):
    """Calculate all relevant metrics"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'accuracy': (y_pred == y).mean(),
        'fraud_recall': (y_pred[y == 1] == 1).mean() if (y == 1).sum() > 0 else 0,
        'fraud_precision': (y_pred[y_pred == 1] == 1).mean() if (y_pred == 1).sum() > 0 else 0,
    }

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n📊 {split_name} Metrics:")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Fraud Recall: {metrics['fraud_recall']:.2%}")
    print(f"   Fraud Precision: {metrics['fraud_precision']:.2%}")
    print(f"   False Positive Rate: {metrics['false_positive_rate']:.4%}")

    return metrics


def plot_feature_importance(model, feature_names, save_path: str):
    importance = model.feature_importances_

    importance_df = (
        pd.DataFrame({'feature': feature_names, 'importance': importance})
        .sort_values('importance', ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)

    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Feature Importance (Freq)', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost - Top 20 Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, importance_df['importance'].values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Feature importance plot saved to: {save_path}")
    return importance_df


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=['Legitimate', 'Fraudulent'],
        yticklabels=['Legitimate', 'Fraudulent'],
    )

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')

    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5,
        -0.1,
        f'TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}',
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {save_path}")


def plot_roc_pr_curves(model, X_val, y_val, save_path: str):
    """Save ROC & PR curves."""
    from sklearn.metrics import roc_curve, precision_recall_curve

    y_score = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_score)
    prec, rec, _ = precision_recall_curve(y_val, y_score)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(fpr, tpr, label='ROC')
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend()

    ax[1].plot(rec, prec, label='PR')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Performance curves saved to: {save_path}")


def train_xgboost():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # scale_pos_weight = (#neg/#pos) so handle divide-by-zero safely
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"\n⚖️ Class imbalance ratio (neg/pos): {ratio:.1f}:1 (neg={n_neg}, pos={n_pos})")

    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': ratio,
        'random_state': 42,
        'eval_metric': 'aucpr',
        'use_label_encoder': False,
        'verbosity': 0,
        'min_child_weight': 1,
        'gamma': 0,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud_detection_xgboost")

    print("\n🚀 Starting MLflow run...")

    with mlflow.start_run(run_name="XGBoost_Fraud_Detection") as run:
        mlflow.log_params(params)

        print("🏋️ Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        print("✅ Model training complete!")

        print("\n📊 Calculating metrics...")
        train_metrics = calculate_metrics(model, X_train, y_train, "Training")
        val_metrics = calculate_metrics(model, X_val, y_val, "Validation")
        test_metrics = calculate_metrics(model, X_test, y_test, "Test")

        for metric_name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"val_{metric_name}", value)
        for metric_name, value in test_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"test_{metric_name}", value)

        # Artifacts
        fi_png = os.path.join(DATA_DIR, 'xgboost_feature_importance.png')
        fi_csv = os.path.join(DATA_DIR, 'xgboost_feature_importance.csv')
        importance_df = plot_feature_importance(model, X_train.columns, fi_png)
        importance_df.to_csv(fi_csv, index=False)
        mlflow.log_artifact(fi_png)
        mlflow.log_artifact(fi_csv)

        curves_png = os.path.join(DATA_DIR, 'xgboost_performance_curves.png')
        plot_roc_pr_curves(model, X_val, y_val, curves_png)
        mlflow.log_artifact(curves_png)

        y_val_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        cm_png = os.path.join(DATA_DIR, 'xgboost_confusion_matrix.png')
        plot_confusion_matrix(y_val, y_val_pred, cm_png)
        mlflow.log_artifact(cm_png)

        # Model saving
        model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path}")
        mlflow.xgboost.log_model(model, "xgboost_model")

        all_metrics = {
            'train': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in train_metrics.items()},
            'val': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in val_metrics.items()},
            'test': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_metrics.items()},
            'params': params,
            'run_id': run.info.run_id,
        }

        metrics_json = os.path.join(MODELS_DIR, 'xgboost_metrics.json')
        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"✅ Metrics saved to: {metrics_json}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE! SUMMARY")
        print("=" * 60)
        print(f"\n🏆 XGBoost Performance:")
        print(f"   Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"   Validation Fraud Recall: {val_metrics['fraud_recall']:.2%}")
        print(f"   Validation False Positive Rate: {val_metrics['false_positive_rate']:.4%}")

        print(f"\n📊 Test Performance (Final):")
        print(f"   Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"   Test Fraud Recall: {test_metrics['fraud_recall']:.2%}")

        print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

        return model, all_metrics


def load_and_predict_example(model_path: str | None = None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')

    model = joblib.load(model_path)
    print("✅ Model loaded successfully")

    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    sample = X_test.iloc[0:1]

    proba = model.predict_proba(sample)[0][1]
    prediction = "FRAUD" if proba > 0.5 else "LEGIT"

    print(f"\n🔮 Sample Prediction:")
    print(f"   Fraud Probability: {proba:.2%}")
    print(f"   Prediction: {prediction}")

    return model


if __name__ == '__main__':
    model, metrics = train_xgboost()
    load_and_predict_example()

