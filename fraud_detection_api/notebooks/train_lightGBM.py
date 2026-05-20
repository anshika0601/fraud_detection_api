"""
LightGBM Training with Hyperparameter Tuning via MLflow.
Tunes learning_rate and num_leaves, logs all metrics and artifacts.

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Path helpers
# -------------------------

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
MLFLOW_TRACKING_URI = Path(os.path.join(PROJECT_DIR, 'mlruns')).absolute().as_uri()
MODEL_REGISTRY_NAME = 'fraud-detector-v1'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('fraud_detection_lightgbm_tuning')


def _ensure_preprocessed_csvs() -> None:
    """Ensure data/X_{split}.csv and data/y_{split}.csv exist.

    If missing, run preprocess_data.py from the project root.
    """
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

    import sys
    import subprocess

    subprocess.check_call([sys.executable, preprocess_script], cwd=PROJECT_DIR)


print("="*60)
print("LIGHTGBM HYPERPARAMETER TUNING WITH MLFLOW")
print("="*60)


def load_data():
    """Load preprocessed data"""
    print("\n📂 Loading preprocessed data...")
    
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


def calculate_metrics(model, X, y, split_name):
    """Calculate all relevant metrics"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'accuracy': (y_pred == y).mean(),
        'fraud_recall': (y_pred[y==1] == 1).mean() if (y==1).sum() > 0 else 0,
        'fraud_precision': (y_pred[y_pred==1] == 1).mean() if (y_pred==1).sum() > 0 else 0
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics


def plot_feature_importance(model, feature_names, learning_rate, num_leaves, save_path):
    """Plot LightGBM feature importance"""
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'LightGBM Feature Importance\n(lr={learning_rate}, leaves={num_leaves})', 
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df


def plot_roc_pr_curves(model, X, y, learning_rate, num_leaves, save_path):
    """Plot ROC and Precision-Recall curves"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    axes[0].plot(fpr, tpr, linewidth=2, color='darkorange', 
                 label=f'LightGBM (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)
    
    axes[1].plot(recall, precision, linewidth=2, color='green', 
                 label=f'LightGBM (PR-AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall (Fraud Detection Rate)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'LightGBM Performance Curves\n(lr={learning_rate}, leaves={num_leaves})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def sanitize_mlflow_model_name(name: str) -> str:
    """MLflow registered model artifact name constraints.

    MLflow model name must be non-empty and cannot contain: ('/', ':', '.', '%', '"', "'')
    """
    forbidden = ['/', ':', '.', '%', '"', "'"]
    safe = str(name).strip()
    for ch in forbidden:
        safe = safe.replace(ch, '_')
    safe = '_'.join(safe.split())
    return safe or 'lightgbm_model'


def tune_lightgbm(n_trials: int = 20):
    """Hyperparameter tuning for LightGBM using Optuna."""
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Calculate class imbalance ratio
    ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"\n⚖️ Class imbalance ratio: {ratio:.1f}:1")
    
    fixed_params = {
        'n_estimators': 100,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'min_child_samples': 20,
        'scale_pos_weight': ratio,
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True,
        'n_jobs': -1
    }
    
    trial_run_ids = {}
    trial_results = {}
    
    def objective(trial):
        lr = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        leaves = trial.suggest_categorical('num_leaves', [15, 31, 63, 127, 255])
        
        params = {
            'learning_rate': lr,
            'num_leaves': leaves,
            **fixed_params
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        val_metrics = calculate_metrics(model, X_val, y_val, 'Validation')
        trial_results[trial.number] = {
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
            'val_recall': val_metrics['fraud_recall'],
            'val_precision': val_metrics['fraud_precision'],
            'val_fpr': val_metrics['false_positive_rate']
        }
        
        run_name = f'Optuna_lr{lr}_leaves{leaves}_trial{trial.number}'
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({'learning_rate': lr, 'num_leaves': leaves, **fixed_params})
            mlflow.log_metric('val_roc_auc', val_metrics['roc_auc'])
            mlflow.log_metric('val_pr_auc', val_metrics['pr_auc'])
            mlflow.log_metric('val_recall', val_metrics['fraud_recall'])
            mlflow.log_metric('val_precision', val_metrics['fraud_precision'])
            mlflow.log_metric('val_false_positive_rate', val_metrics['false_positive_rate'])
            mlflow.log_param('optuna_trial', trial.number)
            trial_run_ids[trial.number] = run.info.run_id
        
        return val_metrics['roc_auc']
    
    print(f"\n🚀 Starting Optuna tuning for {n_trials} trials")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = {
        'learning_rate': best_trial.params['learning_rate'],
        'num_leaves': best_trial.params['num_leaves']
    }
    best_run_id = trial_run_ids.get(best_trial.number)
    
    print(f"\n✅ Optuna completed. Best trial: {best_trial.number} | val_roc_auc={best_trial.value:.4f}")
    print(f"   Best params: {best_params}")
    
    # Final model training on train + validation for deployment
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_params = {
        **fixed_params,
        **best_params
    }
    
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_train_full, y_train_full,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    test_metrics = calculate_metrics(final_model, X_test, y_test, 'Test')
    
    all_results = []
    for trial in study.trials:
        trial_params = trial.params
        metric_record = trial_results.get(trial.number, {})
        all_results.append({
            'trial_number': trial.number,
            'learning_rate': trial_params['learning_rate'],
            'num_leaves': trial_params['num_leaves'],
            'val_roc_auc': metric_record.get('val_roc_auc', trial.value),
            'val_pr_auc': metric_record.get('val_pr_auc', np.nan),
            'val_recall': metric_record.get('val_recall', np.nan),
            'val_precision': metric_record.get('val_precision', np.nan),
            'val_fpr': metric_record.get('val_fpr', np.nan),
            'run_id': trial_run_ids.get(trial.number)
        })
    
    return all_results, best_params, final_model, best_run_id, (X_train, X_val, X_test, y_train, y_val, y_test, test_metrics)


def analyze_results(all_results):
    """Analyze and visualize tuning results"""
    print("\n" + "="*60)
    print("TUNING RESULTS ANALYSIS")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    if 'val_roc_auc' in results_df.columns:
        results_df = results_df.sort_values('val_roc_auc', ascending=False)
    else:
        results_df = results_df.sort_values(results_df.columns[0], ascending=False)
    
    print("\n📊 Top 5 Configurations:")
    cols = ['learning_rate', 'num_leaves', 'val_roc_auc', 'val_fpr']
    print(results_df[cols].head(5).to_string(index=False))
    
    print("\n📊 Bottom 5 Configurations:")
    print(results_df[cols].tail(5).to_string(index=False))
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pivot_auc = results_df.pivot(index='learning_rate', columns='num_leaves', values='val_roc_auc')
    sns.heatmap(pivot_auc, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('ROC-AUC Heatmap', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Num Leaves', fontsize=12)
    axes[0,0].set_ylabel('Learning Rate', fontsize=12)
    
    for leaves in results_df['num_leaves'].unique():
        subset = results_df[results_df['num_leaves'] == leaves]
        axes[0,1].plot(subset['learning_rate'], subset['val_roc_auc'], 
                       marker='o', label=f'leaves={leaves}', linewidth=2)
    axes[0,1].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    axes[0,1].set_title('ROC-AUC vs Learning Rate', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    for lr in results_df['learning_rate'].unique():
        subset = results_df[results_df['learning_rate'] == lr]
        axes[1,0].plot(subset['num_leaves'], subset['val_roc_auc'], 
                       marker='s', label=f'lr={lr}', linewidth=2)
    axes[1,0].set_xlabel('Num Leaves', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    axes[1,0].set_title('ROC-AUC vs Num Leaves', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xscale('log')
    
    if 'val_recall' in results_df.columns and 'val_precision' in results_df.columns:
        scatter = axes[1,1].scatter(results_df['val_recall'], results_df['val_precision'], 
                                    c=results_df['val_roc_auc'], cmap='viridis', 
                                    s=100, alpha=0.6)
        axes[1,1].set_xlabel('Recall (Fraud Detection Rate)', fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Precision', fontsize=12, fontweight='bold')
        axes[1,1].set_title('Recall-Precision Trade-off', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,1], label='ROC-AUC')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].axis('off')
    
    plt.suptitle('LightGBM Hyperparameter Tuning Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    results_path = os.path.join(DATA_DIR, 'lightgbm_tuning_results.png')
    plt.savefig(results_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    results_df.to_csv(os.path.join(MODELS_DIR, 'lightgbm_tuning_results.csv'), index=False)
    print("\n✅ Results saved to: models/lightgbm_tuning_results.csv")
    print("✅ Visualization saved to: data/lightgbm_tuning_results.png")
    
    return results_df


def save_best_model(best_model, best_params, best_run_id, X_test, y_test, test_metrics):
    """Save the best model, log final metrics, and register the model."""
    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)
    
    print(f"\n🏆 Best Configuration:")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   Num Leaves: {best_params['num_leaves']}")
    print(f"   MLflow Best Trial Run ID: {best_run_id}")
    
    print(f"\n📊 Final Test Performance:")
    print(f"   PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"   ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Fraud Recall: {test_metrics['fraud_recall']:.2%}")
    print(f"   Fraud Precision: {test_metrics['fraud_precision']:.2%}")
    print(f"   False Positive Rate: {test_metrics['false_positive_rate']:.4%}")
    
    print("\n🎨 Generating final artifacts for best model...")
    
    importance_df = plot_feature_importance(
        best_model, X_test.columns, 
        best_params['learning_rate'], best_params['num_leaves'],
        'data/lightgbm_best_feature_importance.png'
    )
    
    plot_roc_pr_curves(
        best_model, X_test, y_test,
        best_params['learning_rate'], best_params['num_leaves'],
        'data/lightgbm_best_performance_curves.png'
    )
    
    y_pred = (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Best LightGBM Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/lightgbm_best_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_model_path = os.path.join(MODELS_DIR, 'fraud_detector_v1.pkl')
    joblib.dump(best_model, local_model_path)
    print(f"✅ Best model saved to: {local_model_path}")
    
    best_metrics = {
        'best_params': best_params,
        'best_run_id': best_run_id,
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in test_metrics.items()},
        'top_features': importance_df.head(10)[['feature', 'importance']].to_dict('records')
    }
    
    with open(os.path.join(MODELS_DIR, 'fraud_detector_v1_metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print(f"✅ Best metrics saved to: {os.path.join(MODELS_DIR, 'fraud_detector_v1_metrics.json')}")
    
    print("\n📦 Registering final model to MLflow Model Registry...")
    with mlflow.start_run(run_name='Best_LightGBM_Final') as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            'test_roc_auc': test_metrics['roc_auc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'test_fraud_recall': test_metrics['fraud_recall'],
            'test_fraud_precision': test_metrics['fraud_precision'],
            'test_false_positive_rate': test_metrics['false_positive_rate']
        })
        mlflow.lightgbm.log_model(best_model, 'model')
        model_uri = f"runs:/{run.info.run_id}/model"

    try:
        registered_model = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
        print(f"✅ Registered model '{MODEL_REGISTRY_NAME}' version {registered_model.version}")
    except Exception as exc:
        print(f"⚠️ Model registration failed: {exc}")
        registered_model = None
    
    return test_metrics, local_model_path, registered_model


def main():
    """Main execution"""
    
    # Run Optuna hyperparameter tuning
    all_results, best_params, best_model, best_run_id, data = tune_lightgbm(n_trials=20)
    X_train, X_val, X_test, y_train, y_val, y_test, test_metrics = data
    
    # Analyze results
    results_df = analyze_results(all_results)
    
    # Save and register best model
    test_metrics, model_path, registered_model = save_best_model(
        best_model, best_params, best_run_id, X_test, y_test, test_metrics
    )
    
    print("\n" + "="*60)
    print("TUNING COMPLETE! SUMMARY")
    print("="*60)
    print(f"\n✅ Total trials: {len(all_results)}")
    print(f"✅ Best validation ROC-AUC: {max([r['val_roc_auc'] for r in all_results]):.4f}")
    print(f"✅ Final test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"✅ Local model saved to: {model_path}")
    if registered_model is not None:
        print(f"✅ Registered model: {MODEL_REGISTRY_NAME} (version {registered_model.version})")
    print(f"\n🔗 View all runs: mlflow ui")
    print(f"🌐 Then open: http://localhost:5000")
    
    return best_model, best_params, results_df


if __name__ == "__main__":
    best_model, best_params, results_df = main()
    