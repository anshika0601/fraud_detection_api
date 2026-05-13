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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import itertools
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


# Set MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_lightgbm_tuning")

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


def tune_lightgbm():
    """Hyperparameter tuning for LightGBM"""
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Calculate class imbalance ratio
    ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"\n⚖️ Class imbalance ratio: {ratio:.1f}:1")
    
    # Define hyperparameter grid
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    num_leaves_options = [15, 31, 63, 127, 255]
    
    # Fixed parameters (same for all runs)
    fixed_params = {
        'n_estimators': 300,
        'max_depth': -1,  # Unlimited, let num_leaves control complexity
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'min_child_samples': 20,
        'scale_pos_weight': ratio,
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True  # Faster training
    }
    
    # Store all results
    all_results = []
    best_pr_auc = 0
    best_params = None
    best_model = None
    best_run_id = None
    
    # Total combinations
    total_combinations = len(learning_rates) * len(num_leaves_options)
    current_run = 0
    
    print(f"\n🚀 Starting Hyperparameter Tuning")
    print(f"   Learning rates: {learning_rates}")
    print(f"   Num leaves: {num_leaves_options}")
    print(f"   Total combinations: {total_combinations}")
    print("="*60)
    
    # Grid search
    for lr in learning_rates:
        for leaves in num_leaves_options:
            current_run += 1
            
            # Create run name
            run_name = f"LightGBM_lr{lr}_leaves{leaves}"
            
            print(f"\n[{current_run}/{total_combinations}] Training: {run_name}")
            
            # Start MLflow run
            with mlflow.start_run(run_name=run_name) as run:
                
                # Combine parameters
                params = {
                    'learning_rate': lr,
                    'num_leaves': leaves,
                    **fixed_params
                }
                
                # Log parameters to MLflow
                mlflow.log_params(params)
                
                # Train model
                print(f"   Training with lr={lr}, leaves={leaves}...")
                model = lgb.LGBMClassifier(**params)
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='average_precision',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(0)  # Disable verbose output
                    ]
                )
                
                # Calculate metrics
                train_metrics = calculate_metrics(model, X_train, y_train, "Train")
                val_metrics = calculate_metrics(model, X_val, y_val, "Validation")
                test_metrics = calculate_metrics(model, X_test, y_test, "Test")
                
                # Log metrics to MLflow
                for metric_name, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"val_{metric_name}", value)
                
                for metric_name, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"test_{metric_name}", value)
                
                # Log best iteration
                if hasattr(model, 'best_iteration_'):
                    mlflow.log_metric("best_iteration", model.best_iteration_)
                
                # Generate and log feature importance
                importance_df = plot_feature_importance(
                    model, X_train.columns, lr, leaves,
                    f'data/lgbm_importance_lr{lr}_leaves{leaves}.png'
                )
                mlflow.log_artifact(f'data/lgbm_importance_lr{lr}_leaves{leaves}.png')
                
                # Save importance CSV
                importance_df.to_csv(f'data/lgbm_importance_lr{lr}_leaves{leaves}.csv', index=False)
                mlflow.log_artifact(f'data/lgbm_importance_lr{lr}_leaves{leaves}.csv')
                
                # Generate and log performance curves
                plot_roc_pr_curves(
                    model, X_val, y_val, lr, leaves,
                    f'data/lgbm_curves_lr{lr}_leaves{leaves}.png'
                )
                mlflow.log_artifact(f'data/lgbm_curves_lr{lr}_leaves{leaves}.png')
                
                # Log the model
                safe_model_name = sanitize_mlflow_model_name(
                    f"lightgbm_model_lr{lr}_leaves{leaves}"
                )
                mlflow.lightgbm.log_model(model, name=safe_model_name)
                
                # Save model locally
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, f'models/lightgbm_lr{lr}_leaves{leaves}.pkl')
                
                # Store results
                result = {
                    'learning_rate': lr,
                    'num_leaves': leaves,
                    'val_pr_auc': val_metrics['pr_auc'],
                    'val_roc_auc': val_metrics['roc_auc'],
                    'val_recall': val_metrics['fraud_recall'],
                    'val_precision': val_metrics['fraud_precision'],
                    'val_fpr': val_metrics['false_positive_rate'],
                    'test_pr_auc': test_metrics['pr_auc'],
                    'test_recall': test_metrics['fraud_recall'],
                    'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else 300,
                    'run_id': run.info.run_id
                }
                all_results.append(result)
                
                print(f"   ✅ PR-AUC: {val_metrics['pr_auc']:.4f} | Recall: {val_metrics['fraud_recall']:.2%}")
                
                # Track best model
                if val_metrics['pr_auc'] > best_pr_auc:
                    best_pr_auc = val_metrics['pr_auc']
                    best_params = {'learning_rate': lr, 'num_leaves': leaves}
                    best_model = model
                    best_run_id = run.info.run_id
                    print(f"   🏆 New best model!")
    
    return all_results, best_params, best_model, best_run_id, (X_train, X_val, X_test, y_train, y_val, y_test)


def analyze_results(all_results):
    """Analyze and visualize tuning results"""
    print("\n" + "="*60)
    print("TUNING RESULTS ANALYSIS")
    print("="*60)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('val_pr_auc', ascending=False)
    
    print("\n📊 Top 5 Configurations:")
    print(results_df[['learning_rate', 'num_leaves', 'val_pr_auc', 'val_recall', 'val_fpr']].head(5).to_string(index=False))
    
    print("\n📊 Bottom 5 Configurations:")
    print(results_df[['learning_rate', 'num_leaves', 'val_pr_auc', 'val_recall', 'val_fpr']].tail(5).to_string(index=False))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. PR-AUC heatmap
    pivot_pr = results_df.pivot(index='learning_rate', columns='num_leaves', values='val_pr_auc')
    sns.heatmap(pivot_pr, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('PR-AUC Heatmap', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Num Leaves', fontsize=12)
    axes[0,0].set_ylabel('Learning Rate', fontsize=12)
    
    # 2. PR-AUC by learning rate (for different num_leaves)
    for leaves in results_df['num_leaves'].unique():
        subset = results_df[results_df['num_leaves'] == leaves]
        axes[0,1].plot(subset['learning_rate'], subset['val_pr_auc'], 
                       marker='o', label=f'leaves={leaves}', linewidth=2)
    axes[0,1].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('PR-AUC', fontsize=12, fontweight='bold')
    axes[0,1].set_title('PR-AUC vs Learning Rate', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. PR-AUC vs Num Leaves
    for lr in results_df['learning_rate'].unique():
        subset = results_df[results_df['learning_rate'] == lr]
        axes[1,0].plot(subset['num_leaves'], subset['val_pr_auc'], 
                       marker='s', label=f'lr={lr}', linewidth=2)
    axes[1,0].set_xlabel('Num Leaves', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('PR-AUC', fontsize=12, fontweight='bold')
    axes[1,0].set_title('PR-AUC vs Num Leaves', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xscale('log')
    
    # 4. Recall vs Precision scatter
    scatter = axes[1,1].scatter(results_df['val_recall'], results_df['val_precision'], 
                                c=results_df['val_pr_auc'], cmap='viridis', 
                                s=100, alpha=0.6)
    axes[1,1].set_xlabel('Recall (Fraud Detection Rate)', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1,1].set_title('Recall-Precision Trade-off', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1,1], label='PR-AUC')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('LightGBM Hyperparameter Tuning Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data/lightgbm_tuning_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df.to_csv('models/lightgbm_tuning_results.csv', index=False)
    print("\n✅ Results saved to: models/lightgbm_tuning_results.csv")
    print("✅ Visualization saved to: data/lightgbm_tuning_results.png")
    
    return results_df


def save_best_model(best_model, best_params, best_run_id, X_test, y_test):
    """Save the best model and its metrics"""
    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)
    
    print(f"\n🏆 Best Configuration:")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   Num Leaves: {best_params['num_leaves']}")
    print(f"   MLflow Run ID: {best_run_id}")
    
    # Calculate final metrics on test set
    test_metrics = calculate_metrics(best_model, X_test, y_test, "Test")
    
    print(f"\n📊 Final Test Performance:")
    print(f"   PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"   ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Fraud Recall: {test_metrics['fraud_recall']:.2%}")
    print(f"   Fraud Precision: {test_metrics['fraud_precision']:.2%}")
    print(f"   False Positive Rate: {test_metrics['false_positive_rate']:.4%}")
    
    # Generate final plots for best model
    print("\n🎨 Generating final artifacts for best model...")
    
    # Feature importance
    importance_df = plot_feature_importance(
        best_model, X_test.columns, 
        best_params['learning_rate'], best_params['num_leaves'],
        'data/lightgbm_best_feature_importance.png'
    )
    
    # ROC/PR curves
    plot_roc_pr_curves(
        best_model, X_test, y_test,
        best_params['learning_rate'], best_params['num_leaves'],
        'data/lightgbm_best_performance_curves.png'
    )
    
    # Confusion matrix
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
    
    # Save best model
    joblib.dump(best_model, 'models/lightgbm_best_model.pkl')
    print("✅ Best model saved to: models/lightgbm_best_model.pkl")
    
    # Save metrics
    best_metrics = {
        'best_params': best_params,
        'best_run_id': best_run_id,
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in test_metrics.items()},
        'top_features': importance_df.head(10)[['feature', 'importance']].to_dict('records')
    }
    
    with open('models/lightgbm_best_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print("✅ Best metrics saved to: models/lightgbm_best_metrics.json")
    
    # Print top features
    print("\n📈 Top 10 Most Important Features (Best Model):")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return test_metrics


def main():
    """Main execution"""
    
    # Run hyperparameter tuning
    all_results, best_params, best_model, best_run_id, data = tune_lightgbm()
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Analyze results
    results_df = analyze_results(all_results)
    
    # Save best model
    test_metrics = save_best_model(best_model, best_params, best_run_id, X_test, y_test)
    
    # Final summary
    print("\n" + "="*60)
    print("TUNING COMPLETE! SUMMARY")
    print("="*60)
    print(f"\n✅ Total runs: {len(all_results)}")
    print(f"✅ Best PR-AUC: {best_model.predict_proba(X_val)[:, 1]}")
    print(f"✅ Best model saved with {best_params['learning_rate']} lr and {best_params['num_leaves']} leaves")
    print(f"\n🔗 View all runs: mlflow ui")
    print(f"🌐 Then open: http://localhost:5000")
    
    return best_model, best_params, results_df


if __name__ == "__main__":
    best_model, best_params, results_df = main()
    