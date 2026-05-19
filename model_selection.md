# Model Selection: Why XGBoost

This document explains why XGBoost was chosen over LightGBM for the final fraud detection model.

## Evaluation criteria

The comparison was based on held-out test metrics after hyperparameter tuning with Optuna.
The primary objective was to maximize precision-recall performance while also maintaining high ROC-AUC and low false-positive rate.

### Metrics used

- PR-AUC (primary metric)
- ROC-AUC
- Fraud recall
- Fraud precision
- False positive rate

## Results summary

| Model | Test PR-AUC | Test ROC-AUC | Fraud Recall | Fraud Precision | False Positive Rate |
|---|---|---|---|---|---|
| LightGBM | 0.6818 | 0.9755 | 84.69% | 100.00% | 0.1002% |
| XGBoost | 0.8694 | 0.9832 | 84.69% | 100.00% | 0.0492% |

## Why XGBoost was selected

1. **Stronger PR-AUC**
   - XGBoost achieved `0.8694`, compared to LightGBM's `0.6818`.
   - This is a large relative improvement in the precision-recall trade-off for fraud detection.

2. **Higher ROC-AUC**
   - XGBoost delivered `0.9832`, versus LightGBM's `0.9755`.
   - This shows better overall ranking of fraud risk.

3. **Same precision and recall**
   - Both models reached `100%` fraud precision and `84.69%` fraud recall on the test set.
   - Since the fraud detection recall and precision were equal, the higher PR-AUC and ROC-AUC are the decisive differentiators.

4. **Lower false-positive rate**
   - XGBoost had `0.0492%` false-positive rate, which is roughly half of LightGBM's `0.1002%`.
   - A lower false-positive rate is important for preserving legitimate transactions and reducing unnecessary reviews.

## Implementation details

- LightGBM tuning script: `fraud_detection_api/notebooks/train_lightGBM.py`
- XGBoost Optuna tuning script: `fraud_detection_api/notebooks/train_xgboost_optuna.py`
- Final LightGBM artifact: `fraud_detection_api/models/fraud_detector_v1.pkl`
- Final XGBoost artifact: `fraud_detection_api/models/fraud_detector_xgb_v1.pkl`

## Conclusion

The XGBoost model was chosen because it outperformed LightGBM on the key quantitative metrics used for selection, especially PR-AUC and ROC-AUC, while matching the same fraud precision and recall. This decision is driven by numbers, not gut feel.
