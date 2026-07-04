# WaferWatch Gradient Boosting Baseline Report

## 1. Purpose

This report documents the Gradient Boosting baseline model trained on the selected SPC-enhanced feature table.
The purpose is to add a boosting-style tabular baseline before introducing external libraries such as LightGBM or XGBoost.

## 2. Model Configuration

- Model file: `D:\waferwatch\models\gradient_boosting_selected_baseline.joblib`
- Feature table: `D:\waferwatch\data\processed\demo_spc_selected_feature_table.csv`
- Training rows: `56`
- Test rows: `24`
- Number of features: `6`
- Number of estimators: `200`
- Learning rate: `0.05`
- Max depth: `3`
- Sample weighting: `balanced`

## 3. Evaluation Metrics

| Metric | Value |
|---|---:|
| accuracy | 1.000000 |
| precision | 1.000000 |
| recall | 1.000000 |
| f1 | 1.000000 |
| balanced_accuracy | 1.000000 |
| matthews_corrcoef | 1.000000 |
| roc_auc | 1.000000 |
| pr_auc | 1.000000 |
| precision_at_k | 0.500000 |
| recall_at_k | 1.000000 |
| false_alarms_per_100_lots | 0.000000 |

## 4. Confusion Matrix

| Item | Count |
|---|---:|
| true_negative | 19 |
| false_positive | 0 |
| false_negative | 0 |
| true_positive | 5 |

## 5. Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `spc_max_abs_zscore` | 0.592688 |
| 2 | `spc_violation_count` | 0.407312 |
| 3 | `sensor_std` | 0.000000 |
| 4 | `sensor_mean` | 0.000000 |
| 5 | `sensor_max` | 0.000000 |
| 6 | `sensor_min` | 0.000000 |

## 6. Interpretation

This Gradient Boosting model is a boosting-style nonlinear tabular baseline trained on controlled synthetic SPC-enhanced demo data. Results should be used to compare model behavior in the demo pipeline, not to claim production fab performance.
