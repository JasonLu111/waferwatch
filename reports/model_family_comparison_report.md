# WaferWatch Model Family Comparison Report

## 1. Purpose

This report compares Logistic Regression and Random Forest on the same selected SPC-enhanced feature table.
The goal is to test whether a nonlinear tabular model improves over the current linear baseline.

## 2. Compared Models

| Model | Feature table | Model file |
|---|---|---|
| Logistic Regression | `demo_spc_selected_feature_table.csv` | `spc_selected_logistic_regression.joblib` |
| Random Forest | `demo_spc_selected_feature_table.csv` | `random_forest_selected_baseline.joblib` |

## 3. Metrics

| Metric | Logistic Regression | Random Forest | RF - LR |
|---|---:|---:|---:|
| accuracy | 1.000000 | 1.000000 | 0.000000 |
| precision | 1.000000 | 1.000000 | 0.000000 |
| recall | 1.000000 | 1.000000 | 0.000000 |
| f1 | 1.000000 | 1.000000 | 0.000000 |
| balanced_accuracy | 1.000000 | 1.000000 | 0.000000 |
| matthews_corrcoef | 1.000000 | 1.000000 | 0.000000 |
| roc_auc | 1.000000 | 1.000000 | 0.000000 |
| pr_auc | 1.000000 | 1.000000 | 0.000000 |
| precision_at_k | 0.500000 | 0.500000 | 0.000000 |
| recall_at_k | 1.000000 | 1.000000 | 0.000000 |
| false_alarms_per_100_lots | 0.000000 | 0.000000 | 0.000000 |

## 4. Random Forest Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `spc_violation_count` | 0.374099 |
| 2 | `spc_max_abs_zscore` | 0.369001 |
| 3 | `sensor_std` | 0.119823 |
| 4 | `sensor_mean` | 0.103394 |
| 5 | `sensor_max` | 0.019614 |
| 6 | `sensor_min` | 0.014068 |

## 5. Interpretation

In the current controlled synthetic demo, both Logistic Regression and Random Forest achieve perfect test metrics on the selected SPC-enhanced feature table. This suggests the selected SPC features strongly encode the injected anomaly mechanism. It does not prove production performance. Random Forest additionally provides feature importance, showing that SPC violation count and maximum absolute SPC z-score dominate the demo signal.
