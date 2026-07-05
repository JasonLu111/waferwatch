# WaferWatch Model Family and Anomaly Baseline Comparison Report

## 1. Purpose

This report compares Logistic Regression, Random Forest, Gradient Boosting, and Isolation Forest on the same selected SPC-enhanced feature table.
Logistic Regression, Random Forest, and Gradient Boosting are supervised classifiers. Isolation Forest is an unsupervised anomaly detection baseline fitted only on normal-reference training lots.

## 2. Compared Models

| Model | Learning type | Training label usage | Feature table | Model file |
|---|---|---|---|---|
| Logistic Regression | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `spc_selected_logistic_regression.joblib` |
| Random Forest | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `random_forest_selected_baseline.joblib` |
| Gradient Boosting | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `gradient_boosting_selected_baseline.joblib` |
| Isolation Forest | Unsupervised anomaly detection | Labels used only for evaluation | `demo_spc_selected_feature_table.csv` | `isolation_forest_normal_reference.joblib` |

## 3. Main Metrics

| Metric | Logistic Regression | Random Forest | Gradient Boosting | Isolation Forest |
|---|---:|---:|---:|---:|
| accuracy | 1.000000 | 1.000000 | 1.000000 | 0.708333 |
| precision | 1.000000 | 1.000000 | 1.000000 | 0.416667 |
| recall | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| f1 | 1.000000 | 1.000000 | 1.000000 | 0.588235 |
| balanced_accuracy | 1.000000 | 1.000000 | 1.000000 | 0.815789 |
| matthews_corrcoef | 1.000000 | 1.000000 | 1.000000 | 0.512989 |
| roc_auc | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| pr_auc | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| precision_at_k | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| recall_at_k | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| false_alarms_per_100_lots | 0.000000 | 0.000000 | 0.000000 | 29.166667 |

## 4. Metric Differences

Differences are calculated against the Logistic Regression supervised baseline.

### 4.1 Random Forest minus Logistic Regression

| Metric | Difference |
|---|---:|
| accuracy | 0.000000 |
| precision | 0.000000 |
| recall | 0.000000 |
| f1 | 0.000000 |
| balanced_accuracy | 0.000000 |
| matthews_corrcoef | 0.000000 |
| roc_auc | 0.000000 |
| pr_auc | 0.000000 |
| precision_at_k | 0.000000 |
| recall_at_k | 0.000000 |
| false_alarms_per_100_lots | 0.000000 |

### 4.2 Gradient Boosting minus Logistic Regression

| Metric | Difference |
|---|---:|
| accuracy | 0.000000 |
| precision | 0.000000 |
| recall | 0.000000 |
| f1 | 0.000000 |
| balanced_accuracy | 0.000000 |
| matthews_corrcoef | 0.000000 |
| roc_auc | 0.000000 |
| pr_auc | 0.000000 |
| precision_at_k | 0.000000 |
| recall_at_k | 0.000000 |
| false_alarms_per_100_lots | 0.000000 |

### 4.3 Isolation Forest minus Logistic Regression

| Metric | Difference |
|---|---:|
| accuracy | -0.291667 |
| precision | -0.583333 |
| recall | 0.000000 |
| f1 | -0.411765 |
| balanced_accuracy | -0.184211 |
| matthews_corrcoef | -0.487011 |
| roc_auc | 0.000000 |
| pr_auc | 0.000000 |
| precision_at_k | 0.000000 |
| recall_at_k | 0.000000 |
| false_alarms_per_100_lots | 29.166667 |

## 5. Random Forest Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `spc_violation_count` | 0.374099 |
| 2 | `spc_max_abs_zscore` | 0.369001 |
| 3 | `sensor_std` | 0.119823 |
| 4 | `sensor_mean` | 0.103394 |
| 5 | `sensor_max` | 0.019614 |
| 6 | `sensor_min` | 0.014068 |

## 6. Gradient Boosting Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `spc_max_abs_zscore` | 0.592688 |
| 2 | `spc_violation_count` | 0.407312 |
| 3 | `sensor_std` | 0.000000 |
| 4 | `sensor_mean` | 0.000000 |
| 5 | `sensor_max` | 0.000000 |
| 6 | `sensor_min` | 0.000000 |

## 7. Isolation Forest Top Suspicious Lots

| Rank | Lot ID | True Label | Risk Score | Predicted Label |
|---:|---|---:|---:|---:|
| 1 | `LOT_SPC_024` | 1 | 0.195428 | 1 |
| 2 | `LOT_SPC_077` | 1 | 0.168547 | 1 |
| 3 | `LOT_SPC_078` | 1 | 0.124095 | 1 |
| 4 | `LOT_SPC_027` | 1 | 0.123812 | 1 |
| 5 | `LOT_SPC_021` | 1 | 0.099447 | 1 |
| 6 | `LOT_SPC_034` | 0 | 0.036980 | 1 |
| 7 | `LOT_SPC_025` | 0 | 0.032813 | 1 |
| 8 | `LOT_SPC_065` | 0 | 0.025443 | 1 |
| 9 | `LOT_SPC_047` | 0 | 0.009712 | 1 |
| 10 | `LOT_SPC_003` | 0 | 0.006538 | 1 |

## 8. Interpretation

In the current controlled synthetic SPC demo, the three supervised classifiers achieve perfect headline test metrics on the selected SPC-enhanced feature table. This suggests that the selected SPC features strongly encode the injected anomaly mechanism. Isolation Forest also ranks all held-out failed lots at the top, producing perfect ROC-AUC and PR-AUC, but its default anomaly threshold creates more false alarms than the supervised models. This is a useful manufacturing-style trade-off: an unsupervised anomaly detector can be valuable when labels are rare or delayed, but it requires threshold tuning, top-K review, and false-alarm budget control before it can be used operationally. These results validate the demo workflow and should not be interpreted as production fab performance.
