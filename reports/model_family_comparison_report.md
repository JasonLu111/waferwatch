# WaferWatch Model Family Comparison Report

## 1. Purpose

This report compares Logistic Regression, Random Forest, and Gradient Boosting on the same selected SPC-enhanced feature table.
The goal is to evaluate whether nonlinear tree-based models improve over the linear baseline in the current controlled synthetic demo.

## 2. Compared Models

| Model | Feature table | Model file |
|---|---|---|
| Logistic Regression | `demo_spc_selected_feature_table.csv` | `spc_selected_logistic_regression.joblib` |
| Random Forest | `demo_spc_selected_feature_table.csv` | `random_forest_selected_baseline.joblib` |
| Gradient Boosting | `demo_spc_selected_feature_table.csv` | `gradient_boosting_selected_baseline.joblib` |

## 3. Main Metrics

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|---|---:|---:|---:|
| accuracy | 1.000000 | 1.000000 | 1.000000 |
| precision | 1.000000 | 1.000000 | 1.000000 |
| recall | 1.000000 | 1.000000 | 1.000000 |
| f1 | 1.000000 | 1.000000 | 1.000000 |
| balanced_accuracy | 1.000000 | 1.000000 | 1.000000 |
| matthews_corrcoef | 1.000000 | 1.000000 | 1.000000 |
| roc_auc | 1.000000 | 1.000000 | 1.000000 |
| pr_auc | 1.000000 | 1.000000 | 1.000000 |
| precision_at_k | 0.500000 | 0.500000 | 0.500000 |
| recall_at_k | 1.000000 | 1.000000 | 1.000000 |
| false_alarms_per_100_lots | 0.000000 | 0.000000 | 0.000000 |

## 4. Metric Differences

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

### 4.3 Gradient Boosting minus Random Forest

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

## 7. Interpretation

In the current controlled synthetic SPC demo, Logistic Regression, Random Forest, and Gradient Boosting all achieve perfect headline test metrics on the selected SPC-enhanced feature table. This indicates that the selected SPC features strongly encode the injected anomaly mechanism. The result validates the model comparison workflow, but it should not be interpreted as production fab performance. The tree-based models add value by providing feature importance: Random Forest uses both SPC violation count and maximum absolute SPC z-score, while Gradient Boosting places most importance on maximum absolute SPC z-score and SPC violation count.
