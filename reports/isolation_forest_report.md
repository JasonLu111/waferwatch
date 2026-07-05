# WaferWatch Isolation Forest Baseline Report

## 1. Purpose

This report documents an unsupervised Isolation Forest anomaly detection baseline.
The model is fitted only on normal training lots. Labels are used only for held-out evaluation.

## 2. Model Configuration

- Model file: `D:\waferwatch\models\isolation_forest_normal_reference.joblib`
- Feature table: `D:\waferwatch\data\processed\demo_spc_selected_feature_table.csv`
- Training rows: `56`
- Normal-reference training rows: `45`
- Test rows: `24`
- Number of features: `6`
- Number of estimators: `300`
- Contamination setting: `auto`
- Random state: `42`

## 3. Evaluation Metrics

| Metric | Value |
|---|---:|
| accuracy | 0.708333 |
| precision | 0.416667 |
| recall | 1.000000 |
| f1 | 0.588235 |
| balanced_accuracy | 0.815789 |
| matthews_corrcoef | 0.512989 |
| roc_auc | 1.000000 |
| pr_auc | 1.000000 |
| precision_at_k | 0.500000 |
| recall_at_k | 1.000000 |
| false_alarms_per_100_lots | 29.166667 |

## 4. Confusion Matrix

| Item | Count |
|---|---:|
| true_negative | 12 |
| false_positive | 7 |
| false_negative | 0 |
| true_positive | 5 |

## 5. Top Suspicious Lots

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

## 6. Interpretation

Isolation Forest is used here as an unsupervised anomaly detection baseline. The model is fitted only on normal-reference training lots, and held-out labels are used only for evaluation. This baseline is useful for discussing situations where confirmed failure labels are rare, delayed, or incomplete. Because this is still a controlled synthetic SPC demo, the result should validate the workflow rather than imply real production performance.
