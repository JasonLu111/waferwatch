# WaferWatch PCA Anomaly Detection Baseline Report

## 1. Purpose

This report documents a PCA reconstruction-error anomaly detection baseline.
The PCA model is fitted only on normal-reference training lots. Labels are used only for held-out evaluation.

## 2. Model Configuration

- Model file: `D:\waferwatch\models\pca_anomaly_normal_reference.joblib`
- Feature table: `D:\waferwatch\data\processed\demo_spc_selected_feature_table.csv`
- Training rows: `56`
- Normal-reference training rows: `45`
- Test rows: `24`
- Number of original features: `6`
- PCA components retained: `4`
- Explained variance ratio sum: `0.977926`
- Threshold quantile: `0.95`
- Reconstruction-error threshold: `0.017723`

## 3. Evaluation Metrics

| Metric | Value |
|---|---:|
| accuracy | 0.958333 |
| precision | 0.833333 |
| recall | 1.000000 |
| f1 | 0.909091 |
| balanced_accuracy | 0.973684 |
| matthews_corrcoef | 0.888523 |
| roc_auc | 1.000000 |
| pr_auc | 1.000000 |
| precision_at_k | 0.500000 |
| recall_at_k | 1.000000 |
| false_alarms_per_100_lots | 4.166667 |

## 4. Confusion Matrix

| Item | Count |
|---|---:|
| true_negative | 18 |
| false_positive | 1 |
| false_negative | 0 |
| true_positive | 5 |

## 5. Top Suspicious Lots

| Rank | Lot ID | True Label | Reconstruction Error | Predicted Label |
|---:|---|---:|---:|---:|
| 1 | `LOT_SPC_021` | 1 | 0.843782 | 1 |
| 2 | `LOT_SPC_027` | 1 | 0.823822 | 1 |
| 3 | `LOT_SPC_078` | 1 | 0.820936 | 1 |
| 4 | `LOT_SPC_077` | 1 | 0.818217 | 1 |
| 5 | `LOT_SPC_024` | 1 | 0.816952 | 1 |
| 6 | `LOT_SPC_063` | 0 | 0.022001 | 1 |
| 7 | `LOT_SPC_034` | 0 | 0.015658 | 0 |
| 8 | `LOT_SPC_022` | 0 | 0.015199 | 0 |
| 9 | `LOT_SPC_015` | 0 | 0.013424 | 0 |
| 10 | `LOT_SPC_076` | 0 | 0.006717 | 0 |

## 6. Interpretation

PCA anomaly detection is used here as a reconstruction-error baseline. The model is fitted only on normal-reference training lots, then lots with large reconstruction error are treated as suspicious. This provides a second unsupervised anomaly detection family beyond Isolation Forest. Because this is still a controlled synthetic SPC demo, the result should validate the workflow rather than imply real production performance.
