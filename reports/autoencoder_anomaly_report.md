# WaferWatch Autoencoder Anomaly Detection Baseline Report

## 1. Purpose

This report documents a lightweight autoencoder-style reconstruction-error anomaly detection baseline.
The model is fitted only on normal-reference training lots. Labels are used only for held-out evaluation.

## 2. Model Configuration

- Model file: `D:\waferwatch\models\autoencoder_anomaly_normal_reference.joblib`
- Feature table: `D:\waferwatch\data\processed\demo_spc_selected_feature_table.csv`
- Training rows: `56`
- Normal-reference training rows: `45`
- Test rows: `24`
- Number of features: `6`
- Hidden layers: `[4, 2, 4]`
- Activation: `relu`
- Solver: `adam`
- Max iterations: `3000`
- Threshold quantile: `0.95`
- Reconstruction-error threshold: `0.422426`

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

## 5. Top Suspicious Lots

| Rank | Lot ID | True Label | Reconstruction Error | Predicted Label |
|---:|---|---:|---:|---:|
| 1 | `LOT_SPC_077` | 1 | 6.109482 | 1 |
| 2 | `LOT_SPC_024` | 1 | 6.021887 | 1 |
| 3 | `LOT_SPC_021` | 1 | 2.341400 | 1 |
| 4 | `LOT_SPC_078` | 1 | 2.105110 | 1 |
| 5 | `LOT_SPC_027` | 1 | 1.498571 | 1 |
| 6 | `LOT_SPC_065` | 0 | 0.377507 | 0 |
| 7 | `LOT_SPC_034` | 0 | 0.321889 | 0 |
| 8 | `LOT_SPC_025` | 0 | 0.314094 | 0 |
| 9 | `LOT_SPC_011` | 0 | 0.302727 | 0 |
| 10 | `LOT_SPC_080` | 0 | 0.272448 | 0 |

## 6. Interpretation

This lightweight autoencoder-style baseline uses a neural reconstruction model trained only on normal-reference lots. Lots with high reconstruction error are treated as suspicious. In this controlled synthetic SPC demo, this baseline is used to compare neural reconstruction-error anomaly detection against PCA and Isolation Forest. The result should validate the workflow rather than imply real production performance.
