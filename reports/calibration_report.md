# WaferWatch Calibration Report

## 1. Purpose

This report evaluates whether WaferWatch risk scores are reasonably calibrated.
Calibration matters because the project uses predicted risk scores for thresholding, escalation, and monitoring.

## 2. Model and Data

- Model file: `D:\waferwatch\models\spc_selected_logistic_regression.joblib`
- Feature table: `D:\waferwatch\data\processed\demo_spc_selected_feature_table.csv`
- Test rows: `24`
- Number of bins: `10`

## 3. Calibration Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| Brier score | 0.004395 | Lower is better; measures mean squared probability error |
| Log loss | 0.043329 | Lower is better; penalizes confident wrong predictions |
| Expected calibration error | 0.040760 | Lower is better; weighted average calibration gap |
| Maximum calibration error | 0.265852 | Largest bin-level calibration gap |
| ROC-AUC | 1.000000 | Ranking quality across thresholds |
| PR-AUC | 1.000000 | Ranking quality under class imbalance |

## 4. Reliability Table

| Bin | Range | N | Avg predicted probability | Observed positive rate | Abs calibration error |
|---:|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1] | 19 | 0.027875 | 0.000000 | 0.027875 |
| 1 | [0.1, 0.2] | 0 |  |  |  |
| 2 | [0.2, 0.3] | 0 |  |  |  |
| 3 | [0.3, 0.4] | 0 |  |  |  |
| 4 | [0.4, 0.5] | 0 |  |  |  |
| 5 | [0.5, 0.6] | 0 |  |  |  |
| 6 | [0.6, 0.7] | 0 |  |  |  |
| 7 | [0.7, 0.8] | 1 | 0.734148 | 1.000000 | 0.265852 |
| 8 | [0.8, 0.9] | 1 | 0.867122 | 1.000000 | 0.132878 |
| 9 | [0.9, 1.0] | 3 | 0.983370 | 1.000000 | 0.016630 |

## 5. Interpretation

This calibration report evaluates the probability quality of the current demo model. Because the dataset is synthetic and small, the result should be used to validate the calibration workflow rather than to claim production probability reliability.

## 6. Limitations

- The current calibration analysis uses synthetic demo data.
- The test set is small.
- Calibration results should not be interpreted as production probability quality.
- Future work should evaluate calibration on larger and time-split datasets.
