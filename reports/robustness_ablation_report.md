# WaferWatch Robustness and Ablation Experiment Report

## 1. Purpose

This report tests whether strong demo results remain stable when key SPC features are removed or when the anomaly signal is weakened.
The purpose is to strengthen the experimental design and avoid overclaiming perfect metrics from a controlled synthetic SPC demo.

## 2. Experiment Design

| Experiment type | Scenario | Purpose |
|---|---|---|
| Baseline | Full selected feature table | Establish the reference result |
| Feature ablation | Remove one or more SPC features | Test whether models depend too heavily on engineered SPC signals |
| Anomaly severity stress test | Shrink failed-lot features toward normal-lot means | Test whether models remain stable when anomaly signals become weaker |

## 3. Baseline Full-Feature Results

| Scenario | Model | Features | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_full_features | Logistic Regression | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| baseline_full_features | Random Forest | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| baseline_full_features | Gradient Boosting | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| baseline_full_features | Isolation Forest | 6 | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 |
| baseline_full_features | PCA Anomaly | 6 | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |
| baseline_full_features | Autoencoder Anomaly | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## 4. Feature Ablation Results

| Scenario | Model | Features | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots |
|---|---|---:|---:|---:|---:|---:|---:|
| ablation_without_spc_violation_count | Logistic Regression | 5 | 1.000000 | 0.800000 | 0.888889 | 1.000000 | 0.000000 |
| ablation_without_spc_violation_count | Random Forest | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_violation_count | Gradient Boosting | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_violation_count | Isolation Forest | 5 | 0.357143 | 1.000000 | 0.526316 | 1.000000 | 37.500000 |
| ablation_without_spc_violation_count | PCA Anomaly | 5 | 0.500000 | 0.200000 | 0.285714 | 0.438095 | 4.166667 |
| ablation_without_spc_violation_count | Autoencoder Anomaly | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_max_abs_zscore | Logistic Regression | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_max_abs_zscore | Random Forest | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_max_abs_zscore | Gradient Boosting | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_spc_max_abs_zscore | Isolation Forest | 5 | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 |
| ablation_without_spc_max_abs_zscore | PCA Anomaly | 5 | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |
| ablation_without_spc_max_abs_zscore | Autoencoder Anomaly | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_without_all_spc_features | Logistic Regression | 4 | 0.750000 | 0.600000 | 0.666667 | 0.629762 | 4.166667 |
| ablation_without_all_spc_features | Random Forest | 4 | 0.666667 | 0.800000 | 0.727273 | 0.850000 | 8.333333 |
| ablation_without_all_spc_features | Gradient Boosting | 4 | 0.666667 | 0.800000 | 0.727273 | 0.575000 | 8.333333 |
| ablation_without_all_spc_features | Isolation Forest | 4 | 0.454545 | 1.000000 | 0.625000 | 1.000000 | 25.000000 |
| ablation_without_all_spc_features | PCA Anomaly | 4 | 0.000000 | 0.000000 | 0.000000 | 0.297835 | 4.166667 |
| ablation_without_all_spc_features | Autoencoder Anomaly | 4 | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 |
| ablation_spc_features_only | Logistic Regression | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_spc_features_only | Random Forest | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_spc_features_only | Gradient Boosting | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| ablation_spc_features_only | Isolation Forest | 2 | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 |
| ablation_spc_features_only | PCA Anomaly | 2 | 0.555556 | 1.000000 | 0.714286 | 1.000000 | 16.666667 |
| ablation_spc_features_only | Autoencoder Anomaly | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## 5. Anomaly Severity Stress Test Results

| Scenario | Model | Features | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots |
|---|---|---:|---:|---:|---:|---:|---:|
| anomaly_severity_0.75 | Logistic Regression | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.75 | Random Forest | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.75 | Gradient Boosting | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.75 | Isolation Forest | 6 | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 |
| anomaly_severity_0.75 | PCA Anomaly | 6 | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |
| anomaly_severity_0.75 | Autoencoder Anomaly | 6 | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 |
| anomaly_severity_0.50 | Logistic Regression | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.50 | Random Forest | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.50 | Gradient Boosting | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.50 | Isolation Forest | 6 | 0.416667 | 1.000000 | 0.588235 | 0.966667 | 29.166667 |
| anomaly_severity_0.50 | PCA Anomaly | 6 | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |
| anomaly_severity_0.50 | Autoencoder Anomaly | 6 | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 |
| anomaly_severity_0.25 | Logistic Regression | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.25 | Random Forest | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.25 | Gradient Boosting | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| anomaly_severity_0.25 | Isolation Forest | 6 | 0.300000 | 0.600000 | 0.400000 | 0.587157 | 29.166667 |
| anomaly_severity_0.25 | PCA Anomaly | 6 | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 |
| anomaly_severity_0.25 | Autoencoder Anomaly | 6 | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |

## 6. Interpretation

These experiments make the demo more defensible. If performance remains high after removing SPC features or weakening the anomaly signal, the workflow is more robust. If performance drops sharply, the result is still useful because it identifies which engineered signals drive the model.
The current results should be interpreted as controlled synthetic evidence, not production performance. The main value is the experimental design: supervised baselines, unsupervised anomaly detectors, feature ablation, severity stress tests, and false-alarm analysis are evaluated under one consistent framework.
