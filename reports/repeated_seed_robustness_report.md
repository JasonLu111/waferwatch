# WaferWatch Repeated-Seed Robustness Experiment Report

## 1. Purpose

This report repeats the main six-model baseline comparison under multiple random seeds.
The goal is to test whether the conclusions are stable or dependent on one train-test split.

## 2. Experiment Design

- Seeds: `[7, 21, 42, 84, 168]`
- Split: stratified 70 percent train / 30 percent test
- Dataset: selected SPC-enhanced feature table
- Models per seed: 6
- Total result rows: `30`

## 3. Aggregate Results Across Seeds

| Model | Precision mean | Precision std | Recall mean | Recall std | F1 mean | F1 std | PR-AUC mean | PR-AUC std | False alarms mean | False alarms std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Random Forest | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Gradient Boosting | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Isolation Forest | 0.393322 | 0.060736 | 1.000000 | 0.000000 | 0.562314 | 0.065064 | 0.896476 | 0.146994 | 33.333333 | 9.771699 |
| PCA Anomaly | 0.660354 | 0.169140 | 1.000000 | 0.000000 | 0.785340 | 0.124143 | 1.000000 | 0.000000 | 12.500000 | 8.838835 |
| Autoencoder Anomaly | 0.763492 | 0.164995 | 1.000000 | 0.000000 | 0.858009 | 0.105647 | 1.000000 | 0.000000 | 7.500000 | 6.180165 |

## 4. Worst-Case Checks

| Model | Recall min | Precision min | F1 min | PR-AUC min | False alarms max |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | 1.000000 | 0.294118 | 0.454545 | 0.686190 | 50.000000 |
| PCA Anomaly | 1.000000 | 0.454545 | 0.625000 | 1.000000 | 25.000000 |
| Autoencoder Anomaly | 1.000000 | 0.555556 | 0.714286 | 1.000000 | 16.666667 |

## 5. Per-Seed Results

| Seed | Model | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots | TP | FP | FN | TN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | Autoencoder Anomaly | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 | 5 | 2 | 0 | 17 |
| 7 | Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 7 | Isolation Forest | 0.384615 | 1.000000 | 0.555556 | 0.796190 | 33.333333 | 5 | 8 | 0 | 11 |
| 7 | Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 7 | PCA Anomaly | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 | 5 | 1 | 0 | 18 |
| 7 | Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 21 | Autoencoder Anomaly | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 | 5 | 1 | 0 | 18 |
| 21 | Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 21 | Isolation Forest | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 | 5 | 7 | 0 | 12 |
| 21 | Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 21 | PCA Anomaly | 0.555556 | 1.000000 | 0.714286 | 1.000000 | 16.666667 | 5 | 4 | 0 | 15 |
| 21 | Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 42 | Autoencoder Anomaly | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 42 | Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 42 | Isolation Forest | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 | 5 | 7 | 0 | 12 |
| 42 | Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 42 | PCA Anomaly | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 | 5 | 1 | 0 | 18 |
| 42 | Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 84 | Autoencoder Anomaly | 0.555556 | 1.000000 | 0.714286 | 1.000000 | 16.666667 | 5 | 4 | 0 | 15 |
| 84 | Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 84 | Isolation Forest | 0.294118 | 1.000000 | 0.454545 | 0.686190 | 50.000000 | 5 | 12 | 0 | 7 |
| 84 | Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 84 | PCA Anomaly | 0.454545 | 1.000000 | 0.625000 | 1.000000 | 25.000000 | 5 | 6 | 0 | 13 |
| 84 | Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 168 | Autoencoder Anomaly | 0.714286 | 1.000000 | 0.833333 | 1.000000 | 8.333333 | 5 | 2 | 0 | 17 |
| 168 | Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 168 | Isolation Forest | 0.454545 | 1.000000 | 0.625000 | 1.000000 | 25.000000 | 5 | 6 | 0 | 13 |
| 168 | Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| 168 | PCA Anomaly | 0.625000 | 1.000000 | 0.769231 | 1.000000 | 12.500000 | 5 | 3 | 0 | 16 |
| 168 | Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |

## 6. Interpretation

Repeated-seed evaluation makes the experiment more defensible because it reduces dependence on one lucky train-test split.

If a model has high mean performance and low standard deviation, its result is more stable. If the standard deviation is large, the model may be sensitive to how lots are split into train and test sets.

These repeated-seed results should still be interpreted as controlled synthetic evidence, not production performance.
