# WaferWatch

**Cost-Sensitive Fab Anomaly Monitoring and Evidence-Grounded Root-Cause Triage**

WaferWatch is a manufacturing AI decision-support project for semiconductor-style anomaly monitoring.

The goal is not only to train a machine learning model. The goal is to build a production-style workflow that helps engineers:

- Score lot-level anomaly risk
- Prioritize high-risk lots for review
- Compare feature strategies
- Use SPC signals as manufacturing-aware features
- Tune alert thresholds with operational cost in mind
- Monitor data drift after deployment
- Monitor model performance after labels become available
- Generate engineer-readable alert summaries

This repository is currently a controlled synthetic demo. It does not use real production data and should not be interpreted as proof of real fab performance.

---

## 1. Project Motivation

Manufacturing anomaly detection is not a simple accuracy-maximization problem.

In a real engineering workflow, the key questions are:

- Which lots should be reviewed first?
- How many alerts can engineers realistically handle?
- What is the cost of missing a risky lot?
- What is the cost of creating too many false alarms?
- Has the data distribution shifted after deployment?
- Has the model performance degraded after current labels become available?
- Can the system produce evidence that engineers can inspect?

WaferWatch reframes the problem from simple pass/fail prediction into a cost-sensitive anomaly monitoring and escalation workflow.

---

## 2. Current System Status

The current implementation includes:

| Area | Status |
|---|---|
| Project structure | Completed |
| Configuration utilities | Completed |
| Logging utilities | Completed |
| Data validation | Completed |
| Data ingestion | Completed |
| Data cleaning | Completed |
| Sensor aggregate feature engineering | Completed |
| SPC feature engineering | Completed |
| Feature selection | Completed |
| Baseline model training | Completed |
| Model evaluation | Completed |
| Model comparison | Completed |
| Model family comparison | Completed |
| Random Forest baseline | Completed |
| Gradient Boosting baseline | Completed |
| Isolation Forest baseline | Completed |
| Isolation Forest threshold tuning | Completed |
| PCA anomaly detection baseline | Completed |
| Autoencoder anomaly detection baseline | Completed |
| Robustness and ablation experiments | Completed |
| Calibration analysis | Completed |
| Cost-sensitive thresholding | Completed |
| Batch prediction | Completed |
| Drift monitoring | Completed |
| Performance monitoring | Completed |
| Monitoring alert summary | Completed |
| Data card | Completed |
| Model card | Completed |
| Drift monitoring report | Completed |
| README | Completed |

Planned later modules include:

- Repeated-seed robustness experiments
- Unsupervised anomaly detection
- Model explainability
- Evidence-grounded RAG root-cause triage
- FastAPI model serving
- Streamlit dashboard
- MLflow tracking
- Docker setup
- Automated tests and CI

---

## 3. Repository Structure

```text
waferwatch/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── synthetic/
├── dashboard/
├── docs/
├── notebooks/
├── reports/
│   ├── data_card.md
│   ├── model_card.md
│   ├── drift_report.md
│   ├── monitoring_alert_summary.md
│   └── *.json
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── clean.py
│   │   └── validate.py
│   ├── features/
│   │   ├── build_features.py
│   │   ├── spc_features.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   ├── predict.py
│   │   └── compare.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── performance.py
│   │   └── alerts.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── tests/
├── requirements.txt
└── README.md

```

---

## 4. Current Pipeline

The current WaferWatch pipeline is:

```text
data validation
-> ingestion
-> cleaning
-> feature engineering
-> SPC feature engineering
-> feature selection
-> model training
-> model evaluation
-> model comparison
-> cost-sensitive thresholding
-> batch prediction
-> drift monitoring
-> performance monitoring
-> monitoring alert summary
-> documentation
```

---

## 5. Setup

Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies.

```powershell
pip install -r requirements.txt
```

---

## 6. Reproduce the Demo Pipeline

Run the modules in this order:

```powershell
python -m src.data.ingest
python -m src.data.clean
python -m src.features.spc_features
python -m src.features.build_features
python -m src.features.feature_selection
python -m src.models.train
python -m src.models.evaluate
python -m src.models.compare
python -m src.models.threshold
python -m src.models.predict
python -m src.monitoring.drift
python -m src.monitoring.performance
python -m src.monitoring.alerts
```

Most generated CSV files are saved under:

```text
data/processed/
```

Most generated JSON and Markdown reports are saved under:

```text
reports/
```

Generated data and model artifacts are intentionally ignored by Git.

---

## 7. Feature Engineering

WaferWatch currently creates three feature groups.

### 7.1 Sensor Aggregate Features

| Feature | Meaning |
|---|---|
| `sensor_mean` | Mean value across sensor columns |
| `sensor_std` | Standard deviation across sensor columns |
| `sensor_min` | Minimum sensor value |
| `sensor_max` | Maximum sensor value |
| `sensor_missing_count` | Number of missing sensor values |
| `sensor_missing_ratio` | Ratio of missing sensor values |

### 7.2 SPC Features

SPC means Statistical Process Control.

| Feature | Meaning |
|---|---|
| `spc_violation_count` | Number of sensor values outside control limits |
| `spc_violation_ratio` | Ratio of sensors outside control limits |
| `spc_max_abs_zscore` | Maximum absolute standardized sensor deviation |
| `spc_any_violation` | Whether any SPC violation occurred |
| `spc_top_violating_sensor` | Sensor with strongest SPC deviation |

### 7.3 Selected Features

The current selected model features are:

- `sensor_mean`
- `sensor_std`
- `sensor_min`
- `sensor_max`
- `spc_violation_count`
- `spc_max_abs_zscore`

---

## 8. Model Comparison

The current demo compares three feature strategies:

| Model | Feature strategy |
|---|---|
| Model A | Sensor aggregate features only |
| Model B | Sensor aggregate + SPC features |
| Model C | Sensor aggregate + SPC + feature selection |

Current controlled demo result:

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Accuracy | 0.875 | 1.000 | 1.000 |
| Precision | 0.750 | 1.000 | 1.000 |
| Recall | 0.600 | 1.000 | 1.000 |
| F1 | 0.667 | 1.000 | 1.000 |
| ROC-AUC | 0.611 | 1.000 | 1.000 |
| PR-AUC | 0.630 | 1.000 | 1.000 |
| Precision@K | 0.300 | 0.500 | 0.500 |
| Recall@K | 0.600 | 1.000 | 1.000 |
| False alarms per 100 lots | 4.167 | 0.000 | 0.000 |

Important interpretation:

> This is a controlled synthetic experiment. SPC features perform very well because the anomaly mechanism was intentionally injected into sensor shifts. This result validates the pipeline logic, not real-world fab performance.

### 8.2 Model Family and Anomaly Baseline Comparison

The selected-feature supervised baselines were compared against three unsupervised anomaly detection baselines on the same selected SPC-enhanced feature table.

| Metric | Logistic Regression | Random Forest | Gradient Boosting | Isolation Forest | PCA Anomaly | Autoencoder |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.000000 | 1.000000 | 1.000000 | 0.708333 | 0.958333 | 1.000000 |
| Precision | 1.000000 | 1.000000 | 1.000000 | 0.416667 | 0.833333 | 1.000000 |
| Recall | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| F1 | 1.000000 | 1.000000 | 1.000000 | 0.588235 | 0.909091 | 1.000000 |
| ROC-AUC | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| PR-AUC | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Precision@K | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| Recall@K | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| False alarms per 100 lots | 0.000000 | 0.000000 | 0.000000 | 29.166667 | 4.166667 | 0.000000 |

Interpretation:

In the current controlled synthetic SPC demo, the supervised models achieve perfect headline test metrics because the selected SPC features strongly encode the injected anomaly mechanism. Isolation Forest captures all held-out failed lots but creates more false alarms under its default threshold. PCA reconstruction-error anomaly detection captures all held-out failed lots with fewer false alarms than default Isolation Forest. Autoencoder-style reconstruction-error anomaly detection also captures all held-out failed lots and produces no false alarms in this controlled demo.

This comparison supports a practical manufacturing-style discussion: supervised classifiers perform strongly when labels are available, while unsupervised anomaly detectors remain valuable when confirmed failure labels are rare, delayed, or incomplete.

### 8.4 Robustness and Ablation Experiments

WaferWatch now includes robustness experiments to test whether strong demo results remain stable when key SPC features are removed or when the anomaly signal is weakened.

| Experiment type | Scenario | Purpose |
|---|---|---|
| Baseline | Full selected feature table | Establish reference model behavior |
| Feature ablation | Remove one or more SPC features | Test whether models depend too heavily on engineered SPC signals |
| Anomaly severity stress test | Shrink failed-lot features toward normal-lot means | Test whether models remain stable when anomaly signals become weaker |

Baseline full-feature results:

| Model | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Gradient Boosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Isolation Forest | 0.416667 | 1.000000 | 0.588235 | 1.000000 | 29.166667 |
| PCA Anomaly | 0.833333 | 1.000000 | 0.909091 | 1.000000 | 4.166667 |
| Autoencoder Anomaly | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

Key finding:

When all SPC features are removed, supervised models and reconstruction-error anomaly detectors no longer remain uniformly perfect. This is useful because it shows that the strong headline results are driven by meaningful engineered SPC signals rather than being treated as unexplained perfect metrics.

### 8.5 Isolation Forest Threshold Tuning

Isolation Forest produced strong anomaly ranking performance, but its default decision threshold generated more false alarms than the supervised models. To make the unsupervised workflow more operational, WaferWatch evaluates top-K review and escalation-rate policies.

| Policy | Review Count | Precision | Recall | F1 | False Alarms per 100 Lots | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_isolation_forest_threshold | 12 | 0.416667 | 1.000000 | 0.588235 | 29.166667 | 5 | 7 | 0 | 12 |
| top_3_review | 3 | 1.000000 | 0.600000 | 0.750000 | 0.000000 | 3 | 0 | 2 | 19 |
| top_5_review | 5 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| top_8_review | 8 | 0.625000 | 1.000000 | 0.769231 | 12.500000 | 5 | 3 | 0 | 16 |
| top_10_review | 10 | 0.500000 | 1.000000 | 0.666667 | 20.833333 | 5 | 5 | 0 | 14 |

Selected policy under a 10 false-alarms-per-100-lots budget:

```text
top_5_review
```

This selected policy preserves full recall while reducing false alarms to zero in the controlled demo. This result validates the usefulness of top-K review policies for converting anomaly scores into an operational engineer triage queue.
---

## 9. Calibration Analysis

WaferWatch includes calibration analysis because risk scores are later used for thresholding, escalation, and monitoring.

Current calibration demo results:

| Metric | Value |
|---|---:|
| Brier score | 0.004395 |
| Log loss | 0.043329 |
| Expected calibration error | 0.040760 |
| Maximum calibration error | 0.265852 |
| ROC-AUC | 1.000000 |
| PR-AUC | 1.000000 |

Interpretation:

The current calibration result validates the calibration workflow on a small synthetic demo. It should not be interpreted as production probability reliability.

---

## 10. Cost-Sensitive Thresholding

WaferWatch includes cost-sensitive thresholding because the default 0.5 classifier threshold may not be operationally appropriate.

Current thresholding demo:

| Strategy | Threshold | Total cost |
|---|---:|---:|
| Default threshold | 0.50 | 29.0 |
| Simple cost-sensitive threshold | 0.05 | 37.0 |
| Best threshold by realized cost | 0.42 | 10.0 |

This shows that maximizing recall alone is not always the best operational decision. A very low threshold may catch more risky lots but can also create excessive false alarms.

---

## 11. Monitoring

WaferWatch currently includes two monitoring layers.

### 10.1 Drift Monitoring

Drift monitoring compares reference-period and current-period feature distributions.

Current drift demo:

| Item | Value |
|---|---:|
| Reference rows | 40 |
| Current rows | 40 |
| Features monitored | 6 |
| Features with drift | 4 |

Drifted features:

- `sensor_std`
- `sensor_mean`
- `sensor_min`
- `spc_max_abs_zscore`

### 10.2 Performance Monitoring

Performance monitoring compares model performance after current-period labels become available.

Current performance demo:

| Metric | Reference | Current | Delta |
|---|---:|---:|---:|
| Accuracy | 1.000 | 0.675 | -0.325 |
| Precision | 1.000 | 0.4118 | -0.5882 |
| Recall | 1.000 | 0.7000 | -0.3000 |
| F1 | 1.000 | 0.5185 | -0.4815 |
| PR-AUC | 1.000 | 0.8373 | -0.1627 |
| False alarms per 100 lots | 0.000 | 25.000 | +25.000 |

Alert reasons:

- `recall_drop_exceeds_threshold`
- `pr_auc_drop_exceeds_threshold`
- `false_alarm_increase_exceeds_threshold`

### 10.3 Alert Summary

The combined alert level is currently:

```text
critical
```

This alert is triggered because both feature drift and model performance degradation are detected in the controlled monitoring demo.

---

## 12. Key Reports

| Report | Purpose |
|---|---|
| `reports/data_card.md` | Describes demo data, label definition, limitations, and appropriate use |
| `reports/model_card.md` | Describes model versions, feature strategies, evaluation, monitoring, and limitations |
| `reports/drift_report.md` | Documents drift and performance monitoring results |
| `reports/model_family_comparison_report.md` | Compares Logistic Regression and Random Forest on selected SPC features |
| `reports/random_forest_report.md` | Documents Random Forest metrics and feature importance |
| `reports/gradient_boosting_report.md` | Documents Gradient Boosting metrics and feature importance |
| `reports/isolation_forest_report.md` | Documents unsupervised Isolation Forest anomaly detection results |
| `reports/isolation_threshold_report.md` | Documents Isolation Forest threshold tuning and false-alarm budget analysis |
| `reports/pca_anomaly_report.md` | Documents PCA reconstruction-error anomaly detection results |
| `reports/autoencoder_anomaly_report.md` | Documents autoencoder-style reconstruction-error anomaly detection results |
| `reports/robustness_ablation_report.md` | Documents feature ablation and anomaly severity stress-test results |
| `reports/monitoring_alert_summary.md` | Engineer-readable combined alert summary |

---

## 13. Limitations

The current WaferWatch implementation has important limitations:

- Synthetic demo data only
- Small sample size
- Logistic Regression baseline only
- No real equipment, chamber, recipe, maintenance, or process-event data yet
- No real metrology, inspection, or yield outcome labels
- No external validation dataset
- No full root-cause retrieval pipeline yet
- No dashboard yet
- No model serving API yet
- No production deployment validation

These limitations are documented intentionally to prevent overclaiming.

---

## 14. Next Steps

Planned next steps:

- Add threshold recalibration experiments.
- Add repeated-seed robustness experiments.
- Add unsupervised anomaly detection models.
- Add prediction-score drift monitoring.
- Add model explainability.
- Add synthetic tool event logs and maintenance records.
- Add evidence-grounded RAG root-cause triage.
- Add Streamlit dashboard.
- Add FastAPI inference service.
- Add MLflow experiment tracking.
- Add automated tests.
- Add final thesis-style report.

---

## 15. Project Summary

WaferWatch currently demonstrates an end-to-end manufacturing AI anomaly monitoring workflow:

```text
clean data
-> engineer manufacturing-aware features
-> train baseline models
-> compare feature strategies
-> tune cost-sensitive thresholds
-> generate lot-level risk scores
-> monitor drift
-> monitor performance
-> summarize alerts for engineering review
```

The project is not production-ready, but it demonstrates a serious architecture for cost-sensitive, monitored, human-in-the-loop manufacturing AI.


