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
| Cost-sensitive thresholding | Completed |
| Batch prediction | Completed |
| Drift monitoring | Completed |
| Performance monitoring | Completed |
| Monitoring alert summary | Completed |
| Data card | Completed |
| Model card | Completed |
| Drift monitoring report | Completed |
| README | In progress |

Planned later modules include:

- Calibration analysis
- Random Forest and gradient boosting baselines
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
→ ingestion
→ cleaning
→ feature engineering
→ SPC feature engineering
→ feature selection
→ model training
→ model evaluation
→ model comparison
→ cost-sensitive thresholding
→ batch prediction
→ drift monitoring
→ performance monitoring
→ monitoring alert summary
→ documentation
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

---

## 9. Cost-Sensitive Thresholding

WaferWatch includes cost-sensitive thresholding because the default 0.5 classifier threshold may not be operationally appropriate.

Current thresholding demo:

| Strategy | Threshold | Total cost |
|---|---:|---:|
| Default threshold | 0.50 | 29.0 |
| Simple cost-sensitive threshold | 0.05 | 37.0 |
| Best threshold by realized cost | 0.42 | 10.0 |

This shows that maximizing recall alone is not always the best operational decision. A very low threshold may catch more risky lots but can also create excessive false alarms.

---

## 10. Monitoring

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

## 11. Key Reports

| Report | Purpose |
|---|---|
| `reports/data_card.md` | Describes demo data, label definition, limitations, and appropriate use |
| `reports/model_card.md` | Describes model versions, feature strategies, evaluation, monitoring, and limitations |
| `reports/drift_report.md` | Documents drift and performance monitoring results |
| `reports/monitoring_alert_summary.md` | Engineer-readable combined alert summary |

---

## 12. Limitations

The current WaferWatch implementation has important limitations:

- Synthetic demo data only
- Small sample size
- Logistic Regression baseline only
- No real equipment, chamber, recipe, maintenance, or process-event data yet
- No real metrology, inspection, or yield outcome labels
- No external validation dataset
- No calibration analysis yet
- No full root-cause retrieval pipeline yet
- No dashboard yet
- No model serving API yet
- No production deployment validation

These limitations are documented intentionally to prevent overclaiming.

---

## 13. Next Steps

Planned next steps:

- Add calibration analysis.
- Add threshold recalibration experiments.
- Add Random Forest and gradient boosting baselines.
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

## 14. Project Summary

WaferWatch currently demonstrates an end-to-end manufacturing AI anomaly monitoring workflow:

```text
clean data
→ engineer manufacturing-aware features
→ train baseline models
→ compare feature strategies
→ tune cost-sensitive thresholds
→ generate lot-level risk scores
→ monitor drift
→ monitor performance
→ summarize alerts for engineering review
```

The project is not production-ready, but it demonstrates a serious architecture for cost-sensitive, monitored, human-in-the-loop manufacturing AI.
