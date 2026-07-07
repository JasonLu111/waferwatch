# WaferWatch Final Project Packaging

## 1. Project One-Liner

WaferWatch is an end-to-end manufacturing AI demo for cost-sensitive fab anomaly monitoring and evidence-grounded root-cause triage.

It combines data validation, cleaning, SPC-style feature engineering, supervised classification, unsupervised anomaly detection, threshold tuning, monitoring, robustness experiments, and engineering-facing triage reports.

---

## 2. 60-Second Interview Narrative

WaferWatch is a portfolio project designed to simulate a manufacturing AI workflow for detecting abnormal wafer-lot behavior and translating model outputs into engineering review evidence.

I started by building a structured data pipeline: raw demo data ingestion, validation, cleaning, and feature engineering. Then I created SPC-enhanced features such as violation counts and maximum absolute z-scores. After feature selection, I compared supervised models such as Logistic Regression, Random Forest, and Gradient Boosting with unsupervised anomaly detectors such as Isolation Forest, PCA reconstruction error, and an autoencoder-style reconstruction model.

Because manufacturing anomaly detection is not only about accuracy, I evaluated recall, precision, PR-AUC, false alarms per 100 lots, threshold behavior, and cost-sensitive decision rules. I also added robustness experiments, feature ablation, anomaly severity stress tests, and repeated-seed evaluation to check whether results were stable.

Finally, I built an evidence-grounded root-cause triage module. It does not claim causal proof. Instead, it maps abnormal features to cause hypotheses and recommended engineering review actions, creating a bridge from model prediction to process or equipment investigation.

---

## 3. End-to-End Architecture

```text
Raw demo data
-> Data validation
-> Data cleaning
-> SPC feature engineering
-> Feature selection
-> Supervised baselines
-> Unsupervised anomaly detection
-> Cost-sensitive threshold tuning
-> Monitoring and calibration
-> Robustness and ablation experiments
-> Repeated-seed robustness experiments
-> Evidence-grounded root-cause triage
-> Final engineering review reports

```

---

## 4. Main Components

| Component | File / Report | Purpose |
|---|---|---|
| Data ingestion | `src/data/ingest.py` | Create and load demo raw/interim data |
| Data validation | `src/data/validate.py` | Check schema, missing values, duplicates, and basic data quality |
| Data cleaning | `src/data/clean.py` | Produce cleaned feature-ready data |
| SPC features | `src/features/spc_features.py` | Generate SPC-style manufacturing signals |
| Feature selection | `src/features/feature_selection.py` | Select compact SPC-enhanced feature set |
| Supervised baseline | `src/models/train.py` | Train baseline supervised classifier |
| Model evaluation | `src/models/evaluate.py` | Evaluate model quality with manufacturing-relevant metrics |
| Threshold tuning | `src/models/threshold.py` | Tune threshold using cost-sensitive decision logic |
| Model scoring | `src/models/predict.py` | Score lots and produce review candidates |
| Random Forest | `src/models/random_forest.py` | Tree-based supervised baseline |
| Gradient Boosting | `src/models/gradient_boosting.py` | Boosted supervised baseline |
| Isolation Forest | `src/models/isolation_forest.py` | Unsupervised anomaly detection baseline |
| PCA anomaly detection | `src/models/pca_anomaly.py` | Reconstruction-error anomaly baseline |
| Autoencoder anomaly detection | `src/models/autoencoder_anomaly.py` | Lightweight neural reconstruction anomaly baseline |
| Model family comparison | `src/models/family_compare.py` | Compare six model families under one framework |
| Robustness and ablation | `src/models/robustness_ablation.py` | Test feature ablation and anomaly severity stress cases |
| Repeated-seed robustness | `src/models/repeated_seed_robustness.py` | Test split sensitivity across multiple random seeds |
| Root-cause triage | `src/triage/root_cause_triage.py` | Map feature evidence to cause hypotheses and review actions |

---

## 5. Technical Highlights

### 5.1 Manufacturing-Oriented Evaluation

WaferWatch does not rely only on accuracy. It evaluates:

- Recall
- Precision
- F1 score
- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- False alarms per 100 lots
- Cost-sensitive threshold behavior
- Confusion matrix structure

This is important because manufacturing anomaly detection often has imbalanced labels, costly missed failures, and limited engineering review capacity.

### 5.2 Supervised and Unsupervised Comparison

The project compares six model families:

| Model family | Type |
|---|---|
| Logistic Regression | Supervised |
| Random Forest | Supervised |
| Gradient Boosting | Supervised |
| Isolation Forest | Unsupervised anomaly detection |
| PCA reconstruction error | Unsupervised anomaly detection |
| Autoencoder-style reconstruction error | Unsupervised anomaly detection |

This allows the project to discuss two practical settings:

1. Labels are available and supervised learning is feasible.
2. Labels are rare, delayed, or incomplete, so anomaly detection is needed.

### 5.3 Robustness Experiments

WaferWatch includes:

- Feature ablation
- SPC feature removal
- Anomaly severity stress tests
- Repeated-seed experiments

These experiments make the demo more defensible because they test whether conclusions depend on one feature group, one anomaly strength, or one train-test split.

### 5.4 Evidence-Grounded Root-Cause Triage

The triage module creates:

- Cause-hypothesis table
- Feature contribution ranking
- Lot-level triage reports
- Recommended engineering review actions

It avoids unsupported causal claims. The output is framed as engineering triage evidence, not true causal discovery.

---

## 6. Key Result Summary

In the controlled synthetic SPC demo:

| Finding | Interpretation |
|---|---|
| Supervised baselines perform strongly | SPC-enhanced features encode the injected anomaly mechanism well |
| Isolation Forest captures abnormal lots but creates more false alarms | Threshold tuning and review-budget control are necessary |
| PCA anomaly detection provides a useful reconstruction-error baseline | Reconstruction methods can work when abnormal lots deviate strongly from normal-reference behavior |
| Autoencoder-style reconstruction works in this controlled demo | Neural reconstruction can be useful but should be validated carefully |
| Ablation changes model behavior | SPC features are meaningful drivers, not decoration |
| Repeated-seed results show split stability for supervised baselines | The result is not only from one lucky train-test split |
| Root-cause triage connects evidence to engineering review | The project moves beyond prediction into actionability |

---

## 7. Reproduction Guide

From the project root:

```powershell
python -m src.data.ingest
python -m src.data.validate
python -m src.data.clean
python -m src.features.spc_features
python -m src.features.build_features
python -m src.features.feature_selection

python -m src.models.train
python -m src.models.evaluate
python -m src.models.threshold
python -m src.models.predict

python -m src.models.random_forest
python -m src.models.gradient_boosting
python -m src.models.isolation_forest
python -m src.models.isolation_threshold
python -m src.models.pca_anomaly
python -m src.models.autoencoder_anomaly
python -m src.models.family_compare

python -m src.models.calibration
python -m src.models.robustness_ablation
python -m src.models.repeated_seed_robustness

python -m src.triage.root_cause_triage
```

Main reports:

| Report | Purpose |
|---|---|
| `reports/model_card.md` | Model behavior, limitations, and evaluation summary |
| `reports/data_card.md` | Data assumptions and data quality context |
| `reports/model_family_comparison_report.md` | Six-model comparison |
| `reports/robustness_ablation_report.md` | Feature ablation and severity stress tests |
| `reports/repeated_seed_robustness_report.md` | Repeated-seed stability checks |
| `reports/root_cause_triage_report.md` | Evidence-grounded engineering triage output |

---

## 8. Interview Talking Points

### Q1. Why is this project not just a normal classification project?

Because manufacturing anomaly detection has operational constraints. A model must consider recall, false alarms, engineering review capacity, and cost-sensitive threshold decisions. WaferWatch evaluates these factors instead of relying only on accuracy.

### Q2. Why include unsupervised anomaly detection?

In manufacturing settings, labels may be rare, delayed, incomplete, or expensive to obtain. Unsupervised methods such as Isolation Forest, PCA reconstruction error, and autoencoder-style reconstruction provide alternatives when supervised labels are limited.

### Q3. Why use SPC features?

SPC features encode process-control behavior in a way that is interpretable to manufacturing engineers. They help connect statistical model signals to process monitoring concepts.

### Q4. Why do ablation experiments matter?

Ablation experiments test whether strong results depend on meaningful features. If performance drops after removing SPC features, that supports the argument that SPC signals are important model drivers.

### Q5. Why does the triage module avoid causal language?

Feature deviations can suggest what engineers should review, but they do not prove root cause. True root-cause validation requires process history, equipment logs, maintenance records, metrology, and engineering judgment.

---

## 9. Limitations

This project uses controlled synthetic demo data. The results should not be interpreted as production fab performance.

Important limitations:

- The dataset is synthetic and simplified.
- The anomaly mechanism is controlled.
- Real sensor data would be higher-dimensional and noisier.
- Real labels may be delayed or partially observed.
- Root-cause triage is hypothesis generation, not causal proof.
- Deployment would require monitoring, governance, access control, and process-owner validation.

---

## 10. Future Improvements

Potential next improvements:

- Add real or more realistic time-series sensor simulation.
- Add SQL-based data extraction examples.
- Add experiment tracking.
- Add dashboard-style visualization.
- Add model registry and deployment mockup.
- Add LLM-assisted retrieval over process notes or engineering logs.
- Add human-in-the-loop feedback for triage labels.
- Add more realistic drift simulation.
- Add final GitHub README polish and architecture diagram.

---

## 11. Final Positioning

WaferWatch is best presented as a manufacturing AI portfolio project that demonstrates:

- End-to-end ML workflow design
- Manufacturing-aware feature engineering
- Cost-sensitive anomaly monitoring
- Supervised and unsupervised model comparison
- Robustness and ablation thinking
- Evidence-grounded engineering triage
- Responsible communication of limitations

The strongest interview message is:

> I built WaferWatch to show that I can move beyond model training. I can design a manufacturing AI workflow that starts from data quality, builds interpretable SPC-style features, evaluates models under operational constraints, checks robustness, and turns anomaly scores into engineering-facing triage evidence.