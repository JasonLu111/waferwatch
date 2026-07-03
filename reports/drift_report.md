# WaferWatch Drift and Performance Monitoring Report

## 1. Purpose

This report documents the monitoring layer of **WaferWatch: Cost-Sensitive Fab Anomaly Monitoring and Evidence-Grounded Root-Cause Triage**.

The goal of this monitoring layer is to detect whether the deployed anomaly monitoring pipeline remains reliable after new production data arrives.

WaferWatch separates monitoring into two complementary questions:

1. **Data drift monitoring**  
   Has the feature distribution changed between the reference period and the current period?

2. **Performance monitoring**  
   After labels become available, has the model's decision quality degraded?

This separation is important because data drift does not always imply model failure, and model performance can degrade even when some feature-level drift signals are subtle.

---

## 2. Monitoring Artifacts

The monitoring workflow currently produces the following artifacts:

| Artifact | Description |
|---|---|
| `drift_monitoring_report.json` | Feature-level data drift summary |
| `performance_monitoring_report.json` | Reference vs current model performance comparison |
| `monitoring_alert_summary.json` | Combined drift and performance alert summary |
| `monitoring_alert_summary.md` | Engineer-readable monitoring alert summary |

These reports are designed to support engineering review rather than fully automated decision-making.

---

## 3. Drift Monitoring Method

The drift monitoring module compares a reference period against a current period using numeric features from the selected SPC-enhanced feature table.

The monitored feature table contains the following model features:

- `sensor_mean`
- `sensor_std`
- `sensor_min`
- `sensor_max`
- `spc_violation_count`
- `spc_max_abs_zscore`

The current demo uses 40 rows as the reference period and 40 rows as the current period.

For demonstration purposes, synthetic drift is injected into selected current-period features. In a real deployment, the current period would come from newly processed production lots.

---

## 4. Drift Metrics

WaferWatch currently monitors feature drift using:

| Metric | Meaning |
|---|---|
| Standardized mean shift | Measures how much the current mean moved relative to the reference standard deviation |
| Standard deviation ratio | Compares current-period variability with reference-period variability |
| Missing rate shift | Detects changes in missing-value behavior |
| Population Stability Index | Measures distribution-level change between reference and current periods |

The current drift thresholds are:

| Threshold | Value |
|---|---:|
| PSI threshold | 0.25 |
| Mean shift threshold | 1.0 |
| Missing rate shift threshold | 0.10 |

---

## 5. Drift Monitoring Result

The current drift monitoring run detected drift in 4 out of 6 monitored features.

| Summary Item | Value |
|---|---:|
| Reference rows | 40 |
| Current rows | 40 |
| Features monitored | 6 |
| Features with drift | 4 |
| Overall drift detected | True |

The drifted features were:

- `sensor_std`
- `sensor_mean`
- `sensor_min`
- `spc_max_abs_zscore`

This result suggests that the current-period feature distribution differs meaningfully from the reference period. In a real production setting, these drifted features should be compared against recent equipment, chamber, recipe, maintenance, sampling, and process changes.

---

## 6. Performance Monitoring Method

Performance monitoring evaluates whether the model still makes reliable decisions after current-period labels become available.

The current demo compares:

- Reference-period model performance
- Current-period model performance

For demonstration purposes, current-period degradation is simulated by weakening the alignment between SPC signals and true labels. In a real deployment, this step would use actual post-process labels or downstream quality outcomes.

The selected model used in this monitoring demo is:

```text
spc_selected_logistic_regression.joblib

```

The monitored feature table is:

```text
demo_spc_selected_feature_table.csv
```

---

## 7. Performance Monitoring Result

The monitoring run detected a clear performance alert.

| Metric | Reference | Current | Delta |
|---|---:|---:|---:|
| Accuracy | 1.0 | 0.675 | -0.325 |
| Precision | 1.0 | 0.4118 | -0.5882 |
| Recall | 1.0 | 0.7000 | -0.3000 |
| F1 | 1.0 | 0.5185 | -0.4815 |
| PR-AUC | 1.0 | 0.8373 | -0.1627 |
| False alarms per 100 lots | 0.0 | 25.0 | +25.0 |

The performance alert was triggered by:

- `recall_drop_exceeds_threshold`
- `pr_auc_drop_exceeds_threshold`
- `false_alarm_increase_exceeds_threshold`

The result shows that the current-period model is missing more risky lots and creating substantially more false alarms. This is operationally important because manufacturing anomaly monitoring must balance missed-risk cost and engineer review burden.

---

## 8. Combined Alert Summary

The combined alert module integrates drift and performance signals into a single operational alert summary.

Current alert level:

```text
critical
```

Engineer review required:

```text
True
```

This critical alert is triggered because both of the following are true:

1. Feature distribution drift is detected.
2. Model performance degradation is detected.

This does not automatically identify the root cause. Instead, it indicates that the model, feature pipeline, threshold policy, or underlying process condition should be reviewed.

---

## 9. Recommended Engineering Actions

Based on the current alert summary, WaferWatch recommends:

1. Review the drifted features and compare them with recent tool, chamber, recipe, and maintenance changes.
2. Prioritize investigation of:
   - `sensor_std`
   - `sensor_mean`
   - `sensor_min`
   - `spc_max_abs_zscore`
3. Review current-period false positives and false negatives.
4. Check whether current labels reflect a new operating regime, recipe mix, tool condition, or process variation.
5. Evaluate alert fatigue risk caused by increased false alarms.
6. Consider threshold recalibration if false alarm burden is too high.
7. Investigate missed risky lots and consider retraining, recalibration, or adding new process/event features.
8. Compare top-K risk ranking quality against actual outcomes.

---

## 10. Limitations

This monitoring report is based on a controlled synthetic demo. Therefore:

- The performance numbers should not be interpreted as real fab performance.
- The drift injection is artificial and used only to test monitoring logic.
- The selected dataset is small.
- Real deployment would require production-scale data, stable label definitions, process context, and validation across tools, products, recipes, and time periods.

The value of this demo is that it verifies the monitoring architecture and produces interpretable artifacts that can be extended to real manufacturing data.

---

## 11. Next Steps

The next development steps are:

1. Add an alert dashboard view.
2. Add threshold recalibration logic.
3. Add model retraining triggers.
4. Add root-cause evidence retrieval.
5. Link drifted features with process events, maintenance logs, and engineering notes.
6. Extend monitoring from feature drift to prediction-score drift.
7. Add automated model card updates after each monitoring run.

---

## 12. Summary

WaferWatch now includes a working monitoring layer that can:

- Detect feature distribution drift.
- Monitor model performance after labels become available.
- Identify recall degradation, PR-AUC degradation, and false alarm increase.
- Combine drift and performance signals into a practical alert level.
- Generate engineer-readable alert summaries.

This strengthens WaferWatch from a simple modeling project into a more realistic manufacturing AI decision-support system.