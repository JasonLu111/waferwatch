# WaferWatch Monitoring Alert Summary

**Alert level:** `critical`
**Requires engineer review:** `True`

## 1. Drift Status

- Overall drift detected: `True`
- Features monitored: `6`
- Features with drift: `4`
- Drifted features: sensor_std, sensor_mean, sensor_min, spc_max_abs_zscore

## 2. Performance Status

- Performance alert: `True`
- Alert reasons: recall_drop_exceeds_threshold, pr_auc_drop_exceeds_threshold, false_alarm_increase_exceeds_threshold

### Reference vs Current Metrics

| Metric | Reference | Current | Delta current - reference |
|---|---:|---:|---:|
| accuracy | 1.0 | 0.675 | -0.32499999999999996 |
| precision | 1.0 | 0.4117647058823529 | -0.5882352941176471 |
| recall | 1.0 | 0.7 | -0.30000000000000004 |
| f1 | 1.0 | 0.5185185185185185 | -0.4814814814814815 |
| pr_auc | 1.0 | 0.8372674109516214 | -0.16273258904837862 |
| false_alarms_per_100_lots | 0.0 | 25.0 | 25.0 |

## 3. Recommended Actions

- Review drifted features and compare them with recent tool, chamber, recipe, and maintenance changes.
- Prioritize investigation of drifted features: sensor_std, sensor_mean, sensor_min, spc_max_abs_zscore.
- Review current-period false positives and false negatives to understand whether the alert threshold or model signal has degraded.
- Check whether current labels reflect a new operating regime, recipe mix, tool condition, or process variation.
- Evaluate alert fatigue risk and consider raising the threshold or limiting escalation volume per shift.
- Investigate missed risky lots and consider retraining, recalibration, or adding new process/event features.
- Check whether risk ranking quality degraded; compare top-K lots against actual outcomes.

## 4. Interpretation Note

This alert summary combines data drift and model performance signals. It is a decision-support artifact, not an automatic root-cause diagnosis.
