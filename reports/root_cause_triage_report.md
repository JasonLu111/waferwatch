# WaferWatch Evidence-Grounded Root-Cause Triage Report

## 1. Purpose

This report converts anomaly-monitoring outputs into structured engineering triage evidence.
It does not claim true causal discovery. It generates cause hypotheses based on feature deviations from normal-reference behavior.

## 2. Cause-Hypothesis Table

| Feature | Cause family | Evidence type | Recommended review |
|---|---|---|---|
| `sensor_mean` | Process center shift | Sensor mean evidence | Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions. |
| `sensor_std` | Process instability | Sensor variation evidence | Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows. |
| `sensor_min` | Lower-tail excursion | Sensor low-tail evidence | Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts. |
| `sensor_max` | Upper-tail excursion | Sensor high-tail evidence | Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts. |
| `spc_violation_count` | SPC rule violation | SPC count evidence | Review control-chart rule violations, recipe context, and recent process-window changes. |
| `spc_max_abs_zscore` | SPC excursion | SPC magnitude evidence | Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts. |

## 3. Triage Lot Summary

| Rank | Lot ID | True Label | Max Abs Z-score | Top Feature | Top Cause Family | Evidence Strength |
|---:|---|---:|---:|---|---|---|
| 1 | `LOT_SPC_061` | 1 | 18.830456 | `spc_max_abs_zscore` | SPC excursion | critical |
| 2 | `LOT_SPC_058` | 1 | 16.646828 | `spc_max_abs_zscore` | SPC excursion | critical |
| 3 | `LOT_SPC_071` | 1 | 14.499853 | `spc_max_abs_zscore` | SPC excursion | critical |
| 4 | `LOT_SPC_077` | 1 | 13.387271 | `spc_max_abs_zscore` | SPC excursion | critical |
| 5 | `LOT_SPC_079` | 1 | 10.730695 | `spc_max_abs_zscore` | SPC excursion | critical |
| 6 | `LOT_SPC_036` | 1 | 9.969851 | `spc_max_abs_zscore` | SPC excursion | critical |
| 7 | `LOT_SPC_024` | 1 | 9.519057 | `spc_max_abs_zscore` | SPC excursion | critical |
| 8 | `LOT_SPC_078` | 1 | 9.236081 | `spc_max_abs_zscore` | SPC excursion | critical |
| 9 | `LOT_SPC_068` | 1 | 8.367564 | `spc_max_abs_zscore` | SPC excursion | critical |
| 10 | `LOT_SPC_056` | 1 | 8.264572 | `spc_max_abs_zscore` | SPC excursion | critical |

## 4. Lot-Level Evidence Reports

### 4.1 Lot `LOT_SPC_061`

- True label in controlled demo: `1`
- Max absolute feature z-score: `18.830456`
- Top cause families: `SPC excursion; Lower-tail excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 11.429859 | 1.552137 | 18.830456 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_min` | 6.787246 | 10.993135 | -11.444759 | critical | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |
| `sensor_std` | 3.349757 | 1.662437 | 7.335197 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.2 Lot `LOT_SPC_058`

- True label in controlled demo: `1`
- Max absolute feature z-score: `16.646828`
- Top cause families: `SPC excursion; Process center shift; Lower-tail excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 10.284413 | 1.552137 | 16.646828 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 13.736873 | 12.985070 | 4.200017 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |
| `sensor_min` | 12.419063 | 10.993135 | 3.880133 | critical | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.

### 4.3 Lot `LOT_SPC_071`

- True label in controlled demo: `1`
- Max absolute feature z-score: `14.499853`
- Top cause families: `SPC excursion; Lower-tail excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 9.158193 | 1.552137 | 14.499853 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_min` | 7.623870 | 10.993135 | -9.168199 | critical | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |
| `sensor_mean` | 12.049297 | 12.985070 | -5.227785 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.4 Lot `LOT_SPC_077`

- True label in controlled demo: `1`
- Max absolute feature z-score: `13.387271`
- Top cause families: `SPC excursion; Lower-tail excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 8.574577 | 1.552137 | 13.387271 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_min` | 7.838808 | 10.993135 | -8.583325 | critical | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |
| `sensor_std` | 2.853721 | 1.662437 | 5.178808 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.5 Lot `LOT_SPC_079`

- True label in controlled demo: `1`
- Max absolute feature z-score: `10.730695`
- Top cause families: `SPC excursion; Lower-tail excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 7.181041 | 1.552137 | 10.730695 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_min` | 8.352029 | 10.993135 | -7.186786 | critical | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |
| `sensor_std` | 2.710512 | 1.662437 | 4.556240 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.6 Lot `LOT_SPC_036`

- True label in controlled demo: `1`
- Max absolute feature z-score: `9.969851`
- Top cause families: `SPC excursion; Upper-tail excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 6.781932 | 1.552137 | 9.969851 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 17.723998 | 15.081540 | 6.094566 | critical | Upper-tail excursion | The lot may contain an unusual upper-tail sensor pattern relative to normal-reference behavior. |
| `sensor_std` | 2.642372 | 1.662437 | 4.260019 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.7 Lot `LOT_SPC_024`

- True label in controlled demo: `1`
- Max absolute feature z-score: `9.519057`
- Top cause families: `SPC excursion; Upper-tail excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 6.545463 | 1.552137 | 9.519057 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 18.131247 | 15.081540 | 7.033846 | critical | Upper-tail excursion | The lot may contain an unusual upper-tail sensor pattern relative to normal-reference behavior. |
| `sensor_std` | 2.698869 | 1.662437 | 4.505624 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.8 Lot `LOT_SPC_078`

- True label in controlled demo: `1`
- Max absolute feature z-score: `9.236081`
- Top cause families: `SPC excursion; Process center shift; Lower-tail excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 6.397024 | 1.552137 | 9.236081 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 12.478463 | 12.985070 | -2.830210 | high | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |
| `sensor_min` | 10.055782 | 10.993135 | -2.550658 | high | Lower-tail excursion | The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.
- Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.

### 4.9 Lot `LOT_SPC_068`

- True label in controlled demo: `1`
- Max absolute feature z-score: `8.367564`
- Top cause families: `SPC excursion; Upper-tail excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 5.941434 | 1.552137 | 8.367564 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 17.845876 | 15.081540 | 6.375667 | critical | Upper-tail excursion | The lot may contain an unusual upper-tail sensor pattern relative to normal-reference behavior. |
| `sensor_mean` | 13.826550 | 12.985070 | 4.701005 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.10 Lot `LOT_SPC_056`

- True label in controlled demo: `1`
- Max absolute feature z-score: `8.264572`
- Top cause families: `SPC excursion; Process instability; Upper-tail excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_max_abs_zscore` | 5.887409 | 1.552137 | 8.264572 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_std` | 2.851241 | 1.662437 | 5.168024 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |
| `sensor_max` | 17.233025 | 15.081540 | 4.962186 | critical | Upper-tail excursion | The lot may contain an unusual upper-tail sensor pattern relative to normal-reference behavior. |

Recommended engineering reviews:

- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.
- Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts.

## 5. Interpretation

The triage module links anomalous feature evidence to structured cause hypotheses and review actions.
This creates a bridge from model output to engineering investigation, while avoiding unsupported causal claims.
In a real deployment, these hypotheses would need to be validated with process history, tool logs, maintenance records, metrology, and engineering judgment.
