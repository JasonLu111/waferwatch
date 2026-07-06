# WaferWatch Evidence-Grounded Root-Cause Triage Report

## 1. Purpose

This report converts anomaly-monitoring outputs into structured engineering triage evidence.
It does not claim true causal discovery. It generates cause hypotheses based on feature deviations from normal-reference behavior.

## 2. Cause-Hypothesis Table

| Feature | Cause family | Evidence type | Recommended review |
|---|---|---|---|
| `sensor_mean` | Process center shift | Sensor mean evidence | Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions. |
| `sensor_std` | Process instability | Sensor variation evidence | Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows. |
| `sensor_min` | Low-side excursion | Sensor low-tail evidence | Review possible under-processing, sensor dropout, or low-side transient events. |
| `sensor_max` | High-side excursion | Sensor high-tail evidence | Review possible over-processing, transient spikes, tool instability, or high-side excursion events. |
| `spc_violation_count` | SPC rule violation | SPC count evidence | Review control-chart rule violations, recipe context, and recent process-window changes. |
| `spc_max_abs_zscore` | SPC excursion | SPC magnitude evidence | Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts. |

## 3. Triage Lot Summary

| Rank | Lot ID | True Label | Max Abs Z-score | Top Feature | Top Cause Family | Evidence Strength |
|---:|---|---:|---:|---|---|---|
| 1 | `LOT_SPC_060` | 1 | 2000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 2 | `LOT_SPC_021` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 3 | `LOT_SPC_023` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 4 | `LOT_SPC_024` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 5 | `LOT_SPC_027` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 6 | `LOT_SPC_030` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 7 | `LOT_SPC_036` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 8 | `LOT_SPC_041` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 9 | `LOT_SPC_056` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |
| 10 | `LOT_SPC_058` | 1 | 1000000000.000000 | `spc_violation_count` | SPC rule violation | critical |

## 4. Lot-Level Evidence Reports

### 4.1 Lot `LOT_SPC_060`

- True label in controlled demo: `1`
- Max absolute feature z-score: `2000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; High-side excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 2.000000 | 0.000000 | 2000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 5.031321 | 1.552137 | 6.632564 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 17.415897 | 15.081540 | 5.383962 | critical | High-side excursion | The lot may contain unusually high sensor readings relative to normal-reference behavior. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review possible over-processing, transient spikes, tool instability, or high-side excursion events.

### 4.2 Lot `LOT_SPC_021`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 5.124998 | 1.552137 | 6.811144 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 12.282908 | 12.985070 | -3.922698 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.3 Lot `LOT_SPC_023`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 4.868195 | 1.552137 | 6.321588 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 12.518667 | 12.985070 | -2.605604 | high | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.4 Lot `LOT_SPC_024`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; High-side excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 6.545463 | 1.552137 | 9.519057 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 18.131247 | 15.081540 | 7.033846 | critical | High-side excursion | The lot may contain unusually high sensor readings relative to normal-reference behavior. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review possible over-processing, transient spikes, tool instability, or high-side excursion events.

### 4.5 Lot `LOT_SPC_027`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 4.552708 | 1.552137 | 5.720157 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 13.422410 | 12.985070 | 2.443241 | high | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.6 Lot `LOT_SPC_030`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 5.035527 | 1.552137 | 6.640582 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 12.630598 | 12.985070 | -1.980292 | moderate | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.7 Lot `LOT_SPC_036`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; High-side excursion`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 6.781932 | 1.552137 | 9.969851 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_max` | 17.723998 | 15.081540 | 6.094566 | critical | High-side excursion | The lot may contain unusually high sensor readings relative to normal-reference behavior. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review possible over-processing, transient spikes, tool instability, or high-side excursion events.

### 4.8 Lot `LOT_SPC_041`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 5.617026 | 1.552137 | 7.749126 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 12.374208 | 12.985070 | -3.412638 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

### 4.9 Lot `LOT_SPC_056`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process instability`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 5.887409 | 1.552137 | 8.264572 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_std` | 2.851241 | 1.662437 | 5.168024 | critical | Process instability | The lot may show higher within-lot sensor variability or unstable process behavior. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.

### 4.10 Lot `LOT_SPC_058`

- True label in controlled demo: `1`
- Max absolute feature z-score: `1000000000.000000`
- Top cause families: `SPC rule violation; SPC excursion; Process center shift`

| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |
|---|---:|---:|---:|---|---|---|
| `spc_violation_count` | 1.000000 | 0.000000 | 1000000000.000000 | critical | SPC rule violation | The lot may have repeated process-control rule violations. |
| `spc_max_abs_zscore` | 10.284413 | 1.552137 | 16.646828 | critical | SPC excursion | The lot may contain a large standardized process excursion. |
| `sensor_mean` | 13.736873 | 12.985070 | 4.200017 | critical | Process center shift | The average sensor behavior may have shifted away from the normal reference pattern. |

Recommended engineering reviews:

- Review control-chart rule violations, recipe context, and recent process-window changes.
- Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.
- Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.

## 5. Interpretation

The triage module links anomalous feature evidence to structured cause hypotheses and review actions.
This creates a bridge from model output to engineering investigation, while avoiding unsupported causal claims.
In a real deployment, these hypotheses would need to be validated with process history, tool logs, maintenance records, metrology, and engineering judgment.
