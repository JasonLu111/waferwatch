# WaferWatch Isolation Forest Threshold Tuning Report

## 1. Purpose

This report evaluates threshold and review-budget policies for Isolation Forest anomaly scores.
The goal is to reduce false alarms while preserving high recall for risky lots.

## 2. Why Threshold Tuning Matters

Isolation Forest produces continuous anomaly scores. The default model threshold may be too conservative or too aggressive for an operational monitoring workflow.
A review-budget policy converts risk scores into a practical engineer triage queue, such as reviewing only the top-K highest-risk lots.

## 3. Policy Comparison

| Policy | Review Count | Review Rate | Precision | Recall | F1 | False Alarms per 100 Lots | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_isolation_forest_threshold | 12 | 0.500000 | 0.416667 | 1.000000 | 0.588235 | 29.166667 | 5 | 7 | 0 | 12 |
| top_3_review | 3 | 0.125000 | 1.000000 | 0.600000 | 0.750000 | 0.000000 | 3 | 0 | 2 | 19 |
| top_5_review | 5 | 0.208333 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| top_8_review | 8 | 0.333333 | 0.625000 | 1.000000 | 0.769231 | 12.500000 | 5 | 3 | 0 | 16 |
| top_10_review | 10 | 0.416667 | 0.500000 | 1.000000 | 0.666667 | 20.833333 | 5 | 5 | 0 | 14 |
| top_12_review | 12 | 0.500000 | 0.416667 | 1.000000 | 0.588235 | 29.166667 | 5 | 7 | 0 | 12 |
| top_15_review | 15 | 0.625000 | 0.333333 | 1.000000 | 0.500000 | 41.666667 | 5 | 10 | 0 | 9 |
| top_10pct_review | 3 | 0.125000 | 1.000000 | 0.600000 | 0.750000 | 0.000000 | 3 | 0 | 2 | 19 |
| top_20pct_review | 5 | 0.208333 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 5 | 0 | 0 | 19 |
| top_30pct_review | 8 | 0.333333 | 0.625000 | 1.000000 | 0.769231 | 12.500000 | 5 | 3 | 0 | 16 |
| top_40pct_review | 10 | 0.416667 | 0.500000 | 1.000000 | 0.666667 | 20.833333 | 5 | 5 | 0 | 14 |
| top_50pct_review | 12 | 0.500000 | 0.416667 | 1.000000 | 0.588235 | 29.166667 | 5 | 7 | 0 | 12 |

## 4. Selected Policy Under False-Alarm Budget

- Budget: `10.0` false alarms per 100 lots
- Selected policy: `top_5_review`
- Precision: `1.000000`
- Recall: `1.000000`
- False alarms per 100 lots: `0.000000`
- Review count: `5`

## 5. Top Scored Lots

| Rank | Lot ID | True Label | Risk Score |
|---:|---|---:|---:|
| 1 | `LOT_SPC_024` | 1 | 0.195428 |
| 2 | `LOT_SPC_077` | 1 | 0.168547 |
| 3 | `LOT_SPC_078` | 1 | 0.124095 |
| 4 | `LOT_SPC_027` | 1 | 0.123812 |
| 5 | `LOT_SPC_021` | 1 | 0.099447 |
| 6 | `LOT_SPC_034` | 0 | 0.036980 |
| 7 | `LOT_SPC_025` | 0 | 0.032813 |
| 8 | `LOT_SPC_065` | 0 | 0.025443 |
| 9 | `LOT_SPC_047` | 0 | 0.009712 |
| 10 | `LOT_SPC_003` | 0 | 0.006538 |

## 6. Interpretation

The default Isolation Forest threshold captures all held-out failed lots in this demo, but it also produces several false alarms. Top-K and escalation-rate policies provide a more operational way to control review workload. In this controlled synthetic demo, a smaller top-K review budget can preserve high recall while reducing false alarms. This result should be interpreted as a workflow validation rather than evidence of real production performance.
