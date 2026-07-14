# Synthetic Data V2 Quality and Reproducibility Report

## Scope

This artifact validates research-only synthetic process context generated from public UCI SECOM sensor data. It is not data from a real semiconductor fab.

## Validation Result

| Check | Result |
| --- | --- |
| Output-table structural contract | PASS |
| Five synthetic anomaly mechanisms | PASS |
| Benign recipe/product/tool drift controls | PASS |
| RCA ground truth and evidence references | PASS |
| Byte-for-byte regeneration | PASS |

## Dataset Summary

- Generated lots: 1567
- Synthetic anomaly lots: 235
- Benign drift lots: 72
- RCA ground-truth cases: 238

## Mechanism Counts

| Mechanism | Lots |
| --- | ---: |
| abrupt_mean_shift | 47 |
| gradual_degradation | 47 |
| variance_instability | 47 |
| sensor_fault | 47 |
| contextual_anomaly | 47 |

## Output SHA-256

| Table | SHA-256 |
| --- | --- |
| lots | `1d55897270491e523e217930f2ecb8f1fcafb299d1228a67aebe70fbbf90ca1f` |
| tool_events | `5e91b54fdaeb093ce7fbb8b92ae7ab3865562e4e27c504c7017195ff7ea06e1b` |
| maintenance | `fd85dcb5668b2a461e45d736a87abdcb511bfea379766eb9c38ace90b2c655d0` |
| process_changes | `6eb76ca09fcb52de1c3039f60fb3d1135e62c0182caf5d7b65ecc3fe4acf954b` |
| rca_ground_truth | `315c209095c0f6018eca416dc652d9c0eacf6679a631e18efc3e641f496898a5` |
