# SECOM Data Quality Report

## Summary

- Number of rows: 1567
- Number of sensor features: 590
- Total missing sensor values: 41951
- Features with at least one missing value: 538
- Features with 100% missing values: 0
- Features with missing rate >= 50%: 28
- Features with missing rate between 5% and 50%: 24
- Missing timestamps: 0

## Top 20 Features by Missing Rate

| Rank | Feature | Missing Count | Missing Rate |
|---:|---|---:|---:|
| 1 | `sensor_292` | 1429 | 0.9119 |
| 2 | `sensor_293` | 1429 | 0.9119 |
| 3 | `sensor_158` | 1429 | 0.9119 |
| 4 | `sensor_157` | 1429 | 0.9119 |
| 5 | `sensor_492` | 1341 | 0.8558 |
| 6 | `sensor_085` | 1341 | 0.8558 |
| 7 | `sensor_358` | 1341 | 0.8558 |
| 8 | `sensor_220` | 1341 | 0.8558 |
| 9 | `sensor_244` | 1018 | 0.6496 |
| 10 | `sensor_517` | 1018 | 0.6496 |
| 11 | `sensor_109` | 1018 | 0.6496 |
| 12 | `sensor_246` | 1018 | 0.6496 |
| 13 | `sensor_383` | 1018 | 0.6496 |
| 14 | `sensor_382` | 1018 | 0.6496 |
| 15 | `sensor_384` | 1018 | 0.6496 |
| 16 | `sensor_245` | 1018 | 0.6496 |
| 17 | `sensor_111` | 1018 | 0.6496 |
| 18 | `sensor_110` | 1018 | 0.6496 |
| 19 | `sensor_516` | 1018 | 0.6496 |
| 20 | `sensor_518` | 1018 | 0.6496 |

## Interpretation

The SECOM dataset contains substantial missing sensor values, which is expected for high-dimensional manufacturing sensor data. Phase R1 records this issue explicitly before cleaning, instead of silently imputing values.