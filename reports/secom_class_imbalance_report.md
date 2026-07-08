# SECOM Class Imbalance Report

## Label Meaning

- `-1`: normal / pass
- `1`: abnormal / fail

## Class Distribution

| Label | Meaning | Count | Rate |
|---:|---|---:|---:|
| -1 | Pass | 1463 | 0.9336 |
| 1 | Fail | 104 | 0.0664 |

## Summary

- Total samples: 1567
- Pass samples: 1463
- Fail samples: 104
- Fail rate: 0.0664
- Pass-to-fail imbalance ratio: 14.07:1

## Interpretation

The SECOM dataset is highly imbalanced. This means accuracy alone can be misleading. Later modeling stages should emphasize recall, precision, PR-AUC, confusion matrix, and false alarm burden instead of relying only on accuracy.