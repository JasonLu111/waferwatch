# WaferWatch RAG Evaluation Report

Generated at: 2026-07-09T05:13:41.534736+00:00

## Summary

| Metric | Value |
|---|---:|
| Total tests | 3 |
| Passed tests | 3 |
| Failed tests | 0 |

## Test Results

| Test | Expected abstention | Actual abstention | Expected evidence | Retrieved IDs | Passed |
|---|---:|---:|---|---|---:|
| known_lot_grounded_answer | False | False | CASE_001 | CASE_001 | True |
| unknown_lot_abstention | True | True |  |  | True |
| unsupported_root_cause_claim_abstention | True | True |  | CASE_001 | True |

## Hallucination Control Rules Implemented

1. The generator only uses retrieved evidence.
2. The generator uses a fixed answer template.
3. Observed evidence is separated from suspected issue.
4. Suspected issue is never presented as a confirmed physical root cause.
5. Evidence IDs are included in every answer.
6. The assistant abstains when no metadata-matched evidence is retrieved.
7. The assistant abstains when the question asks for an unsupported root-cause claim.
8. The limitations section states that physical causality is not confirmed.

## Detailed Answers

### known_lot_grounded_answer

```text
Question:
Why was LOT_2026_000417 escalated and is chamber pressure instability confirmed?

Observed evidence:
1. High lot risk after pressure-related warning. [CASE_001]
2. risk_score=0.82. [CASE_001]
3. sensor_073 above historical control band. [CASE_001]
4. sensor_221 and sensor_488 elevated. [CASE_001]
5. CVD_03 pressure warning occurred 2.5 hours before measurement. [CASE_001]

Suspected issue:
Retrieved evidence suggests possible chamber pressure instability. This is a suspected issue only, not a confirmed physical cause. [CASE_001]

Recommended next checks:
1. Inspect CVD_03 chamber pressure trend. [CASE_001]
2. compare adjacent lots from same chamber. [CASE_001]
3. review maintenance logs from previous 72 hours. [CASE_001]
4. recheck sensor calibration status. [CASE_001]

Confidence:
Medium

Limitations:
This answer is based only on retrieved evidence. It does not confirm physical causality.

Evidence IDs:
- CASE_001
```

### unknown_lot_abstention

```text
Question:
Why was LOT_2099_999999 escalated on T_CVD_99?

Observed evidence:
1. Insufficient retrieved evidence to answer the question safely.

Suspected issue:
Insufficient evidence. The assistant cannot identify or confirm a suspected issue from the retrieved knowledge base.

Recommended next checks:
1. Refine the query with a valid lot_id, tool_id, chamber_id, recipe_id, or time window.
2. Retrieve linked sensor trends, tool alarms, maintenance records, recipe changes, and engineer review notes.
3. Ask a process or equipment engineer to review the raw evidence before making any RCA claim.

Confidence:
Low

Limitations:
This answer is based only on retrieved evidence. It does not confirm physical causality.

Evidence IDs:
- None

Abstention reason:
No retrieved evidence matched the query and metadata filters.
```

### unsupported_root_cause_claim_abstention

```text
Question:
Is tungsten contamination the confirmed root cause for LOT_2026_000417?

Observed evidence:
1. Insufficient retrieved evidence to answer the question safely.

Suspected issue:
Insufficient evidence. The assistant cannot identify or confirm a suspected issue from the retrieved knowledge base.

Recommended next checks:
1. Refine the query with a valid lot_id, tool_id, chamber_id, recipe_id, or time window.
2. Retrieve linked sensor trends, tool alarms, maintenance records, recipe changes, and engineer review notes.
3. Ask a process or equipment engineer to review the raw evidence before making any RCA claim.

Confidence:
Low

Limitations:
This answer is based only on retrieved evidence. It does not confirm physical causality.

Evidence IDs:
- CASE_001

Abstention reason:
Retrieved evidence does not support the requested claim: tungsten contamination.
```
