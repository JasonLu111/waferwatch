# WaferWatch Alert Policy

## Purpose

This document defines the local prototype policy for lot escalation, RCA evidence retrieval, and hallucination control.

WaferWatch is a decision-support prototype. It does not confirm physical root causes. It helps engineers retrieve evidence and decide what to check next.

## Alert Escalation Policy

A lot may be escalated when one or more of the following conditions are present:

1. High model risk score relative to the selected operating threshold.
2. Multiple SPC violations in related sensor groups.
3. Recent high-severity tool alarms.
4. Preventive maintenance delay or relevant maintenance note.
5. Recipe change within the previous 24 to 72 hours.
6. Similar historical RCA case with matching lot, tool, chamber, sensor, or process-stage evidence.

## Evidence Types Allowed in RAG Answers

The RAG assistant may only use retrieved evidence from:

1. Synthetic RCA case library.
2. Model reports.
3. Data card.
4. Model card.
5. Alert policy.
6. Overfitting analysis report.
7. Model comparison report.
8. SECOM data quality report.

## Hallucination Control Rules

The RAG assistant must follow these rules:

1. Do not invent a physical root cause.
2. Do not use external knowledge that was not retrieved.
3. Do not convert a suspected issue into a confirmed root cause.
4. Separate observed evidence from suspected issue.
5. Every evidence-based claim must include an evidence ID.
6. If retrieved evidence is weak or missing, abstain.
7. If the user asks for an unsupported root-cause claim, abstain.
8. When a lot_id, tool_id, chamber_id, recipe_id, or process_stage appears in the query, retrieval should apply metadata filtering where possible.
9. Always include limitations.

## Required RAG Answer Format

Question:
...

Observed evidence:
1. ...

Suspected issue:
...

Recommended next checks:
1. ...

Confidence:
Low / Medium / High

Limitations:
This answer is based only on retrieved evidence. It does not confirm physical causality.

Evidence IDs:
- CASE_001
- DOC_001

## Confidence Rules

High confidence:
- Multiple retrieved evidence items agree.
- Metadata matches the query.
- Evidence includes specific lot, tool, chamber, and event records.
- Historical case confidence is high.

Medium confidence:
- At least one relevant case matches the query.
- Evidence suggests a likely investigation direction.
- Physical causality is still not confirmed.

Low confidence:
- Evidence is incomplete.
- Only generic documents were retrieved.
- Metadata does not match strongly.
- The assistant must abstain or recommend additional checks.
