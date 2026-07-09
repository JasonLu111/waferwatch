from __future__ import annotations

import argparse
import re
from typing import Dict, Iterable, List

try:
    from .retrieve import retrieve
except ImportError:
    from retrieve import retrieve

LIMITATION_TEXT = "This answer is based only on retrieved evidence. It does not confirm physical causality."

UNSUPPORTED_CLAIM_TERMS = [
    "tungsten contamination",
    "oxygen leak",
    "gas leak",
    "operator error",
    "chemical contamination",
    "particle contamination",
    "pump failure",
    "confirmed root cause",
]


def split_semicolon_text(text: str) -> List[str]:
    parts = re.split(r";|\|", text)
    return [p.strip().rstrip(".") for p in parts if p.strip()]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = item.lower()
        if normalized not in seen:
            seen.add(normalized)
            output.append(item)
    return output


def evidence_ids(retrieved: List[Dict[str, object]]) -> List[str]:
    return [str(item["id"]) for item in retrieved]


def unsupported_requested_claim(question: str, retrieved: List[Dict[str, object]]) -> str | None:
    q = question.lower()
    combined_text = " ".join(str(item.get("text", "")) for item in retrieved).lower()

    for term in UNSUPPORTED_CLAIM_TERMS:
        if term in q and term not in combined_text:
            return term

    return None


def abstain_answer(question: str, reason: str, retrieved: List[Dict[str, object]] | None = None) -> str:
    ids = evidence_ids(retrieved or [])
    id_lines = "\n".join(f"- {item_id}" for item_id in ids) if ids else "- None"

    return f"""Question:
{question}

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
{LIMITATION_TEXT}

Evidence IDs:
{id_lines}

Abstention reason:
{reason}
"""


def build_observed_evidence(retrieved: List[Dict[str, object]], max_items: int = 5) -> List[str]:
    evidence_items: List[str] = []

    for item in retrieved:
        item_id = str(item["id"])
        fields = item.get("fields", {})

        if isinstance(fields, dict) and fields:
            symptom = fields.get("symptom", "").strip()
            evidence = fields.get("evidence", "").strip()
            spc = fields.get("spc_violations", "").strip()

            if symptom:
                evidence_items.append(f"{symptom}. [{item_id}]")

            for part in split_semicolon_text(evidence):
                evidence_items.append(f"{part}. [{item_id}]")

            if spc:
                evidence_items.append(f"{spc}. [{item_id}]")
        else:
            text = str(item.get("text", "")).replace("\n", " ")
            snippet = text[:220].strip()
            if snippet:
                evidence_items.append(f"{snippet}. [{item_id}]")

    return unique_preserve_order(evidence_items)[:max_items]


def build_suspected_issue(retrieved: List[Dict[str, object]]) -> str:
    causes: List[str] = []
    cause_ids: List[str] = []

    for item in retrieved:
        fields = item.get("fields", {})
        if isinstance(fields, dict):
            cause = fields.get("suspected_cause", "").strip()
            if cause and "insufficient evidence" not in cause.lower():
                causes.append(cause)
                cause_ids.append(str(item["id"]))

    causes = unique_preserve_order(causes)

    if not causes:
        return "The retrieved evidence does not support a specific suspected issue. Treat this as an investigation cue only."

    joined_causes = "; ".join(causes[:2])
    joined_ids = ", ".join(unique_preserve_order(cause_ids))
    return (
        f"Retrieved evidence suggests {joined_causes}. "
        f"This is a suspected issue only, not a confirmed physical cause. [{joined_ids}]"
    )


def build_next_checks(retrieved: List[Dict[str, object]], max_items: int = 5) -> List[str]:
    checks: List[str] = []

    for item in retrieved:
        item_id = str(item["id"])
        fields = item.get("fields", {})
        if isinstance(fields, dict):
            action = fields.get("corrective_action", "").strip()
            for part in split_semicolon_text(action):
                checks.append(f"{part}. [{item_id}]")

    if not checks:
        checks = [
            "Review lot-level sensor trend and SPC violations. [DOC_POLICY]",
            "Check linked tool alarms and maintenance records. [DOC_POLICY]",
            "Compare adjacent lots processed on the same tool or chamber. [DOC_POLICY]",
        ]

    return unique_preserve_order(checks)[:max_items]


def infer_confidence(retrieved: List[Dict[str, object]]) -> str:
    case_confidences = []

    for item in retrieved:
        fields = item.get("fields", {})
        if isinstance(fields, dict):
            confidence = fields.get("confidence_level", "").strip().lower()
            if confidence:
                case_confidences.append(confidence)

    if "high" in case_confidences and len(retrieved) >= 1:
        return "High"
    if "medium" in case_confidences:
        return "Medium"
    return "Low"


def answer_question(question: str, top_k: int = 5) -> str:
    retrieved = retrieve(question, top_k=top_k)

    if not retrieved:
        return abstain_answer(
            question,
            "No retrieved evidence matched the query and metadata filters.",
            retrieved,
        )

    unsupported_claim = unsupported_requested_claim(question, retrieved)
    if unsupported_claim:
        return abstain_answer(
            question,
            f"Retrieved evidence does not support the requested claim: {unsupported_claim}.",
            retrieved,
        )

    observed = build_observed_evidence(retrieved)
    if not observed:
        return abstain_answer(
            question,
            "Retrieved documents did not contain usable evidence statements.",
            retrieved,
        )

    suspected = build_suspected_issue(retrieved)
    checks = build_next_checks(retrieved)
    confidence = infer_confidence(retrieved)
    ids = evidence_ids(retrieved)

    observed_lines = "\n".join(f"{i}. {text}" for i, text in enumerate(observed, start=1))
    check_lines = "\n".join(f"{i}. {text}" for i, text in enumerate(checks, start=1))
    id_lines = "\n".join(f"- {item_id}" for item_id in ids)

    return f"""Question:
{question}

Observed evidence:
{observed_lines}

Suspected issue:
{suspected}

Recommended next checks:
{check_lines}

Confidence:
{confidence}

Limitations:
{LIMITATION_TEXT}

Evidence IDs:
{id_lines}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded WaferWatch RAG answer.")
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved evidence items.")
    args = parser.parse_args()

    print(answer_question(args.query, top_k=args.top_k))


if __name__ == "__main__":
    main()
