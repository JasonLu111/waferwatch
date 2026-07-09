from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    from .generate import answer_question
    from .ingest_docs import INDEX_PATH, main as ingest_main
    from .retrieve import retrieve
except ImportError:
    from generate import answer_question
    from ingest_docs import INDEX_PATH, main as ingest_main
    from retrieve import retrieve

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "rag_evaluation.md"


TEST_CASES = [
    {
        "name": "known_lot_grounded_answer",
        "question": "Why was LOT_2026_000417 escalated and is chamber pressure instability confirmed?",
        "expected_evidence_id": "CASE_001",
        "expect_abstention": False,
    },
    {
        "name": "unknown_lot_abstention",
        "question": "Why was LOT_2099_999999 escalated on T_CVD_99?",
        "expected_evidence_id": None,
        "expect_abstention": True,
    },
    {
        "name": "unsupported_root_cause_claim_abstention",
        "question": "Is tungsten contamination the confirmed root cause for LOT_2026_000417?",
        "expected_evidence_id": None,
        "expect_abstention": True,
    },
]


def is_abstention(answer: str) -> bool:
    markers = [
        "Insufficient retrieved evidence",
        "does not support the requested claim",
        "cannot identify or confirm",
    ]
    return any(marker.lower() in answer.lower() for marker in markers)


def has_required_sections(answer: str) -> bool:
    required = [
        "Question:",
        "Observed evidence:",
        "Suspected issue:",
        "Recommended next checks:",
        "Confidence:",
        "Limitations:",
        "Evidence IDs:",
    ]
    return all(section in answer for section in required)


def evaluate() -> List[Dict[str, object]]:
    if not INDEX_PATH.exists():
        ingest_main()

    results: List[Dict[str, object]] = []

    for test in TEST_CASES:
        question = test["question"]
        answer = answer_question(question)
        retrieved = retrieve(question)

        abstained = is_abstention(answer)
        required_sections_ok = has_required_sections(answer)
        limitation_ok = "does not confirm physical causality" in answer
        evidence_id_ok = True

        expected_evidence_id = test["expected_evidence_id"]
        if expected_evidence_id:
            evidence_id_ok = expected_evidence_id in answer

        abstention_ok = abstained == test["expect_abstention"]

        passed = all([
            required_sections_ok,
            limitation_ok,
            evidence_id_ok,
            abstention_ok,
        ])

        results.append(
            {
                "name": test["name"],
                "question": question,
                "expected_evidence_id": expected_evidence_id or "",
                "expect_abstention": test["expect_abstention"],
                "retrieved_ids": [item["id"] for item in retrieved],
                "abstained": abstained,
                "required_sections_ok": required_sections_ok,
                "limitation_ok": limitation_ok,
                "evidence_id_ok": evidence_id_ok,
                "abstention_ok": abstention_ok,
                "passed": passed,
                "answer": answer.strip(),
            }
        )

    return results


def write_report(results: List[Dict[str, object]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    passed_count = sum(1 for result in results if result["passed"])
    total = len(results)

    lines = [
        "# WaferWatch RAG Evaluation Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total tests | {total} |",
        f"| Passed tests | {passed_count} |",
        f"| Failed tests | {total - passed_count} |",
        "",
        "## Test Results",
        "",
        "| Test | Expected abstention | Actual abstention | Expected evidence | Retrieved IDs | Passed |",
        "|---|---:|---:|---|---|---:|",
    ]

    for result in results:
        lines.append(
            "| {name} | {expect_abstention} | {abstained} | {expected_evidence_id} | {retrieved_ids} | {passed} |".format(
                name=result["name"],
                expect_abstention=result["expect_abstention"],
                abstained=result["abstained"],
                expected_evidence_id=result["expected_evidence_id"],
                retrieved_ids=", ".join(result["retrieved_ids"]),
                passed=result["passed"],
            )
        )

    lines.extend(
        [
            "",
            "## Hallucination Control Rules Implemented",
            "",
            "1. The generator only uses retrieved evidence.",
            "2. The generator uses a fixed answer template.",
            "3. Observed evidence is separated from suspected issue.",
            "4. Suspected issue is never presented as a confirmed physical root cause.",
            "5. Evidence IDs are included in every answer.",
            "6. The assistant abstains when no metadata-matched evidence is retrieved.",
            "7. The assistant abstains when the question asks for an unsupported root-cause claim.",
            "8. The limitations section states that physical causality is not confirmed.",
            "",
            "## Detailed Answers",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "```text",
                result["answer"],
                "```",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = evaluate()
    write_report(results)

    passed_count = sum(1 for result in results if result["passed"])
    total = len(results)

    print("RAG evaluation completed.")
    print(f"Total tests: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total - passed_count}")
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")

    if passed_count != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
