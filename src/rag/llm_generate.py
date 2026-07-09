from __future__ import annotations

import argparse
import importlib
import os
from typing import Dict, List

OpenAI = None
try:
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI", None)
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore  # noqa: F401

try:
    from .generate import (
        LIMITATION_TEXT,
        abstain_answer,
        answer_question as local_answer_question,
        unsupported_requested_claim,
    )
    from .retrieve import retrieve
except ImportError:
    from generate import (
        LIMITATION_TEXT,
        abstain_answer,
        answer_question as local_answer_question,
        unsupported_requested_claim,
    )
    from retrieve import retrieve


REQUIRED_SECTIONS = [
    "Question:",
    "Observed evidence:",
    "Suspected issue:",
    "Recommended next checks:",
    "Confidence:",
    "Limitations:",
    "Evidence IDs:",
]


SYSTEM_INSTRUCTIONS = f"""
You are the WaferWatch evidence-grounded RCA triage assistant.

You must follow these rules:

1. Use only the retrieved evidence provided in the prompt.
2. Do not use external knowledge.
3. Do not invent a physical root cause.
4. Do not say a root cause is confirmed.
5. Distinguish observed evidence from suspected issue.
6. Every claim must be traceable to an evidence ID such as CASE_001 or DOC_001.
7. If evidence is insufficient, abstain clearly.
8. If the user asks for a confirmed root cause, explain that the evidence does not confirm physical causality.
9. Keep the exact required answer format.
10. The Limitations section must contain this exact sentence:
{LIMITATION_TEXT}
""".strip()


def format_retrieved_context(retrieved: List[Dict[str, object]]) -> str:
    blocks: List[str] = []

    for item in retrieved:
        item_id = str(item.get("id", "UNKNOWN"))
        source = str(item.get("source_path", ""))
        text = str(item.get("text", "")).strip()
        metadata = item.get("metadata", {})

        blocks.append(
            "\n".join(
                [
                    f"Evidence ID: {item_id}",
                    f"Source: {source}",
                    f"Metadata: {metadata}",
                    "Evidence text:",
                    text,
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, retrieved: List[Dict[str, object]]) -> str:
    context = format_retrieved_context(retrieved)

    return f"""
Question:
{question}

Retrieved evidence:
{context}

Write the answer using exactly this format:

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
{LIMITATION_TEXT}

Evidence IDs:
- CASE_001
""".strip()


def validate_llm_answer(answer: str, retrieved: List[Dict[str, object]]) -> tuple[bool, str]:
    for section in REQUIRED_SECTIONS:
        if section not in answer:
            return False, f"Missing required section: {section}"

    if LIMITATION_TEXT not in answer:
        return False, "Missing required limitation sentence."

    retrieved_ids = {str(item.get("id")) for item in retrieved}

    if not any(evidence_id in answer for evidence_id in retrieved_ids):
        return False, "No retrieved evidence ID was cited."

    forbidden_phrases = [
        "confirmed root cause",
        "the root cause is",
        "definitely caused by",
        "proves that",
    ]

    lowered = answer.lower()
    for phrase in forbidden_phrases:
        if phrase in lowered:
            return False, f"Unsafe causal language detected: {phrase}"

    return True, "OK"


def call_openai_responses(client: OpenAI, question: str, retrieved: List[Dict[str, object]], model: str) -> str:
    prompt = build_user_prompt(question, retrieved)

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
        max_output_tokens=900,
    )

    output_text = getattr(response, "output_text", "")

    if not output_text:
        raise RuntimeError("OpenAI Responses API did not contain output_text.")

    return output_text.strip()


def call_openai_chat_completions(client: OpenAI, question: str, retrieved: List[Dict[str, object]], model: str) -> str:
    prompt = build_user_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    output_text = response.choices[0].message.content

    if not output_text:
        raise RuntimeError("OpenAI Chat Completions API did not contain message content.")

    return output_text.strip()


def call_openai(question: str, retrieved: List[Dict[str, object]], model: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: python -m pip install --upgrade openai")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()

    errors: List[str] = []

    try:
        return call_openai_responses(client, question, retrieved, model)
    except Exception as exc:
        errors.append(f"Responses API failed with {type(exc).__name__}: {exc}")

    try:
        return call_openai_chat_completions(client, question, retrieved, model)
    except Exception as exc:
        errors.append(f"Chat Completions API failed with {type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors))


def answer_question_with_openai(question: str, top_k: int = 5, model: str | None = None) -> str:
    selected_model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-3.5-turbo")
    retrieved = retrieve(question, top_k=top_k)

    if not retrieved:
        return abstain_answer(
            question,
            "No retrieved evidence matched the query and metadata filters. OpenAI API was not called.",
            retrieved,
        )

    unsupported_claim = unsupported_requested_claim(question, retrieved)
    if unsupported_claim:
        return abstain_answer(
            question,
            f"Retrieved evidence does not support the requested claim: {unsupported_claim}. OpenAI API was not called.",
            retrieved,
        )

    try:
        llm_answer = call_openai(question, retrieved, selected_model)
    except Exception as exc:
        fallback = local_answer_question(question, top_k=top_k)

        error_details = str(exc)
        response_obj = getattr(exc, "response", None)
        if response_obj is not None:
            try:
                error_details = response_obj.text
            except Exception:
                error_details = str(exc)

        return (
            fallback
            + "\nOpenAI API fallback note:\n"
            + f"- OpenAI generation failed, so the deterministic local answer was used.\n"
            + f"- Error type: {type(exc).__name__}\n"
            + f"- Error message: {error_details}\n"
        )

    is_valid, reason = validate_llm_answer(llm_answer, retrieved)

    if not is_valid:
        fallback = local_answer_question(question, top_k=top_k)
        return (
            fallback
            + "\nOpenAI API fallback note:\n"
            + "- OpenAI generation was rejected by post-generation validation.\n"
            + f"- Validation reason: {reason}\n"
        )

    return llm_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a WaferWatch RAG answer using OpenAI API.")
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved evidence items.")
    parser.add_argument("--model", default=None, help="OpenAI model name. Defaults to OPENAI_RAG_MODEL or gpt-5.5.")
    args = parser.parse_args()

    print(answer_question_with_openai(args.query, top_k=args.top_k, model=args.model))


if __name__ == "__main__":
    main()