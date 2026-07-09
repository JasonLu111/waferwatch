from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

CASE_LIBRARY_PATH = ROOT / "data" / "synthetic" / "rca_case_library.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
KNOWLEDGE_BASE_PATH = OUTPUT_DIR / "rag_knowledge_base.jsonl"
INDEX_PATH = OUTPUT_DIR / "rag_tfidf_index.json"

SOURCE_DOC_PATHS = [
    ROOT / "reports" / "secom_data_quality_report.md",
    ROOT / "reports" / "model_card.md",
    ROOT / "reports" / "data_card.md",
    ROOT / "reports" / "model_comparison_report.md",
    ROOT / "reports" / "model_family_comparison_report.md",
    ROOT / "reports" / "overfitting_analysis_report.md",
    ROOT / "docs" / "alert_policy.md",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with", "within", "without", "only", "should", "must", "not",
    "lot", "tool", "chamber", "sensor", "model", "report", "waferwatch",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = (current + "\n\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def make_case_text(row: Dict[str, str]) -> str:
    fields = [
        f"case_id: {row.get('case_id', '')}",
        f"case_time: {row.get('case_time', '')}",
        f"lot_id: {row.get('lot_id', '')}",
        f"tool_id: {row.get('tool_id', '')}",
        f"chamber_id: {row.get('chamber_id', '')}",
        f"recipe_id: {row.get('recipe_id', '')}",
        f"process_stage: {row.get('process_stage', '')}",
        f"symptom: {row.get('symptom', '')}",
        f"evidence: {row.get('evidence', '')}",
        f"suspected_cause: {row.get('suspected_cause', '')}",
        f"corrective_action: {row.get('corrective_action', '')}",
        f"result: {row.get('result', '')}",
        f"confidence_level: {row.get('confidence_level', '')}",
        f"review_decision: {row.get('review_decision', '')}",
        f"spc_violations: {row.get('spc_violations', '')}",
        f"top_sensors: {row.get('top_sensors', '')}",
    ]
    return "\n".join(fields)


def load_case_documents() -> List[Dict[str, object]]:
    if not CASE_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Missing RCA case library: {CASE_LIBRARY_PATH}")

    documents: List[Dict[str, object]] = []

    with CASE_LIBRARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"].strip()
            documents.append(
                {
                    "id": case_id,
                    "type": "case",
                    "source_path": str(CASE_LIBRARY_PATH.relative_to(ROOT)),
                    "title": f"RCA case {case_id}",
                    "text": make_case_text(row),
                    "metadata": {
                        "case_id": case_id,
                        "case_time": row.get("case_time", "").strip(),
                        "lot_id": row.get("lot_id", "").strip(),
                        "tool_id": row.get("tool_id", "").strip(),
                        "chamber_id": row.get("chamber_id", "").strip(),
                        "recipe_id": row.get("recipe_id", "").strip(),
                        "process_stage": row.get("process_stage", "").strip(),
                        "confidence_level": row.get("confidence_level", "").strip(),
                        "review_decision": row.get("review_decision", "").strip(),
                    },
                    "fields": row,
                }
            )

    return documents


def load_markdown_documents() -> tuple[List[Dict[str, object]], List[str]]:
    documents: List[Dict[str, object]] = []
    missing: List[str] = []
    doc_counter = 1

    for path in SOURCE_DOC_PATHS:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue

        text = read_text_file(path)
        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks, start=1):
            doc_id = f"DOC_{doc_counter:03d}"
            doc_counter += 1
            documents.append(
                {
                    "id": doc_id,
                    "type": "document",
                    "source_path": str(path.relative_to(ROOT)),
                    "title": path.name,
                    "text": chunk,
                    "metadata": {
                        "chunk_id": chunk_index,
                        "source_file": path.name,
                    },
                    "fields": {},
                }
            )

    return documents, missing


def build_idf(documents: Iterable[Dict[str, object]]) -> Dict[str, float]:
    doc_tokens = []
    for doc in documents:
        tokens = set(tokenize(str(doc["text"])))
        doc_tokens.append(tokens)

    doc_count = len(doc_tokens)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(tokens)

    return {
        term: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for term, freq in sorted(df.items())
    }


def vectorize(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}

    counts = Counter(tokens)
    max_count = max(counts.values())

    vector = {}
    for term, count in counts.items():
        if term in idf:
            tf = count / max_count
            vector[term] = tf * idf[term]
    return vector


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_docs = load_case_documents()
    markdown_docs, missing_docs = load_markdown_documents()
    documents = case_docs + markdown_docs

    if not documents:
        raise RuntimeError("No documents were loaded for the RAG knowledge base.")

    idf = build_idf(documents)
    vectors = {doc["id"]: vectorize(str(doc["text"]), idf) for doc in documents}

    with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "idf": idf,
        "vectors": vectors,
    }

    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print("RAG ingestion completed.")
    print(f"Documents indexed: {len(documents)}")
    print(f"RCA cases indexed: {len(case_docs)}")
    print(f"Markdown chunks indexed: {len(markdown_docs)}")

    if missing_docs:
        print("Missing source files skipped:")
        for missing in missing_docs:
            print(f"- {missing}")

    print("Outputs:")
    print(f"- {KNOWLEDGE_BASE_PATH.relative_to(ROOT)}")
    print(f"- {INDEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
