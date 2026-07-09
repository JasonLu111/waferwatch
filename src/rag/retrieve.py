from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "processed" / "rag_tfidf_index.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with", "within", "without", "only", "should", "must", "not",
    "lot", "tool", "chamber", "sensor", "model", "report", "waferwatch",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def load_index() -> Dict[str, object]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing RAG index: {INDEX_PATH}. Run: python -m src.rag.ingest_docs"
        )
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def vectorize_query(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = tokenize(query)
    if not tokens:
        return {}

    counts = Counter(tokens)
    max_count = max(counts.values())

    vector: Dict[str, float] = {}
    for term, count in counts.items():
        if term in idf:
            vector[term] = (count / max_count) * idf[term]
    return vector


def cosine_score(query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
    if not query_vector or not doc_vector:
        return 0.0

    shared_terms = set(query_vector).intersection(doc_vector)
    numerator = sum(query_vector[t] * doc_vector[t] for t in shared_terms)
    query_norm = math.sqrt(sum(v * v for v in query_vector.values()))
    doc_norm = math.sqrt(sum(v * v for v in doc_vector.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    return numerator / (query_norm * doc_norm)


def extract_filters_from_query(query: str) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    upper = query.upper()

    lot_match = re.search(r"\bLOT_\d{4}_\d{6}\b", upper)
    tool_match = re.search(r"\bT_(?:CVD|ETCH|CMP|PVD|LITHO|MET)_\d{2}\b", upper)
    chamber_match = re.search(r"\b(?:CVD|ETCH|CMP|PVD|LITHO|MET)_\d{2}\b", upper)
    recipe_match = re.search(r"\bRCP_[A-Z0-9_]+\b", upper)

    if lot_match:
        filters["lot_id"] = lot_match.group(0)
    if tool_match:
        filters["tool_id"] = tool_match.group(0)
    if chamber_match:
        filters["chamber_id"] = chamber_match.group(0)
    if recipe_match:
        filters["recipe_id"] = recipe_match.group(0)

    for stage in ["CVD", "ETCH", "CMP", "PVD", "LITHOGRAPHY", "METROLOGY"]:
        if re.search(rf"\b{stage}\b", upper):
            filters["process_stage"] = stage.title()

    return filters


def metadata_matches(document: Dict[str, object], filters: Dict[str, str]) -> bool:
    if not filters:
        return True

    if document.get("type") != "case":
        return False

    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)

    for key, expected in filters.items():
        actual = str(metadata.get(key, ""))
        if actual.upper() != expected.upper():
            return False

    return True


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.02,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    index = load_index()
    documents = index["documents"]
    idf = index["idf"]
    vectors = index["vectors"]

    assert isinstance(documents, list)
    assert isinstance(idf, dict)
    assert isinstance(vectors, dict)

    if filters is None:
        filters = extract_filters_from_query(query)

    query_vector = vectorize_query(query, idf)
    scored: List[Dict[str, object]] = []

    for document in documents:
        if not metadata_matches(document, filters):
            continue

        doc_id = str(document["id"])
        doc_vector = vectors.get(doc_id, {})
        score = cosine_score(query_vector, doc_vector)

        if score >= min_score:
            item = dict(document)
            item["score"] = round(score, 6)
            scored.append(item)

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve WaferWatch RAG evidence.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve.")
    args = parser.parse_args()

    filters = extract_filters_from_query(args.query)
    results = retrieve(args.query, top_k=args.top_k, filters=filters)

    print("Query:")
    print(args.query)

    if filters:
        print("Metadata filters:")
        for key, value in filters.items():
            print(f"- {key}: {value}")

    if not results:
        print("No retrieved evidence met the score/filter requirements.")
        return

    print("Retrieved evidence:")
    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        source = item.get("source_path", "")
        print(f"{rank}. {item['id']} | score={item['score']} | source={source}")
        if isinstance(metadata, dict):
            compact_metadata = {
                k: v for k, v in metadata.items()
                if k in {"lot_id", "tool_id", "chamber_id", "recipe_id", "process_stage", "confidence_level"}
            }
            if compact_metadata:
                print(f"   metadata={compact_metadata}")


if __name__ == "__main__":
    main()
