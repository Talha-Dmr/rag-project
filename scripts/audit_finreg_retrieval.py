#!/usr/bin/env python3
"""Audit FinReg retrieval quality without running generation or detector logic."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


DEFAULT_QUESTIONS = Path("benchmarks/finreg/full_rag_questions.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/finreg_retrieval_audit")

FAMILY_PATTERNS = {
    "bcbs": re.compile(r"\b(?:bcbs|basel|bis)\b", re.IGNORECASE),
    "eba": re.compile(r"\b(?:eba|european banking authority)\b", re.IGNORECASE),
    "ecb": re.compile(r"\b(?:ecb|european central bank|ssm)\b", re.IGNORECASE),
    "pra_boe": re.compile(r"\b(?:pra|boe|bank of england|prudential regulation authority)\b", re.IGNORECASE),
    "fed_occ": re.compile(r"\b(?:sr\s*11-7|federal reserve|occ)\b", re.IGNORECASE),
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "before", "being",
    "between", "both", "could", "does", "from", "have", "into", "main",
    "more", "must", "only", "other", "over", "risk", "should", "than",
    "that", "their", "there", "these", "this", "through", "under", "when",
    "where", "which", "while", "with", "within", "would",
}

MID_SENTENCE_PREFIXES = {
    "and", "are", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "that", "the", "their", "these", "to", "using", "with",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def phrase_tokens(phrase: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", phrase.lower())
    return [token for token in tokens if len(token) >= 4 and token not in STOPWORDS]


def text_contains_concept(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if normalized_phrase and normalized_phrase in normalized_text:
        return True

    tokens = phrase_tokens(phrase)
    if not tokens:
        return False

    token_hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", normalized_text))
    required = max(1, int(round(len(tokens) * 0.6)))
    return token_hits >= required


def concept_hits(text: str, concepts: list[str]) -> list[str]:
    return [concept for concept in concepts if text_contains_concept(text, concept)]


def infer_expected_families(row: dict[str, Any]) -> list[str]:
    explicit = as_list(row.get("expected_source_families"))
    if explicit:
        return explicit

    text = " ".join(
        str(row.get(key, ""))
        for key in ("topic", "query", "manual_focus")
    )
    families = [family for family, pattern in FAMILY_PATTERNS.items() if pattern.search(text)]
    return families


def looks_mid_sentence_start(text: str) -> bool:
    cleaned = (text or "").lstrip()
    if not cleaned:
        return False
    first_char = cleaned[0]
    if first_char in ",.;:)]}":
        return True
    first_word_match = re.match(r"[A-Za-z]+", cleaned)
    first_word = first_word_match.group(0).lower() if first_word_match else ""
    if first_word in MID_SENTENCE_PREFIXES:
        return True
    return first_char.isalpha() and first_char.islower()


def source_key(metadata: dict[str, Any]) -> str:
    for key in ("id", "source_id", "raw_relpath", "title", "source_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def source_family(metadata: dict[str, Any]) -> str:
    return str(metadata.get("family") or metadata.get("source_org") or "unknown").lower()


def audit_question(rag: RAGPipeline, row: dict[str, Any], k: int) -> dict[str, Any]:
    retrieved_docs = rag.retriever.retrieve(row["query"], k=k)
    if rag.reranker and retrieved_docs:
        top_k = rag.reranker_top_k or k
        retrieved_docs = rag.reranker.rerank(row["query"], retrieved_docs, top_k=top_k)

    context_text = "\n\n".join(doc.get("content", "") for doc in retrieved_docs)
    expected_points = as_list(row.get("expected_answer_points"))
    forbidden_claims = as_list(row.get("forbidden_claims"))
    expected_hits = concept_hits(context_text, expected_points)
    forbidden_hits = concept_hits(context_text, forbidden_claims)

    expected_families = infer_expected_families(row)
    retrieved_families = [source_family(doc.get("metadata") or {}) for doc in retrieved_docs]
    retrieved_source_ids = [source_key(doc.get("metadata") or {}) for doc in retrieved_docs]
    expected_family_hit = (
        any(family in retrieved_families for family in expected_families)
        if expected_families else None
    )
    bad_start_count = sum(1 for doc in retrieved_docs if looks_mid_sentence_start(doc.get("content", "")))
    scores = [float(doc.get("score", 0.0)) for doc in retrieved_docs]

    top_contexts = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.get("metadata") or {}
        top_contexts.append({
            "rank": rank,
            "score": doc.get("score"),
            "family": source_family(metadata),
            "source_id": source_key(metadata),
            "title": metadata.get("title"),
            "chunk_index": metadata.get("chunk_index"),
            "source_url": metadata.get("source_url") or metadata.get("download_url"),
            "starts_mid_sentence": looks_mid_sentence_start(doc.get("content", "")),
            "preview": " ".join(str(doc.get("content", "")).split())[:500],
        })

    return {
        **row,
        "retrieved_count": len(retrieved_docs),
        "top_score": scores[0] if scores else None,
        "mean_score": mean(scores),
        "expected_source_families": expected_families,
        "retrieved_families": sorted(set(retrieved_families)),
        "expected_source_family_hit": expected_family_hit,
        "unique_source_count": len(set(retrieved_source_ids)),
        "unique_family_count": len(set(retrieved_families)),
        "bad_chunk_start_count": bad_start_count,
        "bad_chunk_start_rate": (bad_start_count / len(retrieved_docs)) if retrieved_docs else 0.0,
        "expected_point_retrieval_hits": expected_hits,
        "expected_point_retrieval_coverage": (
            len(expected_hits) / len(expected_points) if expected_points else None
        ),
        "forbidden_retrieval_hits": forbidden_hits,
        "forbidden_retrieval_hit_count": len(forbidden_hits),
        "top_contexts": top_contexts,
    }


def summarize(rows: list[dict[str, Any]], collection_count: int, k: int, config_name: str) -> dict[str, Any]:
    family_labeled = [
        row for row in rows
        if isinstance(row.get("expected_source_family_hit"), bool)
    ]
    coverage_values = [
        float(row["expected_point_retrieval_coverage"])
        for row in rows
        if isinstance(row.get("expected_point_retrieval_coverage"), (int, float))
    ]
    top_scores = [float(row["top_score"]) for row in rows if isinstance(row.get("top_score"), (int, float))]
    mean_scores = [float(row["mean_score"]) for row in rows if isinstance(row.get("mean_score"), (int, float))]
    bad_start_rates = [float(row.get("bad_chunk_start_rate", 0.0)) for row in rows]
    unique_sources = [float(row.get("unique_source_count", 0)) for row in rows]
    forbidden_rows = [row for row in rows if int(row.get("forbidden_retrieval_hit_count") or 0) > 0]
    point_hit_rows = [
        row for row in rows
        if float(row.get("expected_point_retrieval_coverage") or 0.0) > 0.0
    ]
    full_point_rows = [
        row for row in rows
        if float(row.get("expected_point_retrieval_coverage") or 0.0) >= 0.999
    ]
    topic_counts = Counter(str(row.get("topic", "unknown")) for row in rows)

    return {
        "config": config_name,
        "total_questions": len(rows),
        "collection_count": collection_count,
        "retrieval_k": k,
        "mean_top_score": mean(top_scores),
        "mean_retrieved_score": mean(mean_scores),
        "mean_expected_point_retrieval_coverage": mean(coverage_values),
        "any_expected_point_retrieval_hit_rate": len(point_hit_rows) / len(rows) if rows else 0.0,
        "full_expected_point_retrieval_hit_rate": len(full_point_rows) / len(rows) if rows else 0.0,
        "forbidden_retrieval_hit_rate": len(forbidden_rows) / len(rows) if rows else 0.0,
        "source_family_labeled_count": len(family_labeled),
        "source_family_hit_rate": (
            sum(1 for row in family_labeled if row.get("expected_source_family_hit")) / len(family_labeled)
            if family_labeled else None
        ),
        "mean_unique_source_count": mean(unique_sources),
        "mean_bad_chunk_start_rate": mean(bad_start_rates),
        "topic_counts": dict(topic_counts),
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], questions_path: Path) -> None:
    worst = sorted(
        rows,
        key=lambda row: (
            float(row.get("expected_point_retrieval_coverage") or 0.0),
            -int(row.get("forbidden_retrieval_hit_count") or 0),
        ),
    )[:10]
    lines = [
        "# FinReg Retrieval Audit",
        "",
        f"- config: `{summary['config']}`",
        f"- questions: `{questions_path}`",
        f"- collection count: `{summary['collection_count']}`",
        f"- retrieval k: `{summary['retrieval_k']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean top score | {summary['mean_top_score']} |",
        f"| Mean retrieved score | {summary['mean_retrieved_score']} |",
        f"| Mean expected point retrieval coverage | {summary['mean_expected_point_retrieval_coverage']} |",
        f"| Any expected point retrieval hit rate | {summary['any_expected_point_retrieval_hit_rate']:.3f} |",
        f"| Full expected point retrieval hit rate | {summary['full_expected_point_retrieval_hit_rate']:.3f} |",
        f"| Forbidden retrieval hit rate | {summary['forbidden_retrieval_hit_rate']:.3f} |",
        f"| Source family hit rate | {summary['source_family_hit_rate']} |",
        f"| Mean unique source count | {summary['mean_unique_source_count']} |",
        f"| Mean bad chunk start rate | {summary['mean_bad_chunk_start_rate']} |",
        "",
        "## Lowest Coverage Questions",
        "",
        "| ID | Topic | Coverage | Forbidden hits | Top source | Top preview |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in worst:
        top = (row.get("top_contexts") or [{}])[0]
        preview = str(top.get("preview", "")).replace("|", "\\|")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        top_source = f"{top.get('family', '')}/{top.get('source_id', '')}".replace("|", "\\|")
        lines.append(
            f"| {row.get('id')} | {row.get('topic')} | "
            f"{row.get('expected_point_retrieval_coverage')} | "
            f"{row.get('forbidden_retrieval_hit_count')} | {top_source} | {preview} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--disable-reranker", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["llm"] = {"type": "none"}
    config.setdefault("hallucination_detector", {})["enabled"] = False
    if args.disable_reranker:
        config["reranker"] = {}

    rag = RAGPipeline.from_config(config)
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for item in questions:
        row = audit_question(rag, item, args.k)
        rows.append(row)
        print(
            f"{item.get('id')} coverage={row.get('expected_point_retrieval_coverage')} "
            f"families={','.join(row.get('retrieved_families') or [])} "
            f"bad_start={row.get('bad_chunk_start_rate'):.2f}"
        )

    collection_count = rag.vector_store.get_count()
    summary = summarize(rows, collection_count, args.k, args.config)
    run_name = safe_name(args.run_name or f"retrieval_{args.config}_k{args.k}")
    out_dir = args.output_dir / run_name
    write_jsonl(out_dir / "per_question.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "report.md", summary, rows, args.questions)

    print(f"\nWrote: {out_dir}")
    print(f"Mean expected point retrieval coverage: {summary['mean_expected_point_retrieval_coverage']}")
    print(f"Any expected point retrieval hit rate: {summary['any_expected_point_retrieval_hit_rate']:.3f}")
    print(f"Mean bad chunk start rate: {summary['mean_bad_chunk_start_rate']}")


if __name__ == "__main__":
    main()
