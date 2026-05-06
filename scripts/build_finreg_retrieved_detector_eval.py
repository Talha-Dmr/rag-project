#!/usr/bin/env python3
"""Build a retrieved-context FinReg detector eval set.

The output is intentionally detector-focused, not answer-generation-focused:
it uses the real FinReg vector index and reranker to collect evidence spans,
then creates controlled supported / unsupported / contradicted claims over
those spans. This avoids conflating detector quality with the LLM's abstain
behavior.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_loader import load_config
import src.embeddings  # noqa: F401  # register embedders
import src.reranking  # noqa: F401  # register rerankers
import src.vector_stores  # noqa: F401  # register vector stores
from src.embeddings.base_embedder import EmbedderFactory
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.reranking.base_reranker import RerankerFactory
from src.vector_stores.base_store import VectorStoreFactory


LABELS = ("supported", "unsupported", "contradicted")

BAD_PATTERNS = (
    "copyright",
    "cookies",
    "privacy",
    "sitemap",
    "back to top",
    "http://",
    "https://",
    "www.",
    "submit your comments",
    "summary of responses",
)

SOURCE_FAMILY_PATTERNS = {
    "BCBS": re.compile(r"\b(bcbs|basel)\b", re.IGNORECASE),
    "EBA": re.compile(r"\beba\b|european banking authority", re.IGNORECASE),
    "ECB": re.compile(r"\becb\b|european central bank", re.IGNORECASE),
    "PRA-BoE": re.compile(r"\bpra\b|bank of england|\bboe\b", re.IGNORECASE),
    "Fed-OCC": re.compile(r"\bfed\b|federal reserve|\bocc\b", re.IGNORECASE),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip(" \t\r\n-")


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=;)\s+", text)
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def is_good_text(text: str, *, min_len: int = 60, max_len: int = 360) -> bool:
    text = normalize_space(text)
    lowered = text.lower()
    if not (min_len <= len(text) <= max_len):
        return False
    if any(pattern in lowered for pattern in BAD_PATTERNS):
        return False
    if "?" in text or "|" in text:
        return False
    if len(text.split()) < 8:
        return False
    alpha = [char for char in text if char.isalpha()]
    if alpha and sum(char.isupper() for char in alpha) / len(alpha) > 0.35:
        return False
    return True


def claim_score(sentence: str) -> int:
    lowered = sentence.lower()
    markers = (
        "should",
        "must",
        "expect",
        "require",
        "risk",
        "governance",
        "supervisory",
        "controls",
        "data",
        "reporting",
        "framework",
        "management body",
        "board",
        "liquidity",
        "outsourcing",
        "model",
        "climate",
    )
    return sum(1 for marker in markers if marker in lowered)


def clean_claim(sentence: str) -> str | None:
    claim = normalize_space(sentence)
    claim = re.sub(r"^(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+", "", claim)
    claim = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", claim)
    claim = re.sub(r"^[A-Z][A-Za-z -]{2,50}:\s+", "", claim)
    claim = re.sub(r"\s*\([^)]{0,90}\)", "", claim)
    claim = re.sub(r"^(The|This) (guidelines?|guide|principles?|document|report) (states?|notes?|sets out|explains?) that ", "", claim, flags=re.IGNORECASE)
    for marker in (", including ", ", in particular ", ", for example ", ";"):
        if marker in claim and len(claim.split(marker, 1)[0]) >= 70:
            claim = claim.split(marker, 1)[0].strip() + "."
            break
    claim = normalize_space(claim)
    if claim and not claim.endswith((".", "!", "?")):
        claim += "."
    if not is_good_text(claim, min_len=55, max_len=300):
        return None
    return claim


def extract_claim(text: str) -> str | None:
    candidates = []
    for sentence in split_sentences(text):
        if not is_good_text(sentence, min_len=70, max_len=420):
            continue
        score = claim_score(sentence)
        if score <= 0:
            continue
        claim = clean_claim(sentence)
        if claim:
            candidates.append((score, len(claim), claim))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def replace_once(pattern: str, replacement: str, text: str) -> str | None:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    if not regex.search(text):
        return None
    return regex.sub(replacement, text, count=1)


def make_contradiction(claim: str) -> tuple[str, str] | None:
    transforms = (
        ("should_not", r"\bshould\b", "should not"),
        ("must_not", r"\bmust\b", "must not"),
        ("need_not", r"\bneed to\b", "need not"),
        ("expected_not", r"\bare expected to\b", "are not expected to"),
        ("expects_not", r"\bexpects\b", "does not expect"),
        ("requires_not", r"\brequires\b", "does not require"),
        ("include_exclude", r"\binclude\b", "exclude"),
        ("includes_excludes", r"\bincludes\b", "excludes"),
        ("important_not", r"\bis important\b", "is not important"),
        ("critical_not", r"\bare critical\b", "are not critical"),
        ("able_unable", r"\bare able to\b", "are unable to"),
        ("can_cannot", r"\bcan\b", "cannot"),
    )
    for name, pattern, replacement in transforms:
        if name in {"should_not", "must_not"} and re.search(r"\bnot\b", claim, re.IGNORECASE):
            continue
        candidate = replace_once(pattern, replacement, claim)
        if not candidate or candidate == claim:
            continue
        if re.search(r"\bnot\s+not\b", candidate, re.IGNORECASE):
            continue
        candidate = normalize_space(candidate)
        if is_good_text(candidate, min_len=55, max_len=320):
            return name, candidate
    return None


def source_family(metadata: dict[str, Any]) -> str:
    haystack = " ".join(
        str(metadata.get(key) or "")
        for key in ("family", "source_org", "title", "source", "url", "path", "doc_id")
    )
    for family, pattern in SOURCE_FAMILY_PATTERNS.items():
        if pattern.search(haystack):
            return family
    return "unknown"


def doc_key(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    if metadata.get("id") is not None and metadata.get("chunk_index") is not None:
        return f"{metadata.get('id')}::{metadata.get('chunk_index')}"
    return str(
        metadata.get("chunk_id")
        or metadata.get("doc_id")
        or metadata.get("source")
        or metadata.get("path")
        or doc.get("id")
        or hash(doc.get("content", ""))
    )


def make_components(config: dict[str, Any]) -> tuple[Retriever, Any, RAGPipeline, int]:
    embedder_cfg = config.get("embeddings", {})
    embedder = EmbedderFactory.create(embedder_cfg.get("type", "huggingface"), embedder_cfg)

    vector_cfg = config.get("vector_store", {})
    vector_store = VectorStoreFactory.create(vector_cfg.get("type", "chroma"), vector_cfg.get("config", {}))

    retrieval_cfg = config.get("retrieval", {})
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        k=int(retrieval_cfg.get("k", 10)),
        score_threshold=float(retrieval_cfg.get("score_threshold", 0.0)),
    )

    reranker = None
    reranker_top_k = None
    reranker_cfg = config.get("reranker") or {}
    if reranker_cfg:
        reranker = RerankerFactory.create(
            reranker_cfg["type"],
            {"embedder": embedder, **reranker_cfg},
        )
        reranker_top_k = int(reranker_cfg.get("top_k") or retrieval_cfg.get("k", 10))

    pipeline = RAGPipeline(
        data_manager=None,
        chunker=None,
        embedder=embedder,
        vector_store=vector_store,
        llm=None,
        retriever=retriever,
        reranker=reranker,
        reranker_top_k=reranker_top_k,
        hallucination_detector=None,
        gating_config=config.get("gating", {}),
        source_config=config.get("sources", {}),
    )
    retrieve_k = int((config.get("gating") or {}).get("max_k") or retrieval_cfg.get("k", 20) or 20)
    return retriever, reranker, pipeline, retrieve_k


def retrieve_docs(
    query: str,
    retriever: Retriever,
    reranker: Any,
    pipeline: RAGPipeline,
    retrieve_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    docs = retriever.retrieve(query, k=retrieve_k)
    if reranker:
        docs = reranker.rerank(query, docs, top_k=top_k)
        docs = pipeline._balance_source_families(query, docs, top_k)
    return docs[:top_k]


def build_rows(
    questions: list[dict[str, Any]],
    *,
    retriever: Retriever,
    reranker: Any,
    pipeline: RAGPipeline,
    retrieve_k: int,
    top_k: int,
    per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    claim_items: list[dict[str, Any]] = []
    seen_doc_keys: set[str] = set()

    for question in questions:
        query = str(question.get("query") or "")
        if not query:
            continue
        docs = retrieve_docs(query, retriever, reranker, pipeline, retrieve_k, top_k)
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            if key in seen_doc_keys:
                continue
            claim = extract_claim(str(doc.get("content") or ""))
            contradiction = make_contradiction(claim) if claim else None
            if not claim or not contradiction:
                continue
            metadata = dict(doc.get("metadata") or {})
            claim_items.append(
                {
                    "question_id": question.get("id"),
                    "query": query,
                    "rank": rank,
                    "score": float(doc.get("score", 0.0) or 0.0),
                    "evidence_span": str(doc.get("content") or ""),
                    "candidate_answer": claim,
                    "contradiction_answer": contradiction[1],
                    "contradiction_transform": contradiction[0],
                    "source_family": source_family(metadata),
                    "doc_key": key,
                    "metadata": metadata,
                }
            )
            seen_doc_keys.add(key)

    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for item in claim_items:
        common = {
            "query": item["query"],
            "evidence_span": item["evidence_span"],
            "metadata": {
                "question_id": item["question_id"],
                "retrieval_rank": item["rank"],
                "retrieval_score": item["score"],
                "source_family": item["source_family"],
                "source_metadata": item["metadata"],
                "generation_method": "retrieved_controlled_v1",
            },
        }
        by_label["supported"].append(
            {
                **common,
                "candidate_answer": item["candidate_answer"],
                "expected_label": "supported",
                "support_status": "supported",
                "labels": {"support_status": "supported", "nli_label": "entailment"},
            }
        )
        by_label["contradicted"].append(
            {
                **common,
                "candidate_answer": item["contradiction_answer"],
                "expected_label": "contradicted",
                "support_status": "contradicted",
                "labels": {"support_status": "contradicted", "nli_label": "contradiction"},
                "metadata": {
                    **common["metadata"],
                    "contradiction_transform": item["contradiction_transform"],
                },
            }
        )

    for item in claim_items:
        candidates = [
            other
            for other in claim_items
            if other["doc_key"] != item["doc_key"]
            and other["source_family"] != item["source_family"]
            and lexical_overlap(item["evidence_span"], other["candidate_answer"]) < 0.28
        ]
        if not candidates:
            continue
        other = rng.choice(candidates)
        by_label["unsupported"].append(
            {
                "query": item["query"],
                "candidate_answer": other["candidate_answer"],
                "evidence_span": item["evidence_span"],
                "expected_label": "unsupported",
                "support_status": "unsupported",
                "labels": {"support_status": "unsupported", "nli_label": "neutral"},
                "metadata": {
                    "question_id": item["question_id"],
                    "retrieval_rank": item["rank"],
                    "retrieval_score": item["score"],
                    "source_family": item["source_family"],
                    "unsupported_claim_source_family": other["source_family"],
                    "unsupported_claim_doc_key": other["doc_key"],
                    "source_metadata": item["metadata"],
                    "generation_method": "retrieved_controlled_v1",
                },
            }
        )

    selected: list[dict[str, Any]] = []
    for label in LABELS:
        rows = source_balanced_sample(by_label[label], per_label, rng)
        selected.extend(rows)

    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["id"] = f"finreg_retrieved_detector_v1_{index:04d}"
    return selected


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def token_set(text: str) -> set[str]:
    stop = {"the", "and", "for", "that", "with", "from", "this", "are", "should", "must", "their", "into"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def source_balanced_sample(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str((row.get("metadata") or {}).get("source_family") or "unknown")].append(row)
    for bucket in grouped.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    families = sorted(grouped)
    while len(selected) < limit and any(grouped[family] for family in families):
        for family in families:
            if grouped[family]:
                selected.append(grouped[family].pop())
                if len(selected) >= limit:
                    break
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "labels": dict(Counter(row.get("support_status") for row in rows)),
        "source_families": dict(Counter((row.get("metadata") or {}).get("source_family") for row in rows)),
        "generation_methods": dict(Counter((row.get("metadata") or {}).get("generation_method") for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gating_finreg_ebcar_logit_mi_sc009")
    parser.add_argument(
        "--questions",
        default="data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl",
    )
    parser.add_argument("--output", default="data/domain_finreg/finreg_retrieved_detector_eval_v1.jsonl")
    parser.add_argument("--summary-output", default="data/domain_finreg/finreg_retrieved_detector_eval_v1.summary.json")
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-label", type=int, default=45)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    questions = read_jsonl(Path(args.questions))
    if args.limit_questions:
        rng.shuffle(questions)
        questions = questions[: args.limit_questions]

    config = load_config(args.config)
    retriever, reranker, pipeline, retrieve_k = make_components(config)
    rows = build_rows(
        questions,
        retriever=retriever,
        reranker=reranker,
        pipeline=pipeline,
        retrieve_k=retrieve_k,
        top_k=args.top_k,
        per_label=args.per_label,
        seed=args.seed,
    )

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    write_jsonl(output, rows)
    summary = {
        "config": args.config,
        "questions": args.questions,
        "limit_questions": args.limit_questions,
        "top_k": args.top_k,
        "retrieve_k": retrieve_k,
        "per_label": args.per_label,
        "seed": args.seed,
        "summary": summarize(rows),
        "note": "Controlled automatic eval over real retrieved FinReg contexts; use as detector diagnostic, not final human gold.",
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
