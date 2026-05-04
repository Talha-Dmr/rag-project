#!/usr/bin/env python3
"""
Evaluate retrieval methods against a corpus-grounded benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.config_loader import load_config
from src.embeddings import EmbedderFactory  # noqa: F401 - ensures registration
from src.reranking import RerankerFactory  # noqa: F401 - ensures registration
from src.vector_stores import VectorStoreFactory  # noqa: F401 - ensures registration
from src.reranking.rerankers.bm25_reranker import BM25Reranker
from src.reranking.rerankers.ebcar_reranker import EBCARReranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PROJECT_ROOT / "output" / "benchmarks" / "benchmark_finreg_retrieval_50.json"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "finreg"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "retrieval_results"
DEFAULT_CONFIG = "gating_finreg_ebcar_logit_mi_sc009"
TOP_K_VALUES = (1, 3, 5, 10)
VALID_METHODS = ("bm25", "dense", "hybrid", "adaptive_or_stochastic", "all")


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    layer: str
    source_collection: str
    text: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance on a benchmark.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark JSON path.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="Processed corpus root.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Project config name without .yaml.")
    parser.add_argument(
        "--retrieval-method",
        default="all",
        choices=VALID_METHODS,
        help="Retrieval method bucket to evaluate.",
    )
    parser.add_argument("--top-k-max", type=int, default=10, help="Maximum top-k to retrieve.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Base output directory.")
    parser.add_argument("--run-id", default="", help="Optional stable run id suffix.")
    parser.add_argument(
        "--baseline-processed-root",
        type=Path,
        default=None,
        help="Baseline processed corpus root used to resolve benchmark gold chunk texts.",
    )
    parser.add_argument("--candidate-pool", type=int, default=50, help="Candidate pool for hybrid/adaptive methods.")
    parser.add_argument(
        "--dense-collection-name",
        default="rag_finreg_eval_processed",
        help="Collection name for the processed dense index.",
    )
    parser.add_argument(
        "--dense-persist-directory",
        type=Path,
        default=None,
        help="Optional persist directory override for the dense index.",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the processed dense index.")
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size to use while embedding processed chunks for indexing.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional embedder device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--chunk-match-mode",
        choices=("exact_id", "text_substring"),
        default="exact_id",
        help="How chunk-level hits should be scored.",
    )
    parser.add_argument(
        "--fixed-run-dir",
        action="store_true",
        help="Write outputs directly under --output-dir instead of creating a timestamped child folder.",
    )
    return parser.parse_args()


def normalize_doc_id(value: str) -> str:
    return Path(str(value)).stem.strip().lower()


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit(f"Benchmark must be a list: {path}")

    required = {"id", "question", "answer", "doc_id", "layer", "gold_chunks", "difficulty", "question_type"}
    for item in items:
        missing = required - set(item.keys())
        if missing:
            raise SystemExit(f"Benchmark item missing fields {sorted(missing)}: {item}")
    return items


def iter_processed_chunk_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.chunks.jsonl")):
        yield path


def load_processed_chunks(root: Path) -> List[ChunkRecord]:
    chunk_records: List[ChunkRecord] = []
    for path in iter_processed_chunk_paths(root):
        source_collection = path.parent.relative_to(root).as_posix()
        layer = source_collection.split("/", 1)[0] if "/" in source_collection else source_collection
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunk_id = str(payload["chunk_id"])
                doc_id = normalize_doc_id(payload.get("source_file") or chunk_id.rsplit("_", 1)[0])
                metadata = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "layer": layer,
                    "source_collection": payload.get("source_collection", source_collection),
                    "title": payload.get("title", ""),
                    "source_file": payload.get("source_file", ""),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                }
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        layer=layer,
                        source_collection=str(payload.get("source_collection", source_collection)),
                        text=str(payload.get("text", "")),
                        metadata=metadata,
                    )
                )
    return chunk_records


def normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def load_chunk_text_index(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in iter_processed_chunk_paths(root):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunk_id = str(payload.get("chunk_id", "")).strip()
                text = normalize_match_text(str(payload.get("text", "")))
                if chunk_id and text:
                    index[chunk_id] = text
    return index


def build_dense_stack(
    config_name: str,
    collection_name: str,
    persist_directory: Optional[Path],
    embed_batch_size: int,
    device_override: str,
):
    config = load_config(config_name)
    embed_cfg = dict(config.get("embeddings", {}) or {})
    vector_store_cfg = dict(config.get("vector_store", {}) or {})
    vector_store_inner_cfg = dict(vector_store_cfg.get("config", {}) or {})

    if device_override:
        embed_cfg["device"] = device_override
    embed_cfg["batch_size"] = embed_batch_size

    if persist_directory is not None:
        vector_store_inner_cfg["persist_directory"] = str(persist_directory)
    elif vector_store_inner_cfg.get("persist_directory"):
        vector_store_inner_cfg["persist_directory"] = str(
            (PROJECT_ROOT / vector_store_inner_cfg["persist_directory"]).resolve()
            if not Path(str(vector_store_inner_cfg["persist_directory"])).is_absolute()
            else Path(str(vector_store_inner_cfg["persist_directory"]))
        )

    vector_store_inner_cfg["collection_name"] = collection_name

    embedder = EmbedderFactory.create(embed_cfg.get("type", "huggingface"), embed_cfg)
    vector_store = VectorStoreFactory.create(vector_store_cfg.get("type", "chroma"), vector_store_inner_cfg)
    return config, embedder, vector_store


def ensure_dense_index(
    chunk_records: List[ChunkRecord],
    embedder,
    vector_store,
    rebuild_index: bool,
    manifest_path: Path,
    processed_root: Path,
) -> None:
    expected_count = len(chunk_records)
    current_count = vector_store.get_count()

    manifest_matches = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_matches = (
                manifest.get("processed_root") == str(processed_root)
                and int(manifest.get("chunk_count", -1)) == expected_count
            )
        except Exception:
            manifest_matches = False

    if rebuild_index or current_count != expected_count or not manifest_matches:
        vector_store.delete_collection()
        texts = [record.text for record in chunk_records]
        embeddings = embedder.embed_batch(texts)
        metadatas = [record.metadata for record in chunk_records]
        vector_store.add_documents(texts, embeddings, metadatas)
        vector_store.persist()
        manifest = {
            "processed_root": str(processed_root),
            "chunk_count": expected_count,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dense_search(query: str, embedder, vector_store, k: int) -> List[Dict[str, Any]]:
    embedding = embedder.embed_text(query)
    return vector_store.search(query_embedding=embedding, k=k)


def bm25_search(query: str, bm25_reranker: BM25Reranker, corpus_docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    cloned = [
        {"content": doc["content"], "metadata": dict(doc.get("metadata", {})), "score": doc.get("score", 0.0)}
        for doc in corpus_docs
    ]
    return bm25_reranker.rerank(query, cloned, top_k=k)


def hybrid_search(
    query: str,
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}

    def merge(results: Sequence[Dict[str, Any]], label: str) -> None:
        for rank, item in enumerate(results, start=1):
            chunk_id = str(item.get("metadata", {}).get("chunk_id", ""))
            if not chunk_id:
                continue
            bucket = combined.setdefault(
                chunk_id,
                {
                    "content": item.get("content", ""),
                    "metadata": dict(item.get("metadata", {})),
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf": 0.0,
                },
            )
            bucket[f"{label}_score"] = float(item.get("score", 0.0) or 0.0)
            bucket["rrf"] += 1.0 / (60.0 + rank)

    merge(dense_results, "dense")
    merge(bm25_results, "bm25")

    merged_results: List[Dict[str, Any]] = []
    for payload in combined.values():
        payload["score"] = payload["rrf"]
        merged_results.append(payload)

    merged_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return merged_results[:k]


def adaptive_search(
    query: str,
    hybrid_results: List[Dict[str, Any]],
    ebcar_reranker: EBCARReranker,
    k: int,
) -> List[Dict[str, Any]]:
    cloned = [
        {"content": doc["content"], "metadata": dict(doc.get("metadata", {})), "score": float(doc.get("score", 0.0) or 0.0)}
        for doc in hybrid_results
    ]
    return ebcar_reranker.rerank(query, cloned, top_k=k)


def reciprocal_rank(items: Sequence[str], gold_values: Sequence[str]) -> float:
    gold = set(gold_values)
    for rank, item in enumerate(items, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def bool_hit(items: Sequence[str], gold_values: Sequence[str], k: int) -> int:
    gold = set(gold_values)
    return int(any(item in gold for item in items[:k]))


def texts_match(candidate_text: str, gold_text: str) -> bool:
    if not candidate_text or not gold_text:
        return False
    return candidate_text in gold_text or gold_text in candidate_text


def chunk_hit_value(
    top_chunk_ids: Sequence[str],
    top_chunk_texts: Sequence[str],
    gold_chunk_ids: Sequence[str],
    gold_chunk_texts: Sequence[str],
    k: int,
    match_mode: str,
) -> int:
    if match_mode == "exact_id":
        return bool_hit(top_chunk_ids, gold_chunk_ids, k)

    limited_texts = top_chunk_texts[:k]
    return int(
        any(texts_match(candidate_text, gold_text) for candidate_text in limited_texts for gold_text in gold_chunk_texts)
    )


def chunk_reciprocal_rank(
    top_chunk_ids: Sequence[str],
    top_chunk_texts: Sequence[str],
    gold_chunk_ids: Sequence[str],
    gold_chunk_texts: Sequence[str],
    match_mode: str,
) -> float:
    if match_mode == "exact_id":
        return reciprocal_rank(top_chunk_ids, gold_chunk_ids)

    for rank, candidate_text in enumerate(top_chunk_texts, start=1):
        if any(texts_match(candidate_text, gold_text) for gold_text in gold_chunk_texts):
            return 1.0 / rank
    return 0.0


def format_for_csv(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    metrics: Dict[str, float] = {}
    for k in TOP_K_VALUES:
        metrics[f"doc_recall_at_{k}"] = sum(float(row[f"doc_hit_at_{k}"]) for row in rows) / len(rows)
        metrics[f"chunk_recall_at_{k}"] = sum(float(row[f"chunk_hit_at_{k}"]) for row in rows) / len(rows)
    metrics["doc_mrr"] = sum(float(row["doc_rr"]) for row in rows) / len(rows)
    metrics["chunk_mrr"] = sum(float(row["chunk_rr"]) for row in rows) / len(rows)
    metrics["count"] = float(len(rows))
    return metrics


def aggregate_by(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, "unknown"))].append(row)

    aggregated_rows: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        metrics = aggregate_metrics(buckets[key])
        aggregated_rows.append({field: key, **metrics})
    return aggregated_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_for_csv(row.get(key)) for key in fieldnames})


def write_summary(
    path: Path,
    retrieval_method: str,
    benchmark_path: Path,
    processed_root: Path,
    overall_metrics: Dict[str, float],
    by_layer: List[Dict[str, Any]],
    by_difficulty: List[Dict[str, Any]],
    by_question_type: List[Dict[str, Any]],
    failure_count: int,
    corpus_index_desc: str,
) -> None:
    def best_and_worst(rows: List[Dict[str, Any]], field: str) -> Tuple[str, str]:
        if not rows:
            return "n/a", "n/a"
        ordered = sorted(rows, key=lambda item: float(item.get("chunk_recall_at_10", 0.0)))
        worst = f"{ordered[0][field]} ({ordered[0].get('chunk_recall_at_10', 0.0):.3f})"
        best = f"{ordered[-1][field]} ({ordered[-1].get('chunk_recall_at_10', 0.0):.3f})"
        return best, worst

    best_layer, worst_layer = best_and_worst(by_layer, "layer")
    best_difficulty, worst_difficulty = best_and_worst(by_difficulty, "difficulty")
    best_qtype, worst_qtype = best_and_worst(by_question_type, "question_type")

    lines = [
        "# Retrieval Evaluation Summary",
        "",
        f"- Retrieval method used: `{retrieval_method}`",
        f"- Benchmark path: `{benchmark_path}`",
        f"- Corpus/index used: `{corpus_index_desc}`",
        "",
        "## Top-line Metrics",
        "",
        f"- Doc Recall@1: {overall_metrics.get('doc_recall_at_1', 0.0):.4f}",
        f"- Doc Recall@3: {overall_metrics.get('doc_recall_at_3', 0.0):.4f}",
        f"- Doc Recall@5: {overall_metrics.get('doc_recall_at_5', 0.0):.4f}",
        f"- Doc Recall@10: {overall_metrics.get('doc_recall_at_10', 0.0):.4f}",
        f"- Chunk Recall@1: {overall_metrics.get('chunk_recall_at_1', 0.0):.4f}",
        f"- Chunk Recall@3: {overall_metrics.get('chunk_recall_at_3', 0.0):.4f}",
        f"- Chunk Recall@5: {overall_metrics.get('chunk_recall_at_5', 0.0):.4f}",
        f"- Chunk Recall@10: {overall_metrics.get('chunk_recall_at_10', 0.0):.4f}",
        f"- Doc MRR: {overall_metrics.get('doc_mrr', 0.0):.4f}",
        f"- Chunk MRR: {overall_metrics.get('chunk_mrr', 0.0):.4f}",
        "",
        "## Breakdown Highlights",
        "",
        f"- Best layer by chunk@10: {best_layer}",
        f"- Worst layer by chunk@10: {worst_layer}",
        f"- Best difficulty by chunk@10: {best_difficulty}",
        f"- Worst difficulty by chunk@10: {worst_difficulty}",
        f"- Best question type by chunk@10: {best_qtype}",
        f"- Worst question type by chunk@10: {worst_qtype}",
        f"- Total failures at chunk@10: {failure_count}",
        "",
        "## Notes",
        "",
        "- Evaluation is retrieval-only. No answer generation or LLM judging is involved.",
        "- Doc-level success requires at least one retrieved result from the benchmark `doc_id`.",
        "- Chunk-level success requires exact matching against benchmark `gold_chunks`.",
        "- The `adaptive_or_stochastic` bucket is mapped to the project's EBCAR reranker applied over hybrid candidates, because the codebase exposes adaptive reranking logic rather than a standalone stochastic retriever API.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_row_from_results(
    item: Dict[str, Any],
    retrieval_method: str,
    retrieved_results: List[Dict[str, Any]],
    chunk_match_mode: str,
    gold_chunk_texts: Sequence[str],
) -> Dict[str, Any]:
    top_doc_ids = [normalize_doc_id(str(result.get("metadata", {}).get("doc_id", ""))) for result in retrieved_results]
    top_chunk_ids = [str(result.get("metadata", {}).get("chunk_id", "")) for result in retrieved_results]
    top_chunk_texts = [normalize_match_text(str(result.get("content", ""))) for result in retrieved_results]
    top_scores = [float(result.get("score", 0.0) or 0.0) for result in retrieved_results]

    row: Dict[str, Any] = {
        "question_id": item["id"],
        "question": item["question"],
        "retrieval_method": retrieval_method,
        "benchmark_doc_id": normalize_doc_id(item["doc_id"]),
        "benchmark_gold_chunks": list(item["gold_chunks"]),
        "layer": item["layer"],
        "difficulty": item["difficulty"],
        "question_type": item["question_type"],
        "topk_doc_ids": top_doc_ids,
        "topk_chunk_ids": top_chunk_ids,
        "topk_scores": top_scores,
    }

    for k in TOP_K_VALUES:
        row[f"doc_hit_at_{k}"] = bool_hit(top_doc_ids, [row["benchmark_doc_id"]], k)
        row[f"chunk_hit_at_{k}"] = chunk_hit_value(
            top_chunk_ids=top_chunk_ids,
            top_chunk_texts=top_chunk_texts,
            gold_chunk_ids=item["gold_chunks"],
            gold_chunk_texts=gold_chunk_texts,
            k=k,
            match_mode=chunk_match_mode,
        )

    row["doc_rr"] = reciprocal_rank(top_doc_ids, [row["benchmark_doc_id"]])
    row["chunk_rr"] = chunk_reciprocal_rank(
        top_chunk_ids=top_chunk_ids,
        top_chunk_texts=top_chunk_texts,
        gold_chunk_ids=item["gold_chunks"],
        gold_chunk_texts=gold_chunk_texts,
        match_mode=chunk_match_mode,
    )
    return row


def build_run_name(method: str, run_id: str) -> str:
    if run_id:
        return f"{method}_{run_id}"
    return f"{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_method_outputs(
    run_dir: Path,
    retrieval_method: str,
    benchmark_path: Path,
    processed_root: Path,
    per_question_rows: List[Dict[str, Any]],
    corpus_index_desc: str,
) -> Dict[str, float]:
    overall_metrics = aggregate_metrics(per_question_rows)
    by_layer = aggregate_by(per_question_rows, "layer")
    by_difficulty = aggregate_by(per_question_rows, "difficulty")
    by_question_type = aggregate_by(per_question_rows, "question_type")

    failures = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "retrieval_method": row["retrieval_method"],
            "benchmark_doc_id": row["benchmark_doc_id"],
            "benchmark_gold_chunks": row["benchmark_gold_chunks"],
            "retrieved_doc_ids_top10": row["topk_doc_ids"][:10],
            "retrieved_chunk_ids_top10": row["topk_chunk_ids"][:10],
            "layer": row["layer"],
            "difficulty": row["difficulty"],
            "question_type": row["question_type"],
        }
        for row in per_question_rows
        if int(row["chunk_hit_at_10"]) == 0
    ][:10]

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics_overall.json").write_text(json.dumps(overall_metrics, indent=2), encoding="utf-8")
    write_csv(run_dir / "metrics_by_layer.csv", by_layer)
    write_csv(run_dir / "metrics_by_difficulty.csv", by_difficulty)
    write_csv(run_dir / "metrics_by_question_type.csv", by_question_type)
    write_csv(run_dir / "per_question_results.csv", per_question_rows)
    write_csv(run_dir / "failures_top10.csv", failures)
    write_summary(
        run_dir / "summary.md",
        retrieval_method=retrieval_method,
        benchmark_path=benchmark_path,
        processed_root=processed_root,
        overall_metrics=overall_metrics,
        by_layer=by_layer,
        by_difficulty=by_difficulty,
        by_question_type=by_question_type,
        failure_count=sum(1 for row in per_question_rows if int(row["chunk_hit_at_10"]) == 0),
        corpus_index_desc=corpus_index_desc,
    )
    return overall_metrics


def comparison_row(method: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    row = {"retrieval_method": method}
    for key in (
        "doc_recall_at_1",
        "doc_recall_at_3",
        "doc_recall_at_5",
        "doc_recall_at_10",
        "chunk_recall_at_1",
        "chunk_recall_at_3",
        "chunk_recall_at_5",
        "chunk_recall_at_10",
        "doc_mrr",
        "chunk_mrr",
    ):
        row[key] = metrics.get(key, 0.0)
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(args.benchmark)
    processed_chunks = load_processed_chunks(args.processed_root)
    if not processed_chunks:
        raise SystemExit(f"No processed chunks found under {args.processed_root}")

    gold_chunk_text_index: Dict[str, str] = {}
    if args.chunk_match_mode == "text_substring":
        baseline_root = args.baseline_processed_root or args.processed_root
        gold_chunk_text_index = load_chunk_text_index(baseline_root)

    methods = (
        ["bm25", "dense", "hybrid", "adaptive_or_stochastic"]
        if args.retrieval_method == "all"
        else [args.retrieval_method]
    )
    top_k_max = max(args.top_k_max, max(TOP_K_VALUES))

    dense_config, embedder, vector_store = build_dense_stack(
        config_name=args.config,
        collection_name=args.dense_collection_name,
        persist_directory=args.dense_persist_directory,
        embed_batch_size=args.embed_batch_size,
        device_override=args.device,
    )

    manifest_path = args.output_dir / "_dense_index_manifest.json"
    if any(method in {"dense", "hybrid", "adaptive_or_stochastic"} for method in methods):
        ensure_dense_index(
            chunk_records=processed_chunks,
            embedder=embedder,
            vector_store=vector_store,
            rebuild_index=args.rebuild_index,
            manifest_path=manifest_path,
            processed_root=args.processed_root,
        )

    corpus_docs = [
        {
            "content": record.text,
            "metadata": dict(record.metadata),
            "score": 0.0,
        }
        for record in processed_chunks
    ]

    bm25_reranker = BM25Reranker({"top_k": top_k_max})
    ebcar_reranker = EBCARReranker({"embedder": embedder})

    comparison_rows: List[Dict[str, Any]] = []
    session_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    for method in methods:
        per_question_rows: List[Dict[str, Any]] = []
        run_name = build_run_name(method, session_run_id)
        run_dir = args.output_dir if args.fixed_run_dir else args.output_dir / run_name

        for item in benchmark:
            question = str(item["question"])

            if method == "bm25":
                results = bm25_search(question, bm25_reranker, corpus_docs, top_k_max)
            elif method == "dense":
                results = dense_search(question, embedder, vector_store, top_k_max)
            elif method == "hybrid":
                dense_results = dense_search(question, embedder, vector_store, args.candidate_pool)
                bm25_results = bm25_search(question, bm25_reranker, corpus_docs, args.candidate_pool)
                results = hybrid_search(question, dense_results, bm25_results, top_k_max)
            elif method == "adaptive_or_stochastic":
                dense_results = dense_search(question, embedder, vector_store, args.candidate_pool)
                bm25_results = bm25_search(question, bm25_reranker, corpus_docs, args.candidate_pool)
                hybrid_results = hybrid_search(question, dense_results, bm25_results, args.candidate_pool)
                results = adaptive_search(question, hybrid_results, ebcar_reranker, top_k_max)
            else:
                raise SystemExit(f"Unsupported retrieval method: {method}")

            gold_chunk_texts = [
                gold_chunk_text_index.get(chunk_id, "")
                for chunk_id in item["gold_chunks"]
                if gold_chunk_text_index.get(chunk_id, "")
            ]
            row = benchmark_row_from_results(
                item,
                method,
                results[:top_k_max],
                chunk_match_mode=args.chunk_match_mode,
                gold_chunk_texts=gold_chunk_texts,
            )
            per_question_rows.append(row)

        persist_dir = args.dense_persist_directory
        if persist_dir is None:
            persist_dir = Path(dense_config["vector_store"]["config"]["persist_directory"])

        overall_metrics = save_method_outputs(
            run_dir=run_dir,
            retrieval_method=method,
            benchmark_path=args.benchmark,
            processed_root=args.processed_root,
            per_question_rows=per_question_rows,
            corpus_index_desc=f"processed corpus at {args.processed_root} with dense collection {args.dense_collection_name} in {persist_dir}",
        )
        comparison_rows.append(comparison_row(method, overall_metrics))

    if comparison_rows:
        write_csv(args.output_dir / "comparison_summary.csv", comparison_rows)


if __name__ == "__main__":
    main()
