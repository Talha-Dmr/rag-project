#!/usr/bin/env python3
"""
Compare two chunking strategies on retrieval and grounding metrics.

This script is intended to evaluate:
- semantic chunking with MPNet sentence embeddings
- section-aware chunking

It builds or reuses processed corpora, prepares dense indexes, runs retrieval eval,
then runs grounding/gating eval on the same indexed corpus.

Key answered-risk metrics:
- answered_contradiction_rate:
    Fraction of non-abstained answers where at least one retrieved context is
    classified as contradiction by the detector.
- unsupported_answer_rate:
    Fraction of non-abstained answers where none of the retrieved contexts is
    classified as entailment by the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_pdfs import page_lookup_for_chunk, sha1_text, write_json, write_jsonl  # noqa: E402
from eval_retrieval import (  # noqa: E402
    build_dense_stack,
    ensure_dense_index,
    load_processed_chunks,
)
from src.chunking import ChunkerFactory  # noqa: E402,F401
from src.core.config_loader import load_config  # noqa: E402
from src.rag.rag_pipeline import RAGPipeline  # noqa: E402


DEFAULT_CONFIG = "gating_finreg_ebcar_logit_mi_sc009"
DEFAULT_BENCHMARK = PROJECT_ROOT / "output" / "benchmarks" / "benchmark_finreg_retrieval_50.json"
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "domain_finreg" / "questions_finreg_conflict_50.jsonl"
DEFAULT_BASELINE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "finreg"
DEFAULT_ABLATION_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "chunking_pair_eval"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation_results" / "chunking_pair_eval"
DEFAULT_VECTOR_ROOT = PROJECT_ROOT / "data" / "vector_db" / "chunking_pair_eval"
DEFAULT_RETRIEVAL_METHOD = "all"
VALID_RETRIEVAL_METHODS = ("bm25", "dense", "hybrid", "adaptive_or_stochastic", "all")
VALID_STRATEGIES = ("semantic_mpnet", "section_aware")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    processed_dirname: str
    chunking_strategy_label: str


STRATEGY_SPECS: Dict[str, StrategySpec] = {
    "semantic_mpnet": StrategySpec(
        name="semantic_mpnet",
        processed_dirname="semantic_mpnet",
        chunking_strategy_label="semantic_mpnet",
    ),
    "section_aware": StrategySpec(
        name="section_aware",
        processed_dirname="section_aware",
        chunking_strategy_label="section_aware",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config name without .yaml.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--baseline-processed-root", type=Path, default=DEFAULT_BASELINE_PROCESSED_ROOT)
    parser.add_argument("--ablation-processed-root", type=Path, default=DEFAULT_ABLATION_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vector-root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["semantic_mpnet", "section_aware"],
        choices=VALID_STRATEGIES,
        help="Chunking strategies to compare.",
    )
    parser.add_argument(
        "--retrieval-method",
        default=DEFAULT_RETRIEVAL_METHOD,
        choices=VALID_RETRIEVAL_METHODS,
        help="Retrieval method(s) for eval_retrieval.py.",
    )
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="Optional question limit for grounding eval.")
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--chunk-match-mode", choices=("exact_id", "text_substring"), default="text_substring")
    parser.add_argument("--rebuild-corpora", action="store_true")
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-grounding", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--dump-grounding-details",
        action="store_true",
        help="Write per-question grounding details JSONL.",
    )
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.7,
        help="Semantic chunker similarity threshold.",
    )
    parser.add_argument(
        "--semantic-min-chunk-size",
        type=int,
        default=100,
        help="Semantic chunker min chunk size in characters.",
    )
    parser.add_argument(
        "--semantic-max-chunk-size",
        type=int,
        default=1000,
        help="Semantic chunker max chunk size in characters.",
    )
    parser.add_argument(
        "--semantic-model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence model for semantic chunking.",
    )
    return parser.parse_args()


def normalize_strategy_names(raw_names: Sequence[str]) -> List[str]:
    seen = set()
    names: List[str] = []
    for name in raw_names:
        if name not in STRATEGY_SPECS:
            raise SystemExit(f"Unsupported strategy: {name}")
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def iter_processed_documents(root: Path) -> Iterable[Dict[str, Path]]:
    for txt_path in sorted(root.rglob("*.txt")):
        metadata_path = txt_path.with_suffix(".metadata.json")
        pages_path = txt_path.with_suffix(".pages.json")
        chunks_path = txt_path.with_suffix(".chunks.jsonl")
        if metadata_path.exists() and pages_path.exists() and chunks_path.exists():
            yield {
                "txt": txt_path,
                "metadata": metadata_path,
                "pages": pages_path,
                "chunks": chunks_path,
            }


def semantic_chunker_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "similarity_threshold": float(args.semantic_similarity_threshold),
        "min_chunk_size": int(args.semantic_min_chunk_size),
        "max_chunk_size": int(args.semantic_max_chunk_size),
        "embedder": {
            "type": "huggingface",
            "model_name": args.semantic_model_name,
            "device": args.device,
            "cache_folder": str((PROJECT_ROOT / "models" / "embeddings").resolve()),
            "batch_size": int(args.embed_batch_size),
        },
    }


def build_semantic_chunk_records(
    doc_stem: str,
    full_text: str,
    document_metadata: Dict[str, Any],
    chunker,
    strategy_label: str,
) -> List[Dict[str, Any]]:
    chunks = chunker.chunk(full_text)
    records: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(chunks, start=1):
        page_start, page_end = page_lookup_for_chunk(chunk_text, full_text)
        records.append(
            {
                "chunk_id": f"{doc_stem}_{idx:04d}",
                "chunk_index": idx,
                "text": chunk_text,
                "text_sha1": sha1_text(chunk_text),
                "source_file": document_metadata["source_file"],
                "source_collection": document_metadata["source_collection"],
                "layer": document_metadata["layer"],
                "title": document_metadata["title"],
                "regulator": document_metadata["regulator"],
                "doc_type": document_metadata["doc_type"],
                "jurisdiction": document_metadata["jurisdiction"],
                "language": document_metadata["language"],
                "year": document_metadata["year"],
                "page_start": page_start,
                "page_end": page_end,
                "chunking_strategy": strategy_label,
            }
        )
    return records


def build_section_aware_chunk_records(
    doc_stem: str,
    full_text: str,
    document_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    from ingest_pdfs import build_chunk_records_for_document

    return build_chunk_records_for_document(
        doc_stem=doc_stem,
        full_text=full_text,
        document_metadata=document_metadata,
        chunk_strategy="structure_aware",
        chunk_size=1200,
        overlap=200,
    )


def build_strategy_corpus(args: argparse.Namespace, strategy_name: str) -> Path:
    spec = STRATEGY_SPECS[strategy_name]
    destination_root = args.ablation_processed_root / spec.processed_dirname
    manifest_path = destination_root / "manifest.json"

    if manifest_path.exists() and not args.rebuild_corpora:
        return destination_root

    if destination_root.exists() and args.rebuild_corpora:
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    semantic_chunker = None
    if strategy_name == "semantic_mpnet":
        semantic_chunker = ChunkerFactory.create("semantic", semantic_chunker_config(args))

    baseline_docs = list(iter_processed_documents(args.baseline_processed_root))
    total_docs = len(baseline_docs)
    file_rows: List[Dict[str, Any]] = []
    for doc_idx, doc_paths in enumerate(baseline_docs, start=1):
        txt_path = doc_paths["txt"]
        relative_parent = txt_path.parent.relative_to(args.baseline_processed_root)
        target_dir = destination_root / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        chunk_output_path = target_dir / f"{txt_path.stem}.chunks.jsonl"

        if chunk_output_path.exists() and not args.rebuild_corpora:
            if doc_idx == 1 or doc_idx % 10 == 0 or doc_idx == total_docs:
                print(
                    f"[{strategy_name}] reuse {doc_idx}/{total_docs}: {txt_path.name}",
                    flush=True,
                )
            continue

        metadata = json.loads(doc_paths["metadata"].read_text(encoding="utf-8"))
        pages = json.loads(doc_paths["pages"].read_text(encoding="utf-8"))
        full_text = txt_path.read_text(encoding="utf-8")
        doc_stem = txt_path.stem

        shutil.copy2(txt_path, target_dir / txt_path.name)
        shutil.copy2(doc_paths["metadata"], target_dir / doc_paths["metadata"].name)
        shutil.copy2(doc_paths["pages"], target_dir / doc_paths["pages"].name)

        if strategy_name == "semantic_mpnet":
            chunk_records = build_semantic_chunk_records(
                doc_stem=doc_stem,
                full_text=full_text,
                document_metadata=metadata,
                chunker=semantic_chunker,
                strategy_label=spec.chunking_strategy_label,
            )
        elif strategy_name == "section_aware":
            chunk_records = build_section_aware_chunk_records(
                doc_stem=doc_stem,
                full_text=full_text,
                document_metadata=metadata,
            )
        else:
            raise SystemExit(f"Unsupported strategy: {strategy_name}")

        write_jsonl(chunk_output_path, chunk_records)
        file_rows.append(
            {
                "file": str(txt_path),
                "source_collection": metadata.get("source_collection", ""),
                "title": metadata.get("title", ""),
                "pages": len(pages),
                "chars": len(full_text),
                "chunks": len(chunk_records),
                "chunking_strategy": spec.chunking_strategy_label,
            }
        )
        if doc_idx == 1 or doc_idx % 10 == 0 or doc_idx == total_docs:
            print(
                f"[{strategy_name}] built {doc_idx}/{total_docs}: {txt_path.name} -> {len(chunk_records)} chunks",
                flush=True,
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.baseline_processed_root),
        "processed_root": str(destination_root),
        "chunking_strategy": spec.chunking_strategy_label,
        "semantic_config": semantic_chunker_config(args) if strategy_name == "semantic_mpnet" else None,
        "files": file_rows,
    }
    write_json(manifest_path, manifest)
    return destination_root


def prepare_dense_index(
    args: argparse.Namespace,
    strategy_name: str,
    processed_root: Path,
) -> Dict[str, Any]:
    collection_name = f"rag_finreg_pair_eval_{strategy_name}"
    persist_directory = args.vector_root / strategy_name
    dense_output_dir = args.output_dir / strategy_name / "_dense_index"
    dense_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dense_output_dir / "manifest.json"

    chunk_records = load_processed_chunks(processed_root)
    if not chunk_records:
        raise SystemExit(f"No processed chunks found under {processed_root}")

    _, embedder, vector_store = build_dense_stack(
        config_name=args.config,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embed_batch_size=args.embed_batch_size,
        device_override=args.device,
    )
    ensure_dense_index(
        chunk_records=chunk_records,
        embedder=embedder,
        vector_store=vector_store,
        rebuild_index=args.rebuild_indexes,
        manifest_path=manifest_path,
        processed_root=processed_root,
    )
    return {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "manifest_path": manifest_path,
    }


def run_retrieval_eval(
    args: argparse.Namespace,
    strategy_name: str,
    processed_root: Path,
    index_info: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir = args.output_dir / strategy_name / "retrieval" / args.retrieval_method
    metrics_path = run_dir / "metrics_overall.json"
    comparison_path = run_dir / "comparison_summary.csv"
    if args.skip_existing and (comparison_path.exists() or metrics_path.exists()):
        if comparison_path.exists():
            with comparison_path.open("r", encoding="utf-8", newline="") as handle:
                return {"rows": list(csv.DictReader(handle))}
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_retrieval.py"),
        "--benchmark",
        str(args.benchmark),
        "--processed-root",
        str(processed_root),
        "--baseline-processed-root",
        str(args.baseline_processed_root),
        "--config",
        args.config,
        "--retrieval-method",
        args.retrieval_method,
        "--top-k-max",
        str(args.top_k_max),
        "--output-dir",
        str(run_dir),
        "--run-id",
        "pair_eval",
        "--candidate-pool",
        str(args.candidate_pool),
        "--dense-collection-name",
        str(index_info["collection_name"]),
        "--dense-persist-directory",
        str(index_info["persist_directory"]),
        "--embed-batch-size",
        str(args.embed_batch_size),
        "--device",
        args.device,
        "--chunk-match-mode",
        args.chunk_match_mode,
        "--fixed-run-dir",
    ]
    if args.rebuild_indexes:
        command.append("--rebuild-index")

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f"{os.pathsep}{existing_pythonpath}" if existing_pythonpath else "")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)

    if comparison_path.exists():
        with comparison_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {"rows": rows}
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    raise SystemExit(f"Retrieval output missing under {run_dir}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_questions(path: Path, seed: int, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def update_stats_bucket(bucket: Dict[str, List[float]], stats: Dict[str, Any]) -> None:
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            bucket[key].append(float(value))


def summarize_bucket(bucket: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: avg(values) for key, values in bucket.items()}


def build_runtime_config(
    base_config_name: str,
    index_info: Dict[str, Any],
    device: str,
    embed_batch_size: int,
    llm_temperature: Optional[float],
) -> Dict[str, Any]:
    config = load_config(base_config_name)

    embeddings_cfg = dict(config.get("embeddings", {}) or {})
    embeddings_cfg["batch_size"] = int(embed_batch_size)
    if device:
        embeddings_cfg["device"] = device
    config["embeddings"] = embeddings_cfg

    vector_store_cfg = dict(config.get("vector_store", {}) or {})
    vector_store_inner_cfg = dict(vector_store_cfg.get("config", {}) or {})
    vector_store_inner_cfg["collection_name"] = str(index_info["collection_name"])
    vector_store_inner_cfg["persist_directory"] = str(index_info["persist_directory"])
    vector_store_cfg["config"] = vector_store_inner_cfg
    config["vector_store"] = vector_store_cfg

    if llm_temperature is not None:
        llm_cfg = dict(config.get("llm", {}) or {})
        llm_cfg["temperature"] = float(llm_temperature)
        config["llm"] = llm_cfg

    return config


def run_grounding_eval(
    args: argparse.Namespace,
    strategy_name: str,
    index_info: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir = args.output_dir / strategy_name / "grounding"
    summary_path = run_dir / "grounding_metrics.json"
    details_path = run_dir / "grounding_details.jsonl"
    if args.skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    questions = load_questions(args.questions, seed=args.seed, limit=args.limit)
    config = build_runtime_config(
        base_config_name=args.config,
        index_info=index_info,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
        llm_temperature=args.llm_temperature,
    )
    rag = RAGPipeline.from_config(config)
    abstain_message = (config.get("gating", {}) or {}).get("abstain_message", "").strip()

    total = 0
    abstain = 0
    answered = 0
    detector_failures = 0
    answered_contradiction_cases = 0
    unsupported_answer_cases = 0
    action_counts = Counter()
    stats_buckets: Dict[str, Dict[str, List[float]]] = {
        "all": defaultdict(list),
        "answered": defaultdict(list),
        "abstain": defaultdict(list),
    }

    details_handle = None
    if args.dump_grounding_details:
        details_handle = details_path.open("w", encoding="utf-8")

    for item in questions:
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        result = rag.query(
            query_text=query,
            return_context=False,
            detect_hallucinations=True,
        )
        total += 1
        answer = str(result.get("answer", "") or "").strip()
        gating = result.get("gating") or {}
        stats = gating.get("stats") or {}
        action = str(gating.get("action", "none") or "none")
        action_counts[action] += 1

        if result.get("hallucination_detected") is None or result.get("hallucination_error"):
            detector_failures += 1

        is_abstain = bool(abstain_message and answer == abstain_message)
        if is_abstain:
            abstain += 1
            update_stats_bucket(stats_buckets["abstain"], stats)
        else:
            answered += 1
            update_stats_bucket(stats_buckets["answered"], stats)
            hard_contradiction_rate = stats.get("hard_contradiction_rate")
            entailment_rate = stats.get("entailment_rate")
            if isinstance(hard_contradiction_rate, (int, float)) and float(hard_contradiction_rate) > 0.0:
                answered_contradiction_cases += 1
            elif bool(result.get("hallucination_detected")):
                answered_contradiction_cases += 1
            if isinstance(entailment_rate, (int, float)) and float(entailment_rate) <= 0.0:
                unsupported_answer_cases += 1

        update_stats_bucket(stats_buckets["all"], stats)

        if details_handle is not None:
            row = {
                "id": item.get("id"),
                "type": item.get("type"),
                "query": query,
                "action": action,
                "is_abstain": is_abstain,
                "hallucination_detected": result.get("hallucination_detected"),
                "hallucination_error": result.get("hallucination_error"),
                "hard_contradiction_rate": stats.get("hard_contradiction_rate"),
                "entailment_rate": stats.get("entailment_rate"),
                "source_consistency": stats.get("source_consistency"),
                "retrieval_max_score": stats.get("retrieval_max_score"),
                "retrieval_mean_score": stats.get("retrieval_mean_score"),
            }
            details_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if details_handle is not None:
        details_handle.close()

    summary = {
        "strategy": strategy_name,
        "questions_path": str(args.questions),
        "seed": int(args.seed),
        "limit": int(args.limit),
        "total": total,
        "answered": answered,
        "abstain": abstain,
        "coverage": (answered / total) if total else 0.0,
        "abstain_rate": (abstain / total) if total else 0.0,
        "detector_failures": detector_failures,
        "actions": dict(action_counts),
        "answered_contradiction_cases": answered_contradiction_cases,
        "answered_contradiction_rate": (
            answered_contradiction_cases / answered
        ) if answered else 0.0,
        "unsupported_answer_cases": unsupported_answer_cases,
        "unsupported_answer_rate": (
            unsupported_answer_cases / answered
        ) if answered else 0.0,
        "stats_all": summarize_bucket(stats_buckets["all"]),
        "stats_answered": summarize_bucket(stats_buckets["answered"]),
        "stats_abstain": summarize_bucket(stats_buckets["abstain"]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def collect_retrieval_rows(run_dir: Path) -> List[Dict[str, Any]]:
    comparison_csv = run_dir / "comparison_summary.csv"
    overall_json = run_dir / "metrics_overall.json"
    if comparison_csv.exists():
        with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if overall_json.exists():
        payload = json.loads(overall_json.read_text(encoding="utf-8"))
        return [payload]
    return []


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def write_comparison_summary(args: argparse.Namespace, strategies: Sequence[str]) -> None:
    summary_root = args.output_dir / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    retrieval_rows: List[Dict[str, Any]] = []
    grounding_rows: List[Dict[str, Any]] = []
    markdown_lines: List[str] = [
        "# Chunking Pair Eval Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Config: `{args.config}`",
        f"- Benchmark: `{args.benchmark}`",
        f"- Questions: `{args.questions}`",
        "",
        "## Grounding Metrics",
        "",
        "| Strategy | Abstain | Coverage | Answered Contradiction | Unsupported Answer |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for strategy_name in strategies:
        grounding_path = args.output_dir / strategy_name / "grounding" / "grounding_metrics.json"
        if grounding_path.exists():
            grounding = json.loads(grounding_path.read_text(encoding="utf-8"))
            grounding_row = {
                "strategy": strategy_name,
                "abstain_rate": grounding.get("abstain_rate", 0.0),
                "coverage": grounding.get("coverage", 0.0),
                "answered_contradiction_rate": grounding.get("answered_contradiction_rate", 0.0),
                "unsupported_answer_rate": grounding.get("unsupported_answer_rate", 0.0),
                "answered": grounding.get("answered", 0),
                "total": grounding.get("total", 0),
            }
            grounding_rows.append(grounding_row)
            markdown_lines.append(
                f"| {strategy_name} | {format_float(grounding_row['abstain_rate'])} | "
                f"{format_float(grounding_row['coverage'])} | "
                f"{format_float(grounding_row['answered_contradiction_rate'])} | "
                f"{format_float(grounding_row['unsupported_answer_rate'])} |"
            )

        retrieval_run_dir = args.output_dir / strategy_name / "retrieval" / args.retrieval_method
        for row in collect_retrieval_rows(retrieval_run_dir):
            retrieval_rows.append({"strategy": strategy_name, **row})

    markdown_lines.extend(
        [
            "",
            "## Retrieval Metrics",
            "",
            "| Strategy | Method | Doc@10 | Chunk@10 | Chunk@MRR |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in retrieval_rows:
        method = row.get("retrieval_method", args.retrieval_method)
        markdown_lines.append(
            f"| {row.get('strategy', '')} | {method} | "
            f"{format_float(row.get('doc_recall_at_10'))} | "
            f"{format_float(row.get('chunk_recall_at_10'))} | "
            f"{format_float(row.get('chunk_mrr'))} |"
        )

    (summary_root / "comparison.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    write_json(summary_root / "grounding_comparison.json", grounding_rows)
    write_json(summary_root / "retrieval_comparison.json", retrieval_rows)

    if grounding_rows:
        with (summary_root / "grounding_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grounding_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grounding_rows)

    if retrieval_rows:
        with (summary_root / "retrieval_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(retrieval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(retrieval_rows)


def main() -> None:
    args = parse_args()
    strategies = normalize_strategy_names(args.strategies)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.vector_root.mkdir(parents=True, exist_ok=True)
    args.ablation_processed_root.mkdir(parents=True, exist_ok=True)

    for strategy_name in strategies:
        processed_root = build_strategy_corpus(args, strategy_name)
        index_info = prepare_dense_index(args, strategy_name, processed_root)

        if not args.skip_retrieval:
            run_retrieval_eval(
                args=args,
                strategy_name=strategy_name,
                processed_root=processed_root,
                index_info=index_info,
            )
        if not args.skip_grounding:
            run_grounding_eval(
                args=args,
                strategy_name=strategy_name,
                index_info=index_info,
            )

    write_comparison_summary(args, strategies)
    print(f"Wrote pair-eval outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
