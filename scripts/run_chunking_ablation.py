#!/usr/bin/env python3
"""
Run a controlled chunking ablation on the fixed financial regulation benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_pdfs import build_chunk_records_for_document, write_json, write_jsonl  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "output" / "benchmarks" / "benchmark_finreg_retrieval_50.json"
BASELINE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "finreg"
ABLATION_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "chunking_ablation"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "retrieval_results" / "chunking_ablation"
DEFAULT_VECTOR_ROOT = PROJECT_ROOT / "data" / "vector_db" / "chunking_ablation"
DEFAULT_CONFIG = "gating_finreg_ebcar_logit_mi_sc009"

PHASE1_METHODS = ("bm25", "hybrid")
PHASE2_METHODS = ("dense", "adaptive_or_stochastic")
ALL_STRATEGIES = (
    "baseline_existing",
    "structure_aware",
    "structure_aware_small",
    "structure_aware_small_heading",
)


@dataclass
class RunResult:
    chunking_strategy: str
    retrieval_method: str
    output_dir: Path
    metrics: Dict[str, float]
    by_layer: Dict[str, Dict[str, float]]
    by_difficulty: Dict[str, Dict[str, float]]
    by_question_type: Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled chunking ablation study.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--baseline-processed-root", type=Path, default=BASELINE_PROCESSED_ROOT)
    parser.add_argument("--ablation-processed-root", type=Path, default=ABLATION_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vector-root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("phase1", "phase2", "all"), default="all")
    parser.add_argument("--chunking-strategies", nargs="+", default=list(ALL_STRATEGIES))
    parser.add_argument("--phase1-methods", nargs="+", default=list(PHASE1_METHODS))
    parser.add_argument("--phase2-methods", nargs="+", default=list(PHASE2_METHODS))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild-corpora", action="store_true")
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--skip-existing-runs", action="store_true")
    return parser.parse_args()


def processed_root_for_strategy(args: argparse.Namespace, strategy: str) -> Path:
    if strategy == "baseline_existing":
        return args.baseline_processed_root
    return args.ablation_processed_root / strategy


def iter_processed_documents(root: Path) -> Iterable[Dict[str, Path]]:
    for txt_path in sorted(root.rglob("*.txt")):
        if txt_path.name == "manifest.txt":
            continue
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


def build_strategy_corpus(args: argparse.Namespace, strategy: str) -> Path:
    destination_root = processed_root_for_strategy(args, strategy)
    if strategy == "baseline_existing":
        return destination_root

    manifest_path = destination_root / "manifest.json"
    if manifest_path.exists() and not args.rebuild_corpora:
        return destination_root

    if destination_root.exists() and args.rebuild_corpora:
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for doc_paths in iter_processed_documents(args.baseline_processed_root):
        relative_parent = doc_paths["txt"].parent.relative_to(args.baseline_processed_root)
        target_dir = destination_root / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        txt_path = doc_paths["txt"]
        doc_stem = txt_path.stem
        full_text = txt_path.read_text(encoding="utf-8")
        metadata = json.loads(doc_paths["metadata"].read_text(encoding="utf-8"))
        pages = json.loads(doc_paths["pages"].read_text(encoding="utf-8"))

        for source_path in (txt_path, doc_paths["metadata"], doc_paths["pages"]):
            shutil.copy2(source_path, target_dir / source_path.name)

        chunk_records = build_chunk_records_for_document(
            doc_stem=doc_stem,
            full_text=full_text,
            document_metadata=metadata,
            chunk_strategy=strategy,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        write_jsonl(target_dir / f"{doc_stem}.chunks.jsonl", chunk_records)

        results.append(
            {
                "file": str(txt_path),
                "status": "processed",
                "source_collection": metadata.get("source_collection", ""),
                "title": metadata.get("title", ""),
                "pages": metadata.get("num_pages", 0),
                "chars": metadata.get("num_characters", len(full_text)),
                "chunks": len(chunk_records),
                "chunk_strategy": strategy,
                "needs_ocr_review": metadata.get("needs_ocr_review", False),
            }
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_root": str(args.baseline_processed_root),
        "processed_root": str(destination_root),
        "chunk_strategy": strategy,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "total_files": len(results),
        "processed_files": len(results),
        "skipped_existing_files": 0,
        "files": results,
    }
    write_json(manifest_path, manifest)
    return destination_root


def run_eval(
    args: argparse.Namespace,
    strategy: str,
    retrieval_method: str,
    processed_root: Path,
) -> RunResult:
    run_dir = args.output_dir / strategy / retrieval_method
    metrics_path = run_dir / "metrics_overall.json"
    if metrics_path.exists() and args.skip_existing_runs:
        return load_run_result(strategy, retrieval_method, run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    collection_name = f"rag_finreg_ablation_{strategy}"
    persist_directory = args.vector_root / strategy
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
        retrieval_method,
        "--output-dir",
        str(run_dir),
        "--candidate-pool",
        str(args.candidate_pool),
        "--dense-collection-name",
        collection_name,
        "--dense-persist-directory",
        str(persist_directory),
        "--embed-batch-size",
        str(args.embed_batch_size),
        "--device",
        args.device,
        "--chunk-match-mode",
        "text_substring",
        "--run-id",
        "ablation",
        "--fixed-run-dir",
    ]
    if args.rebuild_indexes:
        command.append("--rebuild-index")

    env = dict(**__import__("os").environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f";{existing_pythonpath}" if existing_pythonpath else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)
    return load_run_result(strategy, retrieval_method, run_dir)


def read_csv_metrics(path: Path, key_field: str) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get(key_field, "")).strip()
            if not key:
                continue
            rows[key] = {
                metric: float(value)
                for metric, value in row.items()
                if metric != key_field and value not in ("", None)
            }
    return rows


def load_run_result(strategy: str, retrieval_method: str, run_dir: Path) -> RunResult:
    metrics = json.loads((run_dir / "metrics_overall.json").read_text(encoding="utf-8"))
    return RunResult(
        chunking_strategy=strategy,
        retrieval_method=retrieval_method,
        output_dir=run_dir,
        metrics={key: float(value) for key, value in metrics.items()},
        by_layer=read_csv_metrics(run_dir / "metrics_by_layer.csv", "layer"),
        by_difficulty=read_csv_metrics(run_dir / "metrics_by_difficulty.csv", "difficulty"),
        by_question_type=read_csv_metrics(run_dir / "metrics_by_question_type.csv", "question_type"),
    )


def comparison_row(result: RunResult, phase: str) -> Dict[str, object]:
    return {
        "phase": phase,
        "chunking_strategy": result.chunking_strategy,
        "retrieval_method": result.retrieval_method,
        "doc_recall_at_10": result.metrics.get("doc_recall_at_10", 0.0),
        "chunk_recall_at_10": result.metrics.get("chunk_recall_at_10", 0.0),
        "doc_mrr": result.metrics.get("doc_mrr", 0.0),
        "chunk_mrr": result.metrics.get("chunk_mrr", 0.0),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def strategy_score(results: Sequence[RunResult], strategy: str) -> Dict[str, float]:
    strategy_results = [result for result in results if result.chunking_strategy == strategy]
    hard_scores = [
        result.by_difficulty.get("hard", {}).get("chunk_recall_at_10", 0.0)
        for result in strategy_results
    ]
    eu_basel_scores = []
    for result in strategy_results:
        for layer_name in ("eu_regulations", "basel"):
            if layer_name in result.by_layer:
                eu_basel_scores.append(result.by_layer[layer_name].get("chunk_recall_at_10", 0.0))
    return {
        "chunk_recall_at_10": mean(result.metrics.get("chunk_recall_at_10", 0.0) for result in strategy_results),
        "chunk_mrr": mean(result.metrics.get("chunk_mrr", 0.0) for result in strategy_results),
        "hard_chunk_recall_at_10": mean(hard_scores) if hard_scores else 0.0,
        "eu_basel_chunk_recall_at_10": mean(eu_basel_scores) if eu_basel_scores else 0.0,
    }


def choose_phase1_winner(results: Sequence[RunResult], strategies: Sequence[str]) -> str:
    scores = {strategy: strategy_score(results, strategy) for strategy in strategies}
    ordered = sorted(
        strategies,
        key=lambda strategy: (
            scores[strategy]["chunk_recall_at_10"],
            scores[strategy]["chunk_mrr"],
            scores[strategy]["hard_chunk_recall_at_10"],
            scores[strategy]["eu_basel_chunk_recall_at_10"],
        ),
        reverse=True,
    )
    return ordered[0]


def build_markdown_summary(
    phase1_results: Sequence[RunResult],
    phase2_results: Sequence[RunResult],
    phase1_winner: str,
) -> str:
    phase1_rows = [comparison_row(result, "phase1") for result in phase1_results]
    phase2_rows = [comparison_row(result, "phase2") for result in phase2_results]
    winner_scores = strategy_score(phase1_results, phase1_winner)
    baseline_scores = strategy_score(phase1_results, "baseline_existing")

    comparable_results: Dict[str, Dict[str, RunResult]] = {}
    for result in list(phase1_results) + list(phase2_results):
        comparable_results.setdefault(result.retrieval_method, {})[result.chunking_strategy] = result

    benefit_rows = []
    for method, buckets in comparable_results.items():
        baseline = buckets.get("baseline_existing")
        winner = buckets.get(phase1_winner)
        if baseline and winner:
            benefit_rows.append(
                (
                    method,
                    winner.metrics.get("chunk_recall_at_10", 0.0) - baseline.metrics.get("chunk_recall_at_10", 0.0),
                    winner.metrics.get("chunk_mrr", 0.0) - baseline.metrics.get("chunk_mrr", 0.0),
                )
            )
    benefit_rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_method = benefit_rows[0][0] if benefit_rows else "n/a"

    lines = [
        "# Chunking Ablation Summary",
        "",
        "## Chunking Strategies Tested",
        "",
    ]
    for strategy in sorted({result.chunking_strategy for result in list(phase1_results) + list(phase2_results)}):
        lines.append(f"- {strategy}")

    lines.extend(["", "## Phase 1 Results", ""])
    for row in phase1_rows:
        lines.append(
            f"- {row['chunking_strategy']} / {row['retrieval_method']}: "
            f"chunk@10={row['chunk_recall_at_10']:.4f}, chunk_mrr={row['chunk_mrr']:.4f}, "
            f"doc@10={row['doc_recall_at_10']:.4f}, doc_mrr={row['doc_mrr']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Phase 1 Winner",
            "",
            f"- Winning strategy: `{phase1_winner}`",
            f"- Reason: mean phase-1 chunk@10={winner_scores['chunk_recall_at_10']:.4f}, "
            f"chunk_mrr={winner_scores['chunk_mrr']:.4f}, "
            f"hard chunk@10={winner_scores['hard_chunk_recall_at_10']:.4f}, "
            f"EU/Basel chunk@10={winner_scores['eu_basel_chunk_recall_at_10']:.4f}",
        ]
    )

    lines.extend(["", "## Phase 2 Results", ""])
    for row in phase2_rows:
        lines.append(
            f"- {row['chunking_strategy']} / {row['retrieval_method']}: "
            f"chunk@10={row['chunk_recall_at_10']:.4f}, chunk_mrr={row['chunk_mrr']:.4f}, "
            f"doc@10={row['doc_recall_at_10']:.4f}, doc_mrr={row['doc_mrr']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            f"- Recommended chunking strategy: `{phase1_winner}`",
            f"- Phase-1 baseline mean chunk@10={baseline_scores['chunk_recall_at_10']:.4f}; winner mean chunk@10={winner_scores['chunk_recall_at_10']:.4f}",
            f"- Phase-1 baseline mean chunk_mrr={baseline_scores['chunk_mrr']:.4f}; winner mean chunk_mrr={winner_scores['chunk_mrr']:.4f}",
            f"- Retrieval method with the largest chunk-level gain vs baseline: `{best_method}`",
            "- Interpretation: chunk-level improvements should be judged primarily from chunk@10 and chunk_mrr, with doc-level metrics used as a secondary guardrail.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.vector_root.mkdir(parents=True, exist_ok=True)

    selected_strategies = [strategy for strategy in args.chunking_strategies if strategy in ALL_STRATEGIES]
    if "baseline_existing" not in selected_strategies:
        selected_strategies = ["baseline_existing"] + selected_strategies

    phase1_results: List[RunResult] = []
    if args.phase in {"phase1", "all"}:
        for strategy in selected_strategies:
            processed_root = build_strategy_corpus(args, strategy)
            for method in args.phase1_methods:
                phase1_results.append(run_eval(args, strategy, method, processed_root))

    phase1_rows = [comparison_row(result, "phase1") for result in phase1_results]
    write_csv(args.output_dir / "phase1_comparison.csv", phase1_rows)

    phase1_winner = "baseline_existing"
    if phase1_results:
        phase1_winner = choose_phase1_winner(phase1_results, selected_strategies)

    phase2_results: List[RunResult] = []
    if args.phase in {"phase2", "all"}:
        phase2_strategies = ["baseline_existing", phase1_winner]
        for strategy in phase2_strategies:
            processed_root = build_strategy_corpus(args, strategy)
            for method in args.phase2_methods:
                phase2_results.append(run_eval(args, strategy, method, processed_root))

    phase2_rows = [comparison_row(result, "phase2") for result in phase2_results]
    write_csv(args.output_dir / "phase2_comparison.csv", phase2_rows)

    all_rows = phase1_rows + phase2_rows
    write_csv(args.output_dir / "ablation_summary.csv", all_rows)
    summary_md = build_markdown_summary(phase1_results, phase2_results, phase1_winner)
    (args.output_dir / "ablation_summary.md").write_text(summary_md, encoding="utf-8")


if __name__ == "__main__":
    main()
