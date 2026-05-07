#!/usr/bin/env python3
"""Run report-ready FinReg controlled-candidate or full-RAG benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


DEFAULT_CONTROLLED_CASES = Path("benchmarks/finreg/controlled_candidate_cases.jsonl")
DEFAULT_FULL_RAG_QUESTIONS = Path("benchmarks/finreg/full_rag_questions.jsonl")


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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def short(text: Any, max_chars: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def average_individual_metric(result: dict[str, Any], key: str) -> float | None:
    values: list[float] = []
    for item in result.get("individual_results") or []:
        value = item.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values)


def controlled_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    risk_by_expected: dict[str, list[float]] = defaultdict(list)
    support_by_expected: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        expected = row.get("expected")
        predicted = row.get("predicted")
        label_detail = row.get("label_detail", "unknown")
        is_correct = bool(row.get("correct"))
        by_label[label_detail]["total"] += 1
        by_label[label_detail]["correct"] += int(is_correct)

        if expected == "unsupported" and predicted == "unsupported":
            tp += 1
        elif expected == "supported" and predicted == "supported":
            tn += 1
        elif expected == "supported" and predicted == "unsupported":
            fp += 1
        elif expected == "unsupported" and predicted == "supported":
            fn += 1

        if isinstance(row.get("unsupported_risk"), (int, float)):
            risk_by_expected[str(expected)].append(float(row["unsupported_risk"]))
        if isinstance(row.get("support_score"), (int, float)):
            support_by_expected[str(expected)].append(float(row["support_score"]))

    total = len(rows)
    unsupported_precision = safe_div(tp, tp + fp)
    unsupported_recall = safe_div(tp, tp + fn)
    unsupported_f1 = safe_div(2 * unsupported_precision * unsupported_recall, unsupported_precision + unsupported_recall)

    return {
        "total": total,
        "correct": tp + tn,
        "accuracy": safe_div(tp + tn, total),
        "unsupported_true_positive": tp,
        "supported_true_negative": tn,
        "false_reject_supported_as_unsupported": fp,
        "false_accept_unsupported_as_supported": fn,
        "unsupported_precision": unsupported_precision,
        "unsupported_recall": unsupported_recall,
        "unsupported_f1": unsupported_f1,
        "false_accept_rate": safe_div(fn, tp + fn),
        "false_reject_rate": safe_div(fp, tn + fp),
        "mean_unsupported_risk_by_expected": {
            key: mean(values) for key, values in risk_by_expected.items()
        },
        "mean_support_score_by_expected": {
            key: mean(values) for key, values in support_by_expected.items()
        },
        "by_label_detail": {
            label: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": safe_div(counts["correct"], counts["total"]),
            }
            for label, counts in sorted(by_label.items())
        },
    }


def full_rag_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(row.get("gating_action", "none")) for row in rows)
    type_counts = Counter(str(row.get("question_type", "unknown")) for row in rows)
    abstain_count = sum(1 for row in rows if row.get("abstained"))
    risk_values = [
        float(row["unsupported_risk"])
        for row in rows
        if isinstance(row.get("unsupported_risk"), (int, float))
    ]
    support_values = [
        float(row["support_score"])
        for row in rows
        if isinstance(row.get("support_score"), (int, float))
    ]
    latency_values = [
        float(row["latency_sec"])
        for row in rows
        if isinstance(row.get("latency_sec"), (int, float))
    ]
    return {
        "total": len(rows),
        "action_counts": dict(action_counts),
        "question_type_counts": dict(type_counts),
        "abstain_count": abstain_count,
        "abstain_rate": safe_div(abstain_count, len(rows)),
        "mean_unsupported_risk": mean(risk_values),
        "mean_support_score": mean(support_values),
        "mean_latency_sec": mean(latency_values),
        "manual_scoring_required": True,
    }


def write_controlled_markdown(path: Path, config_name: str, cases_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# FinReg Controlled Candidate Benchmark Report",
        "",
        f"- config: `{config_name}`",
        f"- cases: `{cases_path}`",
        f"- total cases: `{summary['total']}`",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {summary['accuracy']:.3f} |",
        f"| Unsupported precision | {summary['unsupported_precision']:.3f} |",
        f"| Unsupported recall | {summary['unsupported_recall']:.3f} |",
        f"| Unsupported F1 | {summary['unsupported_f1']:.3f} |",
        f"| False accept rate | {summary['false_accept_rate']:.3f} |",
        f"| False reject rate | {summary['false_reject_rate']:.3f} |",
        "",
        "## Confusion Counts",
        "",
        "| Count | Value |",
        "| --- | ---: |",
        f"| Unsupported true positive | {summary['unsupported_true_positive']} |",
        f"| Supported true negative | {summary['supported_true_negative']} |",
        f"| False accept unsupported as supported | {summary['false_accept_unsupported_as_supported']} |",
        f"| False reject supported as unsupported | {summary['false_reject_supported_as_unsupported']} |",
        "",
        "## Label Detail Accuracy",
        "",
        "| Label detail | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, item in summary["by_label_detail"].items():
        lines.append(f"| {label} | {item['correct']} | {item['total']} | {item['accuracy']:.3f} |")
    lines.extend([
        "",
        "## Report Interpretation",
        "",
        "This benchmark isolates the detector: the candidate answer is fixed, so errors are",
        "primarily detector/retrieval errors rather than generation errors. The most important",
        "safety metric is false accept rate: unsupported answers predicted as supported.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_full_rag_markdown(path: Path, config_name: str, questions_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# FinReg Full RAG Benchmark Report",
        "",
        f"- config: `{config_name}`",
        f"- questions: `{questions_path}`",
        f"- total questions: `{summary['total']}`",
        "",
        "## Automatic System Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Abstain rate | {summary['abstain_rate']:.3f} |",
        f"| Mean unsupported risk | {summary['mean_unsupported_risk']} |",
        f"| Mean support score | {summary['mean_support_score']} |",
        f"| Mean latency sec | {summary['mean_latency_sec']} |",
        "",
        "## Gating Actions",
        "",
        "| Action | Count |",
        "| --- | ---: |",
    ]
    for action, count in sorted(summary["action_counts"].items()):
        lines.append(f"| {action} | {count} |")
    lines.extend([
        "",
        "## Manual Review Requirement",
        "",
        "Full RAG evaluation is end-to-end: retrieval, generation, detector, and gating all",
        "contribute to the final answer. The generated answers should be manually labeled",
        "with `supported`, `unsupported`, `contradicted`, `partial`, or `ambiguous` before",
        "claiming final answer quality metrics.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_controlled(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)
    cases = read_jsonl(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        result = rag.verify_candidate_answer(
            query_text=case["query"],
            candidate_answer=case["candidate_answer"],
            k=args.k,
            return_context=True,
            return_sources=True,
        )
        latency = time.perf_counter() - started
        predicted = "unsupported" if result.get("unsupported_answer_detected") else "supported"
        expected = str(case.get("expected", "")).strip().lower()
        details = result.get("hallucination_details") or {}
        top_context = (result.get("context") or [{}])[0]
        row = {
            **case,
            "predicted": predicted,
            "correct": predicted == expected,
            "latency_sec": latency,
            "unsupported_risk": result.get("unsupported_risk"),
            "support_score": result.get("support_score"),
            "hallucination_score": result.get("hallucination_score"),
            "best_context_label": details.get("best_context_label"),
            "best_context_scores": details.get("best_context_scores"),
            "hard_unsupported_rate": details.get("hard_unsupported_rate"),
            "hard_contradiction_rate": details.get("hard_contradiction_rate"),
            "unsupported_prob_mean": details.get("unsupported_prob_mean"),
            "unsupported_prob_topk": details.get("unsupported_prob_topk"),
            "uncertainty_logit_mi_mean": average_individual_metric(result, "uncertainty_logit_mi"),
            "uncertainty_logit_variance_mean": average_individual_metric(result, "uncertainty_logit_variance"),
            "uncertainty_rep_mi_mean": average_individual_metric(result, "uncertainty_rep_mi"),
            "num_docs_retrieved": result.get("num_docs_retrieved"),
            "top_retrieval_score": top_context.get("score"),
            "top_context": top_context.get("content"),
            "sources": result.get("sources"),
        }
        rows.append(row)
        status = "OK" if row["correct"] else "MISS"
        print(
            f"{case.get('id')} [{status}] expected={expected} predicted={predicted} "
            f"risk={row['unsupported_risk']:.3f} support={row['support_score']:.3f}"
        )

    run_name = safe_name(args.run_name or f"controlled_{args.config}")
    out_dir = args.output_dir / run_name
    write_jsonl(out_dir / "per_case.jsonl", rows)
    summary = controlled_summary(rows)
    write_json(out_dir / "summary.json", summary)
    write_controlled_markdown(out_dir / "report.md", args.config, args.cases, summary)
    print(f"\nWrote: {out_dir}")
    print(f"Accuracy: {summary['accuracy']:.3f}")
    print(f"Unsupported recall: {summary['unsupported_recall']:.3f}")
    print(f"False accept rate: {summary['false_accept_rate']:.3f}")


def run_full_rag(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.disable_gating:
        config.setdefault("gating", {})["enabled"] = False
    rag = RAGPipeline.from_config(config)
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    abstain_message = (config.get("gating", {}).get("abstain_message", "") or "").strip()
    rows: list[dict[str, Any]] = []
    for item in questions:
        started = time.perf_counter()
        result = rag.query(
            query_text=item["query"],
            k=args.k,
            return_context=True,
            return_sources=True,
            detect_hallucinations=not args.disable_detector,
        )
        latency = time.perf_counter() - started
        answer = (result.get("answer") or "").strip()
        gating = result.get("gating") or {}
        top_context = (result.get("context") or [{}])[0]
        row = {
            **item,
            "answer": answer,
            "abstained": bool(abstain_message and answer == abstain_message),
            "gating_action": gating.get("action", "none"),
            "gating_stats": gating.get("stats"),
            "latency_sec": latency,
            "hallucination_detected": result.get("hallucination_detected"),
            "unsupported_answer_detected": result.get("unsupported_answer_detected"),
            "unsupported_risk": result.get("unsupported_risk"),
            "support_score": result.get("support_score"),
            "hallucination_score": result.get("hallucination_score"),
            "hallucination_details": result.get("hallucination_details"),
            "num_docs_retrieved": result.get("num_docs_retrieved"),
            "top_retrieval_score": top_context.get("score"),
            "top_context": top_context.get("content"),
            "sources": result.get("sources"),
        }
        rows.append(row)
        print(
            f"{item.get('id')} action={row['gating_action']} "
            f"abstain={row['abstained']} risk={row.get('unsupported_risk')}"
        )
        print(f"  A: {short(answer)}")

    run_name = safe_name(args.run_name or f"fullrag_{args.config}")
    out_dir = args.output_dir / run_name
    write_jsonl(out_dir / "per_question.jsonl", rows)
    summary = full_rag_summary(rows)
    write_json(out_dir / "summary.json", summary)
    write_full_rag_markdown(out_dir / "report.md", args.config, args.questions, summary)

    csv_path = out_dir / "manual_review_sheet.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "topic",
        "question_type",
        "expected_behavior",
        "query",
        "answer",
        "gating_action",
        "abstained",
        "unsupported_risk",
        "support_score",
        "manual_label",
        "error_type",
        "notes",
        "manual_focus",
        "top_context",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"\nWrote: {out_dir}")
    print(f"Abstain rate: {summary['abstain_rate']:.3f}")
    print("Manual review sheet:", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["controlled", "full-rag"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CONTROLLED_CASES)
    parser.add_argument("--questions", type=Path, default=DEFAULT_FULL_RAG_QUESTIONS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/finreg_real_life_benchmark"))
    parser.add_argument("--disable-detector", action="store_true")
    parser.add_argument("--disable-gating", action="store_true")
    args = parser.parse_args()

    if args.config is None:
        args.config = (
            "gating_finreg_modernbert_detector"
            if args.mode == "controlled"
            else "gating_finreg_openrouter_modernbert_detector"
        )

    if args.mode == "controlled":
        if args.disable_detector:
            raise SystemExit("--disable-detector is not valid for controlled mode")
        run_controlled(args)
    else:
        run_full_rag(args)


if __name__ == "__main__":
    main()
