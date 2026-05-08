#!/usr/bin/env python3
"""Run report-ready FinReg controlled-candidate or full-RAG benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_loader import load_config
from src.rag.rag_pipeline import MODEL_ABSTAIN_PHRASES, RAGPipeline


DEFAULT_CONTROLLED_CASES = Path("benchmarks/finreg/controlled_candidate_cases.jsonl")
DEFAULT_FULL_RAG_QUESTIONS = Path("benchmarks/finreg/full_rag_questions.jsonl")
CONCEPT_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "before", "being",
    "between", "both", "could", "does", "from", "have", "into", "main",
    "more", "must", "only", "other", "over", "risk", "should", "than",
    "that", "their", "there", "these", "this", "through", "under", "when",
    "where", "which", "while", "with", "within", "would",
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def normalize_include_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label in {"supported", "included", "answer_include", "answer_included"}:
        return "included"
    if label in {"unsupported", "not_supported", "not_included", "answer_not_included"}:
        return "not_included"
    return label


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def short(text: Any, max_chars: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def text_hit(text: str, phrase: str) -> bool:
    normalized_text = " ".join((text or "").lower().split())
    normalized_phrase = " ".join(phrase.strip().lower().split())
    if normalized_phrase and normalized_phrase in normalized_text:
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized_phrase)
        if len(token) >= 4 and token not in CONCEPT_STOPWORDS
    ]
    if not tokens:
        return False

    token_hits = sum(
        1
        for token in tokens
        if re.search(rf"\b{re.escape(token)}\b", normalized_text)
    )
    required = max(1, int(round(len(tokens) * 0.6)))
    return token_hits >= required


def exact_text_hit(text: str, phrase: str) -> bool:
    return phrase.strip().lower() in (text or "").lower()


def forbidden_claim_hit(text: str, phrase: str) -> bool:
    answer = (text or "").lower()
    claim = phrase.strip().lower()
    if not claim:
        return False

    negation_markers = (
        "no ",
        "no explicit",
        "not ",
        "not explicitly",
        "not stated",
        "not specify",
        "not specified",
        "not mention",
        "does not",
        "do not",
        "should not",
        "without ",
        "cannot",
    )
    for match in re.finditer(re.escape(claim), answer):
        prefix = answer[max(0, match.start() - 90):match.start()]
        suffix = answer[match.end():match.end() + 120]
        local_window = f"{prefix} {suffix}"
        if any(marker in local_window for marker in negation_markers):
            continue
        return True
    return False


def full_rag_expected_behavior_rubric(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer") or ""
    answer_lower = answer.lower()
    expected_points = as_list(row.get("expected_answer_points"))
    forbidden_claims = as_list(row.get("forbidden_claims"))
    point_hits = [point for point in expected_points if text_hit(answer, point)]
    forbidden_hits = [claim for claim in forbidden_claims if forbidden_claim_hit(answer, claim)]
    coverage = safe_div(len(point_hits), len(expected_points))
    abstained = bool(row.get("abstained"))
    expected_behavior = str(row.get("expected_behavior") or "").strip().lower()

    refute_markers = (
        "no,",
        "does not",
        "do not",
        "not require",
        "not stated",
        "no evidence",
        "no explicit",
        "not explicitly",
        "not in the context",
        "not supported",
        "cannot conclude",
        "should not",
        "not encourage",
        "i don't know based on the provided context",
    )
    cautious_markers = (
        "based on the context",
        "the context",
        "evidence",
        "incomplete",
        "mixed",
        "not established",
        "no explicit",
        "not explicit",
        "not explicitly",
        "not stated",
        "no evidence",
        "not enough",
        "does not specify",
        "cannot determine",
        "clarify",
        "gather more",
        "avoid",
        "retrieve more",
        "i don't know based on the provided context",
    )
    has_refutation = any(marker in answer_lower for marker in refute_markers)
    has_caution = any(marker in answer_lower for marker in cautious_markers)

    if expected_behavior == "answer_with_source_support":
        behavior_pass = (not abstained) and coverage >= 0.35 and not forbidden_hits
    elif expected_behavior == "refute_or_abstain":
        behavior_pass = abstained or (has_refutation and not forbidden_hits)
    elif expected_behavior == "cautious_synthesis":
        behavior_pass = (not abstained) and coverage >= 0.30 and not forbidden_hits
    elif expected_behavior == "cautious_or_abstain":
        behavior_pass = abstained or (has_caution and not forbidden_hits)
    else:
        behavior_pass = None

    return {
        "expected_point_hits": point_hits,
        "expected_point_coverage": coverage,
        "forbidden_claim_hits": forbidden_hits,
        "forbidden_claim_hit_count": len(forbidden_hits),
        "expected_behavior_match": behavior_pass,
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def is_abstain_answer(answer: str, configured_message: str = "") -> bool:
    cleaned = (answer or "").strip()
    lower = cleaned.lower()
    if configured_message and cleaned == configured_message:
        return True
    return any(phrase in lower for phrase in MODEL_ABSTAIN_PHRASES)


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
    score_by_expected: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        expected = normalize_include_label(row.get("expected"))
        predicted = normalize_include_label(row.get("predicted"))
        label_detail = row.get("label_detail", "unknown")
        is_correct = bool(row.get("correct"))
        by_label[label_detail]["total"] += 1
        by_label[label_detail]["correct"] += int(is_correct)

        if expected == "not_included" and predicted == "not_included":
            tp += 1
        elif expected == "included" and predicted == "included":
            tn += 1
        elif expected == "included" and predicted == "not_included":
            fp += 1
        elif expected == "not_included" and predicted == "included":
            fn += 1

        if isinstance(row.get("answer_include_risk"), (int, float)):
            risk_by_expected[str(expected)].append(float(row["answer_include_risk"]))
        if isinstance(row.get("answer_include_score"), (int, float)):
            score_by_expected[str(expected)].append(float(row["answer_include_score"]))

    total = len(rows)
    not_included_precision = safe_div(tp, tp + fp)
    not_included_recall = safe_div(tp, tp + fn)
    not_included_f1 = safe_div(
        2 * not_included_precision * not_included_recall,
        not_included_precision + not_included_recall,
    )

    return {
        "total": total,
        "correct": tp + tn,
        "answer_include_accuracy": safe_div(tp + tn, total),
        "accuracy": safe_div(tp + tn, total),
        "not_included_true_positive": tp,
        "included_true_negative": tn,
        "false_exclude_included_as_not_included": fp,
        "false_include_not_included_as_included": fn,
        "not_included_precision": not_included_precision,
        "not_included_recall": not_included_recall,
        "not_included_f1": not_included_f1,
        "false_include_rate": safe_div(fn, tp + fn),
        "false_exclude_rate": safe_div(fp, tn + fp),
        "mean_answer_include_risk_by_expected": {
            key: mean(values) for key, values in risk_by_expected.items()
        },
        "mean_answer_include_score_by_expected": {
            key: mean(values) for key, values in score_by_expected.items()
        },
        # Backward-compatible aliases for older notebooks/reports.
        "unsupported_true_positive": tp,
        "supported_true_negative": tn,
        "false_reject_supported_as_unsupported": fp,
        "false_accept_unsupported_as_supported": fn,
        "unsupported_precision": not_included_precision,
        "unsupported_recall": not_included_recall,
        "unsupported_f1": not_included_f1,
        "false_accept_rate": safe_div(fn, tp + fn),
        "false_reject_rate": safe_div(fp, tn + fp),
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
    detector_run_count = sum(1 for row in rows if row.get("hallucination_detector_ran"))
    expected_behavior_values = [
        bool(row["expected_behavior_match"])
        for row in rows
        if isinstance(row.get("expected_behavior_match"), bool)
    ]
    coverage_values = [
        float(row["expected_point_coverage"])
        for row in rows
        if isinstance(row.get("expected_point_coverage"), (int, float))
    ]
    forbidden_hit_rows = [
        row for row in rows if int(row.get("forbidden_claim_hit_count") or 0) > 0
    ]
    risk_values = [
        float(row["answer_include_risk"])
        for row in rows
        if isinstance(row.get("answer_include_risk"), (int, float))
    ]
    score_values = [
        float(row["answer_include_score"])
        for row in rows
        if isinstance(row.get("answer_include_score"), (int, float))
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
        "answer_count": len(rows) - abstain_count,
        "answer_rate": safe_div(len(rows) - abstain_count, len(rows)),
        "detector_run_count": detector_run_count,
        "detector_run_rate": safe_div(detector_run_count, len(rows)),
        "expected_behavior_match_rate": safe_div(
            sum(expected_behavior_values),
            len(expected_behavior_values),
        ),
        "mean_expected_point_coverage": mean(coverage_values),
        "forbidden_claim_hit_count": len(forbidden_hit_rows),
        "forbidden_claim_hit_rate": safe_div(len(forbidden_hit_rows), len(rows)),
        "mean_answer_include_risk": mean(risk_values),
        "mean_answer_include_score": mean(score_values),
        # Backward-compatible aliases.
        "mean_unsupported_risk": mean(risk_values),
        "mean_support_score": mean(score_values),
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
        f"| Answer include accuracy | {summary['answer_include_accuracy']:.3f} |",
        f"| Not-included precision | {summary['not_included_precision']:.3f} |",
        f"| Not-included recall | {summary['not_included_recall']:.3f} |",
        f"| Not-included F1 | {summary['not_included_f1']:.3f} |",
        f"| False include rate | {summary['false_include_rate']:.3f} |",
        f"| False exclude rate | {summary['false_exclude_rate']:.3f} |",
        "",
        "## Confusion Counts",
        "",
        "| Count | Value |",
        "| --- | ---: |",
        f"| Not-included true positive | {summary['not_included_true_positive']} |",
        f"| Included true negative | {summary['included_true_negative']} |",
        f"| False include not-included as included | {summary['false_include_not_included_as_included']} |",
        f"| False exclude included as not-included | {summary['false_exclude_included_as_not_included']} |",
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
        "safety metric is false include rate: not-included answers predicted as included.",
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
        f"| Answer rate | {summary['answer_rate']:.3f} |",
        f"| Detector run rate | {summary['detector_run_rate']:.3f} |",
        f"| Expected behavior match rate | {summary['expected_behavior_match_rate']:.3f} |",
        f"| Mean expected point coverage | {summary['mean_expected_point_coverage']} |",
        f"| Forbidden claim hit rate | {summary['forbidden_claim_hit_rate']:.3f} |",
        f"| Mean answer include risk | {summary['mean_answer_include_risk']} |",
        f"| Mean answer include score | {summary['mean_answer_include_score']} |",
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
        "with `included`, `not_included`, `contradicted`, `partial`, or `ambiguous` before",
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
        predicted = "included" if result.get("answer_include_detected") else "not_included"
        expected = normalize_include_label(case.get("expected"))
        details = result.get("hallucination_details") or {}
        top_context = (result.get("context") or [{}])[0]
        answer_include_risk = result.get("answer_include_risk", result.get("unsupported_risk"))
        answer_include_score = result.get("answer_include_score", result.get("support_score"))
        row = {
            **case,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "latency_sec": latency,
            "answer_include_detected": result.get("answer_include_detected"),
            "answer_include_risk": answer_include_risk,
            "answer_include_score": answer_include_score,
            # Backward-compatible aliases.
            "unsupported_risk": result.get("unsupported_risk"),
            "support_score": result.get("support_score"),
            "hallucination_score": result.get("hallucination_score"),
            "best_context_label": details.get("best_context_label"),
            "best_context_scores": details.get("best_context_scores"),
            "hard_answer_not_included_rate": details.get("hard_answer_not_included_rate"),
            "hard_unsupported_rate": details.get("hard_unsupported_rate"),
            "hard_contradiction_rate": details.get("hard_contradiction_rate"),
            "answer_include_prob_mean": details.get("answer_include_prob_mean"),
            "answer_include_prob_topk": details.get("answer_include_prob_topk"),
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
            f"include_risk={answer_include_risk:.3f} include_score={answer_include_score:.3f}"
        )

    run_name = safe_name(args.run_name or f"controlled_{args.config}")
    out_dir = args.output_dir / run_name
    write_jsonl(out_dir / "per_case.jsonl", rows)
    summary = controlled_summary(rows)
    write_json(out_dir / "summary.json", summary)
    write_controlled_markdown(out_dir / "report.md", args.config, args.cases, summary)
    print(f"\nWrote: {out_dir}")
    print(f"Answer include accuracy: {summary['answer_include_accuracy']:.3f}")
    print(f"Not-included recall: {summary['not_included_recall']:.3f}")
    print(f"False include rate: {summary['false_include_rate']:.3f}")


def run_full_rag(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.disable_gating:
        config.setdefault("gating", {})["enabled"] = False
    if args.disable_detector:
        config.setdefault("hallucination_detector", {})["enabled"] = False
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
            "abstained": is_abstain_answer(answer, abstain_message),
            "gating_action": gating.get("action", "none"),
            "gating_stats": gating.get("stats"),
            "latency_sec": latency,
            "hallucination_detected": result.get("hallucination_detected"),
            "hallucination_detector_ran": result.get("hallucination_detector_ran"),
            "answer_include_detected": result.get("answer_include_detected"),
            "answer_include_risk": result.get("answer_include_risk"),
            "answer_include_score": result.get("answer_include_score"),
            # Backward-compatible aliases.
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
        row.update(full_rag_expected_behavior_rubric(row))
        rows.append(row)
        print(
            f"{item.get('id')} action={row['gating_action']} "
            f"abstain={row['abstained']} "
            f"expected_behavior={row.get('expected_behavior_match')} "
            f"include_risk={row.get('answer_include_risk')}"
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
        "answer_include_risk",
        "answer_include_score",
        "expected_behavior_match",
        "expected_point_coverage",
        "forbidden_claim_hit_count",
        "expected_point_hits",
        "forbidden_claim_hits",
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
