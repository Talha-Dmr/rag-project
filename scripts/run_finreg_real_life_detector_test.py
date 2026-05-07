#!/usr/bin/env python3
"""Run retrieval-backed real-life smoke tests for the FinReg detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


DEFAULT_CASES = [
    {
        "id": "supported_model_risk",
        "expected": "included",
        "query": "What is model risk management in the context of banking regulation?",
        "candidate_answer": (
            "Model risk management requires independent validation, use constraints, "
            "and override governance."
        ),
    },
    {
        "id": "partial_not_included_model_risk",
        "expected": "not_included",
        "query": "What is model risk management in the context of banking regulation?",
        "candidate_answer": (
            "Model risk management requires independent validation, use constraints, "
            "override governance, and regulator pre-approval within five business days."
        ),
    },
    {
        "id": "contradiction_model_risk",
        "expected": "not_included",
        "query": "What is model risk management in the context of banking regulation?",
        "candidate_answer": (
            "Model risk management does not require independent validation or "
            "governance controls."
        ),
    },
    {
        "id": "supported_bcbs239",
        "expected": "included",
        "query": "What is the core objective of BCBS 239?",
        "candidate_answer": (
            "BCBS 239 targets accurate and timely risk data aggregation through "
            "governance, architecture, and controls that support decision-useful "
            "risk reporting."
        ),
    },
    {
        "id": "not_included_bcbs239_portal",
        "expected": "not_included",
        "query": "What is the core objective of BCBS 239?",
        "candidate_answer": (
            "BCBS 239 requires banks to submit daily XML risk reports through a "
            "regulator-hosted portal."
        ),
    },
]


def read_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CASES)

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


def short(text: str, max_chars: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def normalize_include_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label in {"supported", "included", "answer_include", "answer_included"}:
        return "included"
    if label in {"unsupported", "not_supported", "not_included", "answer_not_included"}:
        return "not_included"
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gating_finreg_modernbert_detector")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/finreg_real_life_detector_test.jsonl"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)
    cases = read_cases(args.cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0

    with args.output.open("w", encoding="utf-8") as out:
        for case in cases:
            result = rag.verify_candidate_answer(
                query_text=case["query"],
                candidate_answer=case["candidate_answer"],
                k=args.k,
                return_context=True,
                return_sources=True,
            )

            predicted = "included" if result.get("answer_include_detected") else "not_included"
            expected = normalize_include_label(case.get("expected"))
            is_correct = expected is None or predicted == expected
            correct += int(is_correct)

            details = result.get("hallucination_details") or {}
            top_context = (result.get("context") or [{}])[0]
            answer_include_risk = result.get("answer_include_risk", result.get("unsupported_risk"))
            answer_include_score = result.get("answer_include_score", result.get("support_score"))
            row = {
                "id": case.get("id"),
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "query": case["query"],
                "candidate_answer": case["candidate_answer"],
                "answer_include_detected": result.get("answer_include_detected"),
                "answer_include_risk": answer_include_risk,
                "answer_include_score": answer_include_score,
                # Backward-compatible aliases.
                "unsupported_risk": result.get("unsupported_risk"),
                "support_score": result.get("support_score"),
                "best_context_label": details.get("best_context_label"),
                "best_context_scores": details.get("best_context_scores"),
                "num_docs_retrieved": result.get("num_docs_retrieved"),
                "top_retrieval_score": top_context.get("score"),
                "top_context": top_context.get("content"),
                "sources": result.get("sources"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            status = "OK" if is_correct else "MISS"
            print(
                f"{case.get('id')} [{status}] expected={expected} predicted={predicted} "
                f"include_risk={answer_include_risk:.3f} "
                f"include_score={answer_include_score:.3f} "
                f"best={details.get('best_context_label')}"
            )
            print(f"  Q: {short(case['query'])}")
            print(f"  A: {short(case['candidate_answer'])}")
            print(f"  Top context: {short(top_context.get('content', ''))}")

    total = len(cases)
    print("\nSummary")
    print(f"cases: {total}")
    print(f"correct: {correct}/{total} ({correct / total:.2%})" if total else "correct: 0/0")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
