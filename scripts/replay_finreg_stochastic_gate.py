#!/usr/bin/env python3
"""Replay FinReg stochastic gate candidates from saved per-question stats.

This avoids rerunning retrieval/LLM generation for every stochastic adapter.
Create the input with scripts/eval_grounding_proxy.py --dump-details, then use
this script to sweep epistemic sources and thresholds on the same fixed outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.eval_grounding_proxy import (
    augment_stats_with_evidence_gate,
    compute_shadow_epistemic,
    compute_shadow_uncertainty,
    decide_shadow_action_2d,
    summarize,
    update_stats,
)
from src.rag.stochastic_epistemic_adapter import ADAPTER_SOURCES


DEFAULT_EPI_SOURCES = list(ADAPTER_SOURCES)


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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_labels(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {
        str(row.get("id")): row
        for row in read_jsonl(Path(path))
        if row.get("id")
    }


def load_question_ids(path: str) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    for row in read_jsonl(Path(path)):
        query = str(row.get("query") or "").strip()
        qid = str(row.get("id") or "").strip()
        if query and qid:
            mapping[query] = qid
    return mapping


def expected_action_family(label: dict[str, Any]) -> str:
    task_family = str(label.get("task_family") or "")
    support_level = str(label.get("support_level") or "")
    ambiguity_family = str(label.get("ambiguity_family") or "")

    if task_family == "sanity_anchor" and ambiguity_family in ("none_or_low", ""):
        return "answer"
    if support_level.endswith("_hard"):
        return "retrieve_more"
    if task_family in ("comparison_conflict", "supervisory_escalation"):
        return "retrieve_more"
    return "answer"


def action_utility(action: str, label: dict[str, Any]) -> float:
    expected = expected_action_family(label)
    support_level = str(label.get("support_level") or "")
    task_family = str(label.get("task_family") or "")

    if expected == "answer":
        return {"answer": 1.0, "retrieve_more": 0.45, "abstain": 0.0}.get(action, 0.0)

    # Conflict/hard cases: asking for more evidence is the safest useful action.
    if action == "retrieve_more":
        return 1.0
    if action == "abstain":
        return 0.75 if support_level.endswith("_hard") else 0.45
    if action == "answer":
        return 0.45 if task_family == "comparison_conflict" else 0.35
    return 0.0


def operating_score(
    balanced_utility: float | None,
    expected_accuracy: dict[str, float],
    abstain_rate: float,
    answer_contradiction_rate: float | None,
) -> float | None:
    if balanced_utility is None:
        return None

    answer_acc = expected_accuracy.get("answer", 0.0)
    retrieve_acc = expected_accuracy.get("retrieve_more", 0.0)
    answer_contra = answer_contradiction_rate if answer_contradiction_rate is not None else 0.0

    # Penalize degenerate operating points: "always retrieve_more" can look good
    # on conflict-heavy sets but fails basic answerable sanity questions.
    return (
        balanced_utility
        - 0.20 * max(0.0, 0.40 - answer_acc)
        - 0.10 * max(0.0, 0.40 - retrieve_acc)
        - 0.10 * abstain_rate
        - 0.15 * answer_contra
    )


def replay(
    rows: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    question_ids_by_query: dict[str, str],
    epi_sources: list[str],
    epi_thresholds: list[float],
    ale_thresholds: list[float],
    policy: str,
    uncertainty_formula: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    for epi_source in epi_sources:
        for epi_threshold in epi_thresholds:
            for ale_threshold in ale_thresholds:
                action_counts: Counter[str] = Counter()
                buckets: dict[str, dict[str, list[float]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                u_epi_values: list[float] = []
                u_epi_stochastic_values: list[float] = []
                u_ale_values: list[float] = []
                utility_values: list[float] = []
                utilities_by_expected: dict[str, list[float]] = defaultdict(list)
                expected_counts: Counter[str] = Counter()
                correct_family_counts: Counter[str] = Counter()
                by_expected_action: dict[str, Counter[str]] = defaultdict(Counter)

                for row in rows:
                    stats = augment_stats_with_evidence_gate(
                        row.get("stats") or {},
                        row.get("evidence_sampling_gate") or {},
                    )
                    qid = str(row.get("id") or "")
                    if not qid:
                        qid = question_ids_by_query.get(str(row.get("query") or "").strip(), "")
                    label = labels_by_id.get(qid, {})
                    metrics = compute_shadow_uncertainty(
                        stats,
                        formula=uncertainty_formula,
                    )
                    u_epi = float(metrics["u_epi"])
                    u_epi_stochastic = compute_shadow_epistemic(
                        stats=stats,
                        u_epi_baseline=u_epi,
                        epi_source=epi_source,
                    )
                    u_ale = float(metrics["u_ale"])
                    action = decide_shadow_action_2d(
                        u_epi=u_epi_stochastic,
                        u_ale=u_ale,
                        epi_threshold=epi_threshold,
                        ale_threshold=ale_threshold,
                        policy=policy,
                    )

                    action_counts[action] += 1
                    if label:
                        expected = expected_action_family(label)
                        expected_counts[expected] += 1
                        by_expected_action[expected][action] += 1
                        if action == expected:
                            correct_family_counts[expected] += 1
                        utility = action_utility(action, label)
                        utility_values.append(utility)
                        utilities_by_expected[expected].append(utility)
                    update_stats(buckets[action], stats)
                    if action != "abstain":
                        update_stats(buckets["non_abstain"], stats)
                    u_epi_values.append(u_epi)
                    u_epi_stochastic_values.append(u_epi_stochastic)
                    u_ale_values.append(u_ale)

                total = len(rows)
                answer_stats = summarize(buckets["answer"])
                retrieve_stats = summarize(buckets["retrieve_more"])
                abstain_stats = summarize(buckets["abstain"])
                non_abstain_stats = summarize(buckets["non_abstain"])
                expected_accuracy = {
                    key: (
                        correct_family_counts[key] / expected_counts[key]
                        if expected_counts[key]
                        else 0.0
                    )
                    for key in expected_counts
                }
                balanced_utility = (
                    sum(
                        sum(values) / len(values)
                        for values in utilities_by_expected.values()
                        if values
                    )
                    / len(utilities_by_expected)
                    if utilities_by_expected
                    else None
                )
                answer_rate = action_counts["answer"] / total if total else 0.0
                retrieve_more_rate = action_counts["retrieve_more"] / total if total else 0.0
                abstain_rate = action_counts["abstain"] / total if total else 0.0
                answer_contra = answer_stats.get("contradiction_rate")

                runs.append(
                    {
                        "epi_source": epi_source,
                        "epi_threshold": epi_threshold,
                        "ale_threshold": ale_threshold,
                        "policy": policy,
                        "uncertainty_formula": uncertainty_formula,
                        "total": total,
                        "actions": dict(action_counts),
                        "answer_rate": answer_rate,
                        "retrieve_more_rate": retrieve_more_rate,
                        "abstain_rate": abstain_rate,
                        "answer_contradiction_rate": answer_contra,
                        "retrieve_contradiction_rate": retrieve_stats.get("contradiction_rate"),
                        "abstain_contradiction_rate": abstain_stats.get("contradiction_rate"),
                        "non_abstain_contradiction_rate": non_abstain_stats.get("contradiction_rate"),
                        "answer_uncertainty_mean": answer_stats.get("uncertainty_mean"),
                        "label_aware_utility": (
                            sum(utility_values) / len(utility_values)
                            if utility_values
                            else None
                        ),
                        "balanced_label_aware_utility": balanced_utility,
                        "operating_score": operating_score(
                            balanced_utility=balanced_utility,
                            expected_accuracy=expected_accuracy,
                            abstain_rate=abstain_rate,
                            answer_contradiction_rate=answer_contra,
                        ),
                        "expected_action_counts": dict(expected_counts),
                        "expected_action_accuracy": expected_accuracy,
                        "actions_by_expected": {
                            key: dict(value)
                            for key, value in by_expected_action.items()
                        },
                        "u_epi_mean": sum(u_epi_values) / len(u_epi_values) if u_epi_values else 0.0,
                        "u_epi_stochastic_mean": (
                            sum(u_epi_stochastic_values) / len(u_epi_stochastic_values)
                            if u_epi_stochastic_values
                            else 0.0
                        ),
                        "u_ale_mean": sum(u_ale_values) / len(u_ale_values) if u_ale_values else 0.0,
                    }
                )

    def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        utility = item.get("operating_score")
        if utility is None:
            utility = item.get("balanced_label_aware_utility")
        if utility is None:
            utility = item.get("label_aware_utility")
        if utility is not None:
            return (
                -float(utility),
                float(item.get("abstain_rate") or 0.0),
                float(item.get("answer_contradiction_rate") or 1.0),
                -float(item.get("answer_rate") or 0.0),
            )
        answer_contra = item.get("answer_contradiction_rate")
        if answer_contra is None:
            answer_contra = 1.0
        return (
            float(answer_contra),
            float(item.get("abstain_rate") or 0.0),
            -float(item.get("answer_rate") or 0.0),
            float(item.get("non_abstain_contradiction_rate") or 1.0),
        )

    runs.sort(key=rank_key)
    return {
        "total": len(rows),
        "labels_loaded": len(labels_by_id),
        "policy": policy,
        "uncertainty_formula": uncertainty_formula,
        "epi_sources": epi_sources,
        "epi_thresholds": epi_thresholds,
        "ale_thresholds": ale_thresholds,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay stochastic FinReg gate candidates from eval dump-details JSONL."
    )
    parser.add_argument("--input", required=True, help="Path to --dump-details JSONL.")
    parser.add_argument(
        "--labels",
        default="",
        help="Optional question labels JSONL for label-aware action utility.",
    )
    parser.add_argument(
        "--questions",
        default="",
        help="Optional questions JSONL used to recover ids when dump-details lacks id.",
    )
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    parser.add_argument(
        "--epi-sources",
        nargs="+",
        default=DEFAULT_EPI_SOURCES,
        help="Epistemic adapters to replay.",
    )
    parser.add_argument(
        "--epi-thresholds",
        default="0.01,0.02,0.03,0.05",
        help="Comma-separated epistemic thresholds.",
    )
    parser.add_argument(
        "--ale-thresholds",
        default="0.25,0.30,0.35,0.40",
        help="Comma-separated aleatoric thresholds.",
    )
    parser.add_argument(
        "--shadow-policy",
        default="epi_coupled_v2",
        choices=["legacy_ale_dominant", "epi_coupled_v1", "epi_coupled_v2"],
    )
    parser.add_argument(
        "--shadow-uncertainty-formula",
        default="v2_conflict_aware",
        choices=["v2_conflict_aware", "v3_decoupled"],
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    labels_by_id = load_labels(args.labels)
    question_ids_by_query = load_question_ids(args.questions)
    report = replay(
        rows=rows,
        labels_by_id=labels_by_id,
        question_ids_by_query=question_ids_by_query,
        epi_sources=args.epi_sources,
        epi_thresholds=parse_float_list(args.epi_thresholds),
        ale_thresholds=parse_float_list(args.ale_thresholds),
        policy=args.shadow_policy,
        uncertainty_formula=args.shadow_uncertainty_formula,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
