#!/usr/bin/env python3
"""Replay shadow policies over FinReg evidence-subset sampling outputs.

The input is produced by scripts/eval_finreg_evidence_sampling_shadow.py. This
script does not rerun RAG; it converts subset action/risk distributions into a
candidate final action and compares it with the baseline gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rag.evidence_sampling_policy import decide_policy as shared_decide_policy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_labels(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {
        str(row.get("id")): row
        for row in read_jsonl(Path(path))
        if row.get("id")
    }


def expected_action_family(label: dict[str, Any], question_type: str) -> str:
    task_family = str(label.get("task_family") or "")
    support_level = str(label.get("support_level") or "")
    ambiguity_family = str(label.get("ambiguity_family") or "")

    if question_type == "sanity":
        return "answer"
    if task_family == "sanity_anchor" and ambiguity_family in ("none_or_low", ""):
        return "answer"
    if support_level.endswith("_hard"):
        return "retrieve_more"
    if task_family in ("comparison_conflict", "supervisory_escalation"):
        return "retrieve_more"
    return "answer"


def canonical_action(action: str) -> str:
    return "answer" if action in ("none", "answer") else action


def action_utility(action: str, expected: str, question_type: str) -> float:
    action = canonical_action(action)
    if expected == "answer":
        return {"answer": 1.0, "retrieve_more": 0.45, "abstain": 0.0}.get(action, 0.0)

    if action == "retrieve_more":
        return 1.0
    if action == "abstain":
        return 0.65 if question_type == "conflict" else 0.45
    if action == "answer":
        return 0.35
    return 0.0


def decide_policy(row: dict[str, Any], policy: str) -> tuple[str, str]:
    qtype = str(row.get("type") or "")
    baseline = str(row.get("baseline_action") or "none")

    if row.get("candidate_unavailable"):
        return baseline, "candidate_unavailable"

    answer_rate = float(row.get("subset_answer_rate") or 0.0)
    non_answer_rate = float(row.get("subset_non_answer_rate") or 0.0)
    instability = float(row.get("subset_action_instability") or 0.0)
    risk_mean = float(row.get("answer_include_risk_mean_across_subsets") or 0.0)
    risk_max = float(row.get("answer_include_risk_max_across_subsets") or 0.0)

    if policy == "baseline":
        return baseline, "baseline"

    if policy == "subset_majority":
        if non_answer_rate > answer_rate:
            return "retrieve_more", "subset_majority_non_answer"
        return "none", "subset_majority_answer"

    if policy == "guarded_v1":
        # Guard sanity questions against over-conservatism unless subset risk is
        # both frequent and high. Conflict questions can use instability as a
        # stronger retrieve_more signal.
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.75 and risk_mean >= 0.66:
            return "abstain", "conflict_stable_high_risk"
        if non_answer_rate >= 0.50 or instability >= 0.50:
            return "retrieve_more", "conflict_unstable_or_non_answer"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v2":
        # More conservative on conflict questions, but still protects sanity.
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            if non_answer_rate >= 0.50 and risk_max >= 0.68:
                return "retrieve_more", "sanity_high_risk_tie"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 1.0 and risk_mean >= 0.66:
            return "abstain", "conflict_all_subsets_risky"
        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_non_answer_majority_or_tie"
        if instability >= 0.50 and risk_max >= 0.64:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v3":
        # First-pass sequential policy:
        # - subset instability/high risk can request more evidence
        # - it should not abstain immediately because risk may still be reducible
        # - final abstain should be handled by a later evidence-budget stage
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_subset_non_answer"
        if instability >= 0.50 and risk_max >= 0.64:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    raise ValueError(f"Unknown policy: {policy}")


def replay(
    rows: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    policies: list[str],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    for policy in policies:
        actions: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        expected_counts: Counter[str] = Counter()
        correct_counts: Counter[str] = Counter()
        utility_values: list[float] = []
        utilities_by_expected: dict[str, list[float]] = defaultdict(list)
        by_expected: dict[str, Counter[str]] = defaultdict(Counter)
        by_type: dict[str, Counter[str]] = defaultdict(Counter)
        changes: list[dict[str, Any]] = []

        for row in rows:
            qid = str(row.get("id") or "")
            qtype = str(row.get("type") or "")
            label = labels_by_id.get(qid, {})
            expected = expected_action_family(label, qtype)
            action, reason = shared_decide_policy(row, policy)
            baseline = str(row.get("baseline_action") or "none")

            actions[action] += 1
            reasons[reason] += 1
            expected_counts[expected] += 1
            by_expected[expected][canonical_action(action)] += 1
            by_type[qtype][canonical_action(action)] += 1
            if canonical_action(action) == expected:
                correct_counts[expected] += 1

            utility = action_utility(action, expected, qtype)
            utility_values.append(utility)
            utilities_by_expected[expected].append(utility)

            if action != baseline:
                changes.append(
                    {
                        "id": qid,
                        "type": qtype,
                        "expected": expected,
                        "baseline_action": baseline,
                        "policy_action": action,
                        "reason": reason,
                        "subset_actions": row.get("subset_actions"),
                        "subset_action_instability": row.get("subset_action_instability"),
                        "subset_non_answer_rate": row.get("subset_non_answer_rate"),
                        "answer_include_risk_max_across_subsets": row.get(
                            "answer_include_risk_max_across_subsets"
                        ),
                        "query": row.get("query"),
                    }
                )

        expected_accuracy = {
            key: correct_counts[key] / expected_counts[key]
            for key in expected_counts
        }
        balanced_utility = (
            sum(sum(values) / len(values) for values in utilities_by_expected.values())
            / len(utilities_by_expected)
            if utilities_by_expected
            else None
        )
        utility = sum(utility_values) / len(utility_values) if utility_values else None
        runs.append(
            {
                "policy": policy,
                "total": len(rows),
                "actions": dict(actions),
                "action_rates": {
                    key: value / len(rows) if rows else 0.0
                    for key, value in actions.items()
                },
                "reasons": dict(reasons),
                "expected_action_counts": dict(expected_counts),
                "expected_action_accuracy": expected_accuracy,
                "actions_by_expected": {
                    key: dict(value)
                    for key, value in by_expected.items()
                },
                "actions_by_type": {
                    key: dict(value)
                    for key, value in by_type.items()
                },
                "label_aware_utility": utility,
                "balanced_label_aware_utility": balanced_utility,
                "changes_from_baseline": len(changes),
                "changes": changes,
            }
        )

    return {"total": len(rows), "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay evidence-sampling shadow policies.")
    parser.add_argument("--input", required=True, help="Evidence sampling shadow JSON.")
    parser.add_argument("--labels", default="", help="Optional FinReg labels JSONL.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--policies",
        default=(
            "baseline,subset_majority,guarded_v1,guarded_v2,"
            "guarded_v3,guarded_v4,guarded_v4_lite"
        ),
        help="Comma-separated policies.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    report = replay(rows, load_labels(args.labels), policies)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    for run in report["runs"]:
        print(
            run["policy"],
            "actions=",
            run["actions"],
            "balanced_utility=",
            run["balanced_label_aware_utility"],
            "changes=",
            run["changes_from_baseline"],
        )


if __name__ == "__main__":
    main()
